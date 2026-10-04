import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import click

logger = logging.getLogger(__name__)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
LF_COLOR = "#2a78d6"
HF_COLOR = "#eb6834"


def load_routing_snapshots(file_folder: str) -> dict:
    """Loads gate_routing_snapshots.npz, written by AdaptiveGridSearch.
    save_routing_snapshots() after main.py's FE reruns. Most arrays are shape
    (n_snapshots, ...) where n_snapshots = n_reruns * len(usage_values); 'rerun'
    and 'target_usage' (both length n_snapshots) identify each row. The
    exceptions are lf_latent/latent_2d/boundary_xx/boundary_yy, which are
    identical across every snapshot (same fixed test-set lf_latent, same
    cached projection basis/grid) and so are saved once, unstacked."""
    path = os.path.join(file_folder, "gate_routing_snapshots.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. This file is written by AdaptiveGridSearch."
            "save_routing_snapshots() after main.py's FE reruns — rerun main.py "
            "to produce it (older result dirs predate this)."
        )
    data = np.load(path)
    return {key: data[key] for key in data.files}


def compute_ground_truth(lf_correct: np.ndarray, hf_correct: np.ndarray):
    """HF needed = HF gets it right and LF doesn't (escalating actually helps).
    LF fine = HF wrong OR (HF correct AND LF correct) — the exact logical
    complement of "HF needed", so every sample falls in exactly one bucket."""
    hf_needed = hf_correct & ~lf_correct
    lf_fine = ~hf_needed
    return hf_needed, lf_fine


def compute_routing_counts(choice: np.ndarray, hf_needed: np.ndarray, lf_fine: np.ndarray) -> dict:
    routed_hf = choice == 1
    routed_lf = ~routed_hf
    return {
        "tp": int(np.sum(routed_hf & hf_needed)),  # HF used, HF needed
        "fp": int(np.sum(routed_hf & lf_fine)),    # HF used, LF fine (wasted HF call)
        "fn": int(np.sum(routed_lf & hf_needed)),  # LF used, HF needed (missed escalation)
        "tn": int(np.sum(routed_lf & lf_fine)),    # LF used, LF fine (correctly kept cheap)
    }


def compute_dense_oracle_curve(
        lf_correct: np.ndarray, hf_correct: np.ndarray, grid_step_pct: float = 0.5,
        lf_pixel_acc: np.ndarray | None = None, hf_pixel_acc: np.ndarray | None = None,
        clip_negative_gain: bool = True):
    """Best achievable accuracy (%) at every possible usage budget (not just a
    sparse set of usage_values targets), escalating samples in order of gain -
    the greedy, provably cost-optimal policy. Rank once by gain descending,
    take one cumulative sum over that ranking, and read off the oracle
    accuracy at prefix length k (usage=k/N) for every k=0..N in a single
    O(N log N) pass - then subsample to an evenly-spaced grid for plotting.
    Replaces the old per-usage-value loop (compute_oracle_curve), which only
    ever produced ~10 points and so only spanned a narrow x-range instead of
    the full 0-100% usage axis.

    Candidates for escalation are ranked by gain = hf_pixel_acc - lf_pixel_acc
    (not the boolean per-image-majority lf_correct/hf_correct) - this is the
    strict generalization of "HF needed" to continuous accuracy: with
    0/1-valued lf_pixel_acc/hf_pixel_acc it reduces to exactly
    hf_correct & ~lf_correct, but with real per-pixel fractions it also
    (correctly) catches e.g. a sample where HF gets more pixels right without
    crossing the same >50% majority threshold LF also fell short of. Using
    the boolean partition instead can push the "oracle" *below* HF's own
    plain accuracy - not a valid ceiling - since it would never escalate a
    sample sitting in the boolean "LF fine" bucket even when HF's continuous
    accuracy on it is higher.

    Passing lf_pixel_acc=hf_pixel_acc=None (the default) reproduces the
    boolean-only formula exactly - this is a strict generalization, not a
    behavior change, when the inputs happen to be 0/1-valued.

    clip_negative_gain controls what happens once usage exceeds the fraction
    of samples that actually benefit from HF:
      - True (default): clip negative gain to 0, so a budget large enough to
        reach harmful samples never pulls the curve down - "never voluntarily
        escalate a sample that hurts," a monotonic, true budget-constrained
        ceiling. Used for the main pareto.png.
      - False: no clipping - once every helpful sample is covered, the curve
        is forced to keep escalating (since usage is an exact fraction, not
        an upper bound), including harmful ones, so it can decline after its
        peak and is guaranteed to end exactly at HF-alone accuracy at
        usage=100%. Matches the original dense-oracle script this was first
        prototyped in (historical/cub_oracle_pareto_update.py) - used for the
        scatter pareto plot (pareto_scatter.png), where the declining tail is
        informative (it shows the literal cost of being forced to escalate
        strictly by rank past the point where it stops helping).

    Returns (usage_pct_grid, oracle_acc_pct_grid), both length
    round(100/grid_step_pct)+1, evenly spaced over [0, 100]."""
    if lf_pixel_acc is None or hf_pixel_acc is None:
        lf_pixel_acc = lf_correct.astype(float)
        hf_pixel_acc = hf_correct.astype(float)
    else:
        lf_pixel_acc = lf_pixel_acc.astype(float)
        hf_pixel_acc = hf_pixel_acc.astype(float)
    n = len(lf_pixel_acc)

    gain = hf_pixel_acc - lf_pixel_acc
    order = np.argsort(-gain)  # descending: highest-gain samples escalated first
    sorted_gain = gain[order]
    if clip_negative_gain:
        # Samples with gain <= 0 are never worth escalating - clip their
        # contribution to 0 so a budget large enough to reach them doesn't pull
        # the oracle curve down below its own peak (matching compute_oracle_curve's
        # old "never escalate gain<=0 regardless of leftover budget" behavior).
        sorted_gain = np.clip(sorted_gain, 0, None)

    lf_acc = lf_pixel_acc.mean()  # usage=0 (k=0): nobody escalated, exactly LF-alone accuracy
    cum_gain_mean = np.cumsum(sorted_gain) / n  # mean gain contributed by the top-k samples, k=1..N
    oracle_acc_per_k = lf_acc + cum_gain_mean  # (N,), index k-1 = usage (k/n)

    # Prepend k=0 (usage=0, exactly lf_acc) so the curve starts at the true origin.
    oracle_acc_per_k = np.concatenate([[lf_acc], oracle_acc_per_k])  # (N+1,)

    # Dense, evenly-spaced plotting grid - nearest available k for each target.
    grid_usage_pct = np.arange(0, 100 + grid_step_pct, grid_step_pct)
    grid_k = np.clip(np.round(grid_usage_pct / 100 * n).astype(int), 0, n)
    grid_oracle_acc_pct = oracle_acc_per_k[grid_k] * 100

    return grid_usage_pct, grid_oracle_acc_pct


def _style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)
    ax.tick_params(colors=INK_MUTED, labelsize=7)


def _grid_shape(n, max_cols=5):
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def plot_routing_projection_grid(snapshots: dict, rerun_idx: int, save_path: str) -> None:
    """One panel per usage target (for a fixed rerun): a 2D scatter of
    lf_latent, colored by the gate's routing decision, marker-shaped by
    ground-truth need. The 2D plane is a supervised linear projection (fit in
    AdaptiveGridSearch._project_latent) — axis 1 is a logistic-regression
    direction separating hf_needed from lf_fine, axis 2 is the top PCA
    direction of the leftover variance — rather than plain unsupervised PCA,
    since PCA's top-variance directions have no reason to align with where the
    gate actually needs to escalate.

    The contour is *not* a proxy classifier re-fit on the 2D points — it's
    boundary_zz, computed in AdaptiveGridSearch by running the actual trained
    fe_model over this same plane's grid (reconstructed back to the real
    latent dimensionality via the projection basis's transpose). So it
    reflects that model's own nonlinearity. It's still a slice/approximation,
    since the plane can't capture the other latent_dim-2 axes — but it's the
    real model's slice, not an independently-fit stand-in.

    Assumes lf_latent is a flat (N, D) vector per sample (the classification FE
    model) — not applicable as-is to the segmentation "cnn_head" case.
    """
    mask = snapshots["rerun"] == rerun_idx
    target_usages = sorted(set(snapshots["target_usage"][mask].tolist()))

    coords = snapshots["latent_2d"]
    xx = snapshots["boundary_xx"]
    yy = snapshots["boundary_yy"]

    nrows, ncols = _grid_shape(len(target_usages))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    axes = np.atleast_1d(axes).ravel()

    for i, usage in enumerate(target_usages):
        idx = np.where((snapshots["rerun"] == rerun_idx) & (snapshots["target_usage"] == usage))[0][0]
        choice = snapshots["choice"][idx]
        hf_correct = snapshots["hf_correct"][idx]
        lf_correct = snapshots["lf_correct"][idx]
        hf_needed, lf_fine = compute_ground_truth(lf_correct, hf_correct)
        zz = snapshots["boundary_zz"][idx]

        ax = axes[i]
        _style_axes(ax)

        ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=[LF_COLOR, HF_COLOR], alpha=0.12, zorder=1)
        ax.contour(xx, yy, zz, levels=[0.5], colors=[INK_SECONDARY], linewidths=1, zorder=2)

        for need_mask, marker in [(hf_needed, "^"), (lf_fine, "o")]:
            for routed_mask, color in [(choice == 0, LF_COLOR), (choice == 1, HF_COLOR)]:
                sel = need_mask & routed_mask
                if sel.any():
                    ax.scatter(coords[sel, 0], coords[sel, 1], c=color, marker=marker,
                               s=14, alpha=0.65, linewidths=0, zorder=3)

        ax.set_title(f"usage={usage:.1f}", fontsize=9, color=INK_PRIMARY)

    for j in range(len(target_usages), len(axes)):
        axes[j].axis("off")

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LF_COLOR, label="Routed LF", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HF_COLOR, label="Routed HF", markersize=8),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=INK_MUTED, label="HF needed", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INK_MUTED, label="LF fine", markersize=8),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"Gate routing in supervised projection of lf_latent — rerun {rerun_idx}", color=INK_PRIMARY, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved routing projection plot (rerun {rerun_idx}) to {save_path}")


def plot_routing_table_grid(snapshots: dict, rerun_idx: int, save_path: str) -> None:
    """One 2x2 panel per usage target (for a fixed rerun): rows = routing
    decision (HF used / LF used), columns = ground-truth need (HF needed / LF
    fine) — a confusion matrix for the routing decision against the oracle."""
    mask = snapshots["rerun"] == rerun_idx
    target_usages = sorted(set(snapshots["target_usage"][mask].tolist()))

    nrows, ncols = _grid_shape(len(target_usages))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    axes = np.atleast_1d(axes).ravel()

    for i, usage in enumerate(target_usages):
        idx = np.where((snapshots["rerun"] == rerun_idx) & (snapshots["target_usage"] == usage))[0][0]
        hf_correct = snapshots["hf_correct"][idx]
        lf_correct = snapshots["lf_correct"][idx]
        choice = snapshots["choice"][idx]
        hf_needed, lf_fine = compute_ground_truth(lf_correct, hf_correct)
        counts = compute_routing_counts(choice, hf_needed, lf_fine)

        table = np.array([[counts["tp"], counts["fp"]],
                           [counts["fn"], counts["tn"]]])

        ax = axes[i]
        ax.imshow(table, cmap="Blues", aspect="auto", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["HF needed", "LF fine"], fontsize=7, color=INK_SECONDARY)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["HF used", "LF used"], fontsize=7, color=INK_SECONDARY)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        half_max = table.max() / 2 if table.max() > 0 else 1
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{table[r, c]}", ha="center", va="center", fontsize=10,
                        color="white" if table[r, c] > half_max else INK_PRIMARY)

        ax.set_title(f"usage={usage:.1f}", fontsize=9, color=INK_PRIMARY)

    for j in range(len(target_usages), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Routing decision vs ground-truth need — rerun {rerun_idx}", color=INK_PRIMARY, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved routing table plot (rerun {rerun_idx}) to {save_path}")


@click.command()
@click.option("--results_folder", required=True, help="Experiment result dir, e.g. results/bird_grayscale-resnet-resnet-42-128")
def main(results_folder):
    results_folder = os.path.normpath(results_folder)
    file_folder = os.path.join(results_folder, "files")
    image_folder = os.path.join(results_folder, "images")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    snapshots = load_routing_snapshots(file_folder)
    for rerun_idx in sorted(set(snapshots["rerun"].tolist())):
        plot_routing_projection_grid(snapshots, rerun_idx, os.path.join(image_folder, f"gate_routing_projection_rerun{rerun_idx}.png"))
        plot_routing_table_grid(snapshots, rerun_idx, os.path.join(image_folder, f"gate_routing_table_rerun{rerun_idx}.png"))


if __name__ == "__main__":
    main()
