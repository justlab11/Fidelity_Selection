import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import click

logger = logging.getLogger(__name__)

# Fixed categorical color order (never cycled) so a baseline always maps to the
# same color across every plot this script produces. "alpha" controls marker/
# line opacity in render_pareto_plot/render_pareto_scatter_plot - FE/FE+SR
# stay fully opaque (they're the primary curves these two plots were built
# for), while the four comparison baselines (SR/SelectiveNet x2/SAT) are
# drawn more transparent so they read as secondary context without
# obscuring FE vs. FE+SR's own distinguishability from each other.
BASELINE_STYLE = {
    "FE":                        {"file": "fe_results.npz",                     "color": "#2a78d6", "alpha": 1.0},
    "FE+SR":                     {"file": "fe_sr_results.npz",                  "color": "#8b5cf6", "alpha": 1.0},
    "SR":                        {"file": "sr_results.npz",                     "color": "#eb6834", "alpha": 0.4},
    # "SelectiveNet" (file: selectivenet_results.npz) is the native/
    # paper-faithful (fixed threshold=0.5) variant - that filename is kept
    # stable for back-compat with anything that only knows this one file.
    "SelectiveNet (native)":     {"file": "selectivenet_results.npz",           "color": "#1baf7a", "alpha": 0.4},
    "SelectiveNet (calibrated)": {"file": "selectivenet_calibrated_results.npz","color": "#7c3aed", "alpha": 0.4},
    "SAT":                       {"file": "sat_results.npz",                    "color": "#eda100", "alpha": 0.4},
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
RANDOM_BASELINE_COLOR = "#d6453d"


def load_pareto_curves(file_folder: str) -> dict:
    """Loads whichever of {fe,fe_sr,sr,selectivenet,sat}_results.npz exist under
    file_folder. Each has 'usage'/'acc' arrays on a 0-100 scale. Missing files
    are skipped (with a warning) so partial overlays still render.

    fe_results.npz additionally carries 'acc_std'/'usage_std' (across
    fe_training.reruns repeats of the FE search) — zeros when reruns=1. The
    other baselines don't have reruns, so their std defaults to zeros."""
    curves = {}

    for label, style in BASELINE_STYLE.items():
        path = os.path.join(file_folder, style["file"])
        if not os.path.exists(path):
            logger.warning(f"{label}: '{path}' not found, skipping")
            continue

        data = np.load(path)
        usage = np.asarray(data["usage"], dtype=float)
        acc = np.asarray(data["acc"], dtype=float)
        acc_std = np.asarray(data["acc_std"], dtype=float) if "acc_std" in data.files else np.zeros_like(acc)
        usage_std = np.asarray(data["usage_std"], dtype=float) if "usage_std" in data.files else np.zeros_like(usage)

        order = np.argsort(usage)
        curves[label] = (usage[order], acc[order], acc_std[order], usage_std[order])

    return curves


def render_pareto_plot(curves: dict, dataset_label: str, save_path: str, oracle: tuple | None = None,
                        oracle_label: str = "Oracle (ceiling)") -> None:
    """oracle, if given, is (usage, acc) on the same 0-100 scale as the other
    curves — the best achievable accuracy at each usage budget when escalation
    is spent only on samples that actually need it (the budget-constrained
    greedy ranking - top-usage% by benefit). Drawn as a dashed reference line
    above every real method. A dense oracle (fine-grained usage grid) will
    rise from exactly LF-alone accuracy at usage=0 to exactly HF-alone
    accuracy at usage=100, flattening in between once every positive-gain
    sample is already covered - no separate "unconstrained ceiling" line is
    needed, since that ceiling is just this same curve's own value at
    usage=100."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.grid(True, color=GRIDLINE, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)

    if oracle is not None:
        oracle_usage, oracle_acc = oracle
        ax.plot(
            oracle_usage, oracle_acc,
            color=INK_MUTED, linewidth=1.5, linestyle="--",
            label=oracle_label, zorder=2
        )

    for label, style in BASELINE_STYLE.items():
        if label not in curves:
            continue
        usage, acc, acc_std, _usage_std = curves[label]
        color = mcolors.to_rgba(style["color"], alpha=style.get("alpha", 1.0))

        if np.any(acc_std > 0):
            # vertical error bars only (±1 std of test accuracy across reruns at
            # each usage_values target) — usage itself also varies across reruns,
            # but we only asked for/plot the accuracy spread here
            ax.errorbar(
                usage, acc, yerr=acc_std,
                color=color, linewidth=2, marker="o", markersize=6,
                markerfacecolor=color, markeredgewidth=0,
                ecolor=color, elinewidth=1.5, capsize=4,
                label=label, zorder=3
            )
        else:
            ax.plot(
                usage, acc,
                color=color, linewidth=2, marker="o", markersize=6,
                markerfacecolor=color, markeredgewidth=0,
                label=label, zorder=3
            )

    ax.set_xlabel("Usage (% routed to HF)", color=INK_SECONDARY)
    ax.set_ylabel("Accuracy (%)", color=INK_PRIMARY)
    ax.set_title(dataset_label, color=INK_PRIMARY, fontweight="bold")
    ax.tick_params(colors=INK_MUTED)

    if curves:
        ax.legend(frameon=False, labelcolor=INK_PRIMARY)
    else:
        logger.warning(f"No baseline result files found for '{dataset_label}' — plot will be empty")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved Pareto plot to {save_path}")


def render_pareto_scatter_plot(curves: dict, dataset_label: str, save_path: str, oracle: tuple | None = None,
                                oracle_label: str = "Oracle (ceiling)",
                                random_baseline: tuple | None = None) -> None:
    """Same data as render_pareto_plot, but FE/FE+SR etc. are drawn as
    unconnected scatter + error-bar points (both directions - usage_std as
    well as acc_std, when available) rather than a connected line - the style
    the old per-dataset 5-rerun scripts (e.g. historical/cub_pareto_with_std.py)
    used before render_pareto_plot's connected-line style became the default.

    oracle is the same (usage, acc) dense curve as render_pareto_plot's,
    typically with clip_negative_gain=False (see compute_dense_oracle_curve) -
    that variant's declining tail past its peak is the point of pairing it
    with this plot specifically.

    random_baseline, if given, is (lf_acc_pct, hf_acc_pct) - LF-alone accuracy
    at usage=0 and HF-alone accuracy at usage=100, connected by a dotted
    line. This is the expected accuracy of routing a fraction p of samples to
    HF uniformly at random (no gate at all): (1-p)*lf_acc + p*hf_acc is
    exactly the straight line between these two points, so it's drawn as one
    - a "no-intelligence" reference distinct from the oracle (best case)."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.grid(True, color=GRIDLINE, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)

    if oracle is not None:
        oracle_usage, oracle_acc = oracle
        ax.plot(
            oracle_usage, oracle_acc,
            color=INK_MUTED, linewidth=1.5, linestyle="--",
            label=oracle_label, zorder=2
        )

    if random_baseline is not None:
        lf_acc_pct, hf_acc_pct = random_baseline
        ax.plot(
            [0, 100], [lf_acc_pct, hf_acc_pct],
            color=RANDOM_BASELINE_COLOR, linewidth=1.5, linestyle=":",
            label="Random routing (expected)", zorder=2
        )

    for label, style in BASELINE_STYLE.items():
        if label not in curves:
            continue
        usage, acc, acc_std, usage_std = curves[label]
        base_alpha = style.get("alpha", 1.0)

        ax.errorbar(
            usage, acc, yerr=acc_std, xerr=usage_std,
            fmt="o", linestyle="none", color=mcolors.to_rgba(style["color"], alpha=base_alpha), markersize=10,
            markerfacecolor=mcolors.to_rgba(style["color"], alpha=0.65 * base_alpha),
            markeredgecolor=mcolors.to_rgba(style["color"], alpha=base_alpha), markeredgewidth=1.2,
            ecolor=mcolors.to_rgba(style["color"], alpha=base_alpha), elinewidth=1.5, capsize=4,
            label=label, zorder=3
        )

    ax.set_xlabel("Usage (% routed to HF)", color=INK_SECONDARY)
    ax.set_ylabel("Accuracy (%)", color=INK_PRIMARY)
    ax.set_title(dataset_label, color=INK_PRIMARY, fontweight="bold")
    ax.tick_params(colors=INK_MUTED)

    if curves:
        ax.legend(frameon=False, labelcolor=INK_PRIMARY)
    else:
        logger.warning(f"No baseline result files found for '{dataset_label}' — plot will be empty")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved Pareto scatter plot to {save_path}")


def plot_one_dataset(results_folder: str, output_folder: str | None) -> None:
    results_folder = os.path.normpath(results_folder)
    file_folder = os.path.join(results_folder, "files")
    dataset_label = os.path.basename(results_folder).split("-")[0]

    curves = load_pareto_curves(file_folder)
    save_folder = output_folder if output_folder else os.path.join(results_folder, "images")

    oracle = None
    oracle_uncapped = None
    random_baseline = None
    try:
        from plot_gate_routing import load_routing_snapshots, compute_dense_oracle_curve
        snapshots = load_routing_snapshots(file_folder)
        lf_pixel_acc = snapshots.get("lf_pixel_acc", [None])[0]
        hf_pixel_acc = snapshots.get("hf_pixel_acc", [None])[0]
        lf_correct = snapshots["lf_correct"][0]
        hf_correct = snapshots["hf_correct"][0]

        oracle = compute_dense_oracle_curve(
            lf_correct, hf_correct, lf_pixel_acc=lf_pixel_acc, hf_pixel_acc=hf_pixel_acc
        )
        oracle_uncapped = compute_dense_oracle_curve(
            lf_correct, hf_correct, lf_pixel_acc=lf_pixel_acc, hf_pixel_acc=hf_pixel_acc,
            clip_negative_gain=False
        )
        lf_acc_pct = 100 * (lf_pixel_acc.mean() if lf_pixel_acc is not None else lf_correct.astype(float).mean())
        hf_acc_pct = 100 * (hf_pixel_acc.mean() if hf_pixel_acc is not None else hf_correct.astype(float).mean())
        random_baseline = (lf_acc_pct, hf_acc_pct)
    except FileNotFoundError:
        logger.warning(f"No gate_routing_snapshots.npz under '{file_folder}' — skipping oracle curve")

    render_pareto_plot(
        curves, dataset_label, os.path.join(save_folder, "pareto.png"),
        oracle=oracle, oracle_label="Oracle (budget-constrained)"
    )
    render_pareto_scatter_plot(
        curves, dataset_label, os.path.join(save_folder, "pareto_scatter.png"),
        oracle=oracle_uncapped, oracle_label="Oracle (budget-constrained)",
        random_baseline=random_baseline
    )


@click.command()
@click.option(
    "--results_folder", "results_folders", multiple=True, required=True,
    help="Experiment result dir, e.g. results/mnist_noise-resnet-resnet-42-128. "
         "Pass this flag once per dataset to get one plot each."
)
@click.option(
    "--output_folder", default=None,
    help="Where to save PNGs; defaults to <results_folder>/images for each folder."
)
def main(results_folders, output_folder):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    for folder in results_folders:
        plot_one_dataset(folder, output_folder)


if __name__ == "__main__":
    main()
