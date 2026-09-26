import os
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import matplotlib.pyplot as plt
import click
from sklearn.isotonic import IsotonicRegression
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

SERIES_COLOR = "#2a78d6"
RAW_POINT_COLOR = "#898781"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"


def load_gate_log(file_folder: str):
    """Loads gate_search_log.npz, written by AdaptiveGridSearch.save_evaluated_points()
    after main.py's FE usage sweep. r/usage/acc are all raw fractions (0-1), not
    percentages — matching AdaptiveGridSearch.evaluated_points' own scale."""
    path = os.path.join(file_folder, "gate_search_log.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. This file is written by AdaptiveGridSearch."
            "save_evaluated_points() after main.py's FE usage sweep — rerun main.py "
            "to produce it (older result dirs predate this)."
        )
    data = np.load(path)
    return data["r_values"], data["usage_values"], data["acc_values"]


def load_test_benefit(latent_test_folder: str) -> np.ndarray:
    """Loads (lf_loss - hf_loss) for every sample in latent/test/ - the same
    per-sample values save_latent/compute_fidelity_loss_correct computed for
    gate training (helpers.py), read back here read-only (no retraining
    involved). "benefit" is exactly the quantity the Bayes-optimal gate policy
    thresholds against c_h (see fit_isotonic_usage's docstring: escalate iff
    lf_loss - hf_loss > c_h), so it's what compute_oracle_usage_curve and
    compute_oracle_usage_density need.

    Raises FileNotFoundError if latent_test_folder is missing (e.g. an older
    result dir, or one where train_body=True and the raw latent/ folder was
    never populated) - callers should treat that as "no oracle overlay
    available" rather than a hard failure, since every other plot here still
    works without it.
    """
    if not os.path.isdir(latent_test_folder):
        raise FileNotFoundError(
            f"'{latent_test_folder}' not found - can't compute the oracle usage(c_h) "
            "curve without the test split's per-sample lf_loss/hf_loss."
        )

    files = sorted(f for f in os.listdir(latent_test_folder) if f.endswith(".pt"))
    if not files:
        raise FileNotFoundError(f"'{latent_test_folder}' has no .pt files")

    def _read_benefit(fname):
        # Each file also carries lf_latent (e.g. 1024x14x14 for a cnn_head
        # gate) that this doesn't need, but torch.save/load has no partial-
        # read - reading and discarding it is unavoidable per file. What is
        # avoidable is doing those reads one at a time: this is the same
        # disk-I/O-bound situation num_workers addressed for the DataLoader
        # case (see main.py), so a thread pool (I/O-bound, so still
        # effective despite the GIL) is used here for the same reason.
        data = torch.load(os.path.join(latent_test_folder, fname), weights_only=True)
        return float(data["lf_loss"]) - float(data["hf_loss"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        benefit = np.fromiter(pool.map(_read_benefit, files), dtype=np.float64, count=len(files))
    return benefit


def compute_oracle_usage_curve(benefit: np.ndarray, c_h_grid: np.ndarray) -> np.ndarray:
    """Exact Bayes-optimal usage(c_h) = P(lf_loss - hf_loss > c_h), i.e.
    1 - ECDF(c_h) of the per-sample benefit distribution. This has nothing to
    do with the trained gate - it's the usage a perfect gate would achieve at
    each c_h, computed directly from the frozen LF/HF models' own losses, so
    it's exact and noise-free (unlike the empirical points, each of which is
    one independently retrained gate). Evaluate on a dense, evenly-spaced
    c_h_grid (not the sparse adaptively-sampled r values) for a smooth
    reference line.
    """
    benefit_sorted = np.sort(benefit)
    counts_le = np.searchsorted(benefit_sorted, c_h_grid, side="right")
    return 1.0 - counts_le / len(benefit_sorted)


def compute_oracle_usage_density(benefit: np.ndarray, c_h_grid: np.ndarray, bw_method=None) -> np.ndarray:
    """d(oracle_usage)/d(c_h) = -density(c_h): the negative KDE-estimated
    density of the benefit distribution at each c_h_grid point.

    Deliberately NOT a finite difference of compute_oracle_usage_curve at the
    empirical r_sorted points - those points come from an adaptive bisection
    search, so they cluster tightly in some regions and are sparse elsewhere;
    differencing at exactly those locations would make the "exact" oracle
    derivative just as spiky as the empirical one, for the uninteresting
    reason that the two curves would be sampled unevenly rather than because
    the underlying benefit distribution actually has that structure. A KDE
    (Scott's rule bandwidth by default via bw_method=None, the same default
    scipy.stats.gaussian_kde uses) gives a properly smoothed density estimate
    that doesn't depend on where c_h happened to get sampled.
    """
    kde = gaussian_kde(benefit, bw_method=bw_method)
    return -kde(c_h_grid)


def dedupe_and_sort(r: np.ndarray, usage: np.ndarray, acc: np.ndarray, round_decimals: int = 4):
    """Points come from multiple independent binary searches (one per target usage
    in main.py's usage sweep), so they aren't jointly ordered and can land near-but-
    not-exactly the same c_h. Group by rounded c_h, average within each group (using
    the group's true, unrounded c_h as the x-position), then sort ascending."""
    rounded = np.round(r, round_decimals)
    unique_vals, inverse = np.unique(rounded, return_inverse=True)

    r_sum = np.zeros(len(unique_vals))
    usage_sum = np.zeros(len(unique_vals))
    acc_sum = np.zeros(len(unique_vals))
    counts = np.zeros(len(unique_vals))

    np.add.at(r_sum, inverse, r)
    np.add.at(usage_sum, inverse, usage)
    np.add.at(acc_sum, inverse, acc)
    np.add.at(counts, inverse, 1)

    r_grouped = r_sum / counts
    usage_grouped = usage_sum / counts
    acc_grouped = acc_sum / counts

    order = np.argsort(r_grouped)
    return r_grouped[order], usage_grouped[order], acc_grouped[order]


def compute_usage_derivative(r_sorted: np.ndarray, usage_sorted: np.ndarray, smooth_window: int = 1):
    """d(usage)/d(c_h). np.gradient with an explicit coordinate array handles the
    non-uniform spacing correctly — adaptive binary search naturally clusters points
    densely near steep regions of the curve and sparsely elsewhere."""
    derivative = np.gradient(usage_sorted, r_sorted)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        derivative = np.convolve(derivative, kernel, mode="same")
    return derivative


def find_zoom_range(r_sorted: np.ndarray, usage_sorted: np.ndarray, low: float, high: float):
    """r range where usage crosses [low, high], via interpolation on the usage-sorted
    data. Also sanity-checks the monotonicity assumption AdaptiveGridSearch's own
    bisection relies on (usage non-increasing in c_h)."""
    diffs = np.diff(usage_sorted)
    frac_decreasing = float(np.mean(diffs <= 0)) if len(diffs) > 0 else 1.0
    if frac_decreasing < 0.8:
        logger.warning(
            f"Usage is not consistently non-increasing in c_h ({frac_decreasing:.0%} of "
            "consecutive sorted-by-c_h steps decrease) — the zoom window below may be unreliable."
        )

    usage_order = np.argsort(usage_sorted)
    usage_asc = usage_sorted[usage_order]
    r_by_usage_asc = r_sorted[usage_order]

    r_at_low = float(np.interp(low, usage_asc, r_by_usage_asc))
    r_at_high = float(np.interp(high, usage_asc, r_by_usage_asc))

    return min(r_at_low, r_at_high), max(r_at_low, r_at_high)


def report_max_usage(r_sorted: np.ndarray, usage_sorted: np.ndarray):
    idx = int(np.argmax(usage_sorted))
    return float(usage_sorted[idx]), float(r_sorted[idx])


def fit_isotonic_usage(r: np.ndarray, usage: np.ndarray):
    """Fits a non-increasing (isotonic) regression of usage on c_h.

    The *Bayes-optimal* usage(c_h) is provably non-increasing: for any fixed
    sample, the per-sample-optimal rule is "choose HF if hf_loss + c_h <
    lf_loss" — raising c_h only ever makes the HF side larger, so once a
    sample's optimal choice flips from HF to LF as c_h grows, it can never
    flip back. Usage is just the fraction of samples on the HF side, so the
    true optimal curve can only decrease (or hold), never rise, as c_h grows.

    The raw points here (one per independently-retrained gate, reset_all_weights
    + trained from scratch — see AdaptiveGridSearch.train_fe_model) are noisy
    estimates scattered around that true curve, not the curve itself — the
    non-monotonicity we've actually observed is estimation noise from that
    per-run retraining, not evidence the underlying relationship isn't
    monotonic. Isotonic regression recovers the denoised trend those noisy
    per-c_h estimates are scattered around, without assuming any particular
    smooth functional form (unlike e.g. a polynomial fit).
    """
    order = np.argsort(r)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(r[order], usage[order])
    return iso


def plot_isotonic_smoothing(r: np.ndarray, usage: np.ndarray, save_path: str, benefit: np.ndarray = None) -> None:
    """Raw (c_h, usage) points from every gate retrained during the sweep —
    scattered to show just how noisy per-c_h estimates really are — overlaid
    with the isotonic (non-increasing) fit: the denoised estimate of the true
    Bayes-optimal usage(c_h) curve (see fit_isotonic_usage).

    benefit, if given (see load_test_benefit), adds the *exact* Bayes-optimal
    oracle_usage(c_h) curve too — the isotonic fit only denoises the trained
    gate's own noisy estimates, it's still an approximation of the true curve,
    whereas the oracle line is that true curve, computed directly from the
    frozen models' losses with no training involved.
    """
    iso = fit_isotonic_usage(r, usage)
    r_line = np.linspace(r.min(), r.max(), 200)
    usage_line = iso.predict(r_line)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _style_axes(ax)

    ax.scatter(r, usage * 100, s=20, color=RAW_POINT_COLOR, alpha=0.55, zorder=2,
               label="Raw per-run samples (training noise)")
    ax.plot(r_line, usage_line * 100, color=SERIES_COLOR, linewidth=2.5, zorder=3,
            label="Isotonic fit (non-increasing)")

    if benefit is not None:
        oracle_usage_line = compute_oracle_usage_curve(benefit, r_line)
        ax.plot(r_line, oracle_usage_line * 100, color=INK_MUTED, linewidth=1.5, linestyle="--", zorder=4,
                label="Oracle (exact Bayes-optimal)")

    ax.set_xlabel("c_h", color=INK_SECONDARY)
    ax.set_ylabel("Usage (%)", color=INK_PRIMARY)
    ax.set_title("Usage vs c_h: raw training noise vs. isotonic-smoothed trend", color=INK_PRIMARY, fontweight="bold")
    legend = ax.legend(frameon=False, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved isotonic smoothing plot to {save_path}")


def _style_axes(ax):
    ax.grid(True, color=GRIDLINE, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)
    ax.tick_params(colors=INK_MUTED)


def plot_derivative(r_sorted: np.ndarray, derivative: np.ndarray, save_path: str,
                     benefit: np.ndarray = None, kde_bandwidth=None) -> None:
    """benefit, if given (see load_test_benefit), overlays the exact oracle
    d(usage)/d(c_h) = -density(c_h) - see compute_oracle_usage_density for why
    this is KDE-smoothed rather than a finite difference of the oracle usage
    curve at r_sorted's own (sparse, adaptively-sampled) points.

    The oracle goes on its own (right) y-axis, not the empirical curve's -
    np.gradient on tightly clustered bisection points routinely spikes into
    the tens/hundreds (a small usage change over a tiny c_h gap), while the
    oracle is a proper density bounded to a much smaller range; sharing one
    axis makes the oracle line flatten out near zero and disappear next to
    those spikes, defeating the point of the comparison.
    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _style_axes(ax)

    empirical_line, = ax.plot(
        r_sorted, derivative, color=SERIES_COLOR, linewidth=2, marker="o", markersize=6,
        markerfacecolor=SERIES_COLOR, markeredgewidth=0, zorder=3, label="Empirical (trained gate)"
    )
    ax.axhline(0, color=AXIS_LINE, linewidth=1, zorder=1)

    ax.set_xlabel("c_h", color=INK_SECONDARY)
    ax.set_ylabel("d(usage) / d(c_h)  [empirical]", color=SERIES_COLOR)
    ax.tick_params(axis="y", colors=SERIES_COLOR)
    ax.set_title("Gate usage sensitivity to c_h", color=INK_PRIMARY, fontweight="bold")

    if benefit is not None:
        ax2 = ax.twinx()
        ax2.set_facecolor("none")
        for spine in ("top", "right"):
            ax2.spines[spine].set_visible(False)

        c_h_grid = np.linspace(r_sorted.min(), r_sorted.max(), 300)
        oracle_density = compute_oracle_usage_density(benefit, c_h_grid, bw_method=kde_bandwidth)
        oracle_line, = ax2.plot(
            c_h_grid, oracle_density, color=INK_MUTED, linewidth=1.5, linestyle="--", zorder=2,
            label="Oracle (KDE density of lf_loss - hf_loss)"
        )
        ax2.set_ylabel("d(usage) / d(c_h)  [oracle]", color=INK_MUTED)
        ax2.tick_params(axis="y", colors=INK_MUTED)

        legend = ax.legend([empirical_line, oracle_line], [empirical_line.get_label(), oracle_line.get_label()],
                            frameon=False, loc="best")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved derivative plot to {save_path}")


def plot_zoom(r_sorted: np.ndarray, usage_sorted: np.ndarray, zoom_range, save_path: str,
              benefit: np.ndarray = None) -> None:
    r_min, r_max = zoom_range
    pad = (r_max - r_min) * 0.1 if r_max > r_min else 0.01
    mask = (r_sorted >= r_min - pad) & (r_sorted <= r_max + pad)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _style_axes(ax)

    ax.plot(r_sorted[mask], usage_sorted[mask] * 100, color=SERIES_COLOR, linewidth=2,
             marker="o", markersize=6, markerfacecolor=SERIES_COLOR, markeredgewidth=0, zorder=3,
             label="Trained gate" if benefit is not None else None)

    if benefit is not None:
        c_h_grid = np.linspace(r_min - pad, r_max + pad, 200)
        oracle_usage = compute_oracle_usage_curve(benefit, c_h_grid)
        ax.plot(c_h_grid, oracle_usage * 100, color=INK_MUTED, linewidth=1.5, linestyle="--", zorder=2,
                label="Oracle (exact Bayes-optimal)")
        legend = ax.legend(frameon=False, loc="best")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    ax.set_xlabel("c_h", color=INK_SECONDARY)
    ax.set_ylabel("Usage (%)", color=INK_PRIMARY)
    ax.set_title("Usage vs c_h (zoomed to the 70-90% usage crossing)", color=INK_PRIMARY, fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved zoomed usage plot to {save_path}")


def plot_full_curve(r_sorted: np.ndarray, usage_sorted: np.ndarray, save_path: str, benefit: np.ndarray = None) -> None:
    """Same usage-vs-c_h curve as plot_zoom, but over the full sampled range
    instead of a masked crossing window."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _style_axes(ax)

    ax.plot(r_sorted, usage_sorted * 100, color=SERIES_COLOR, linewidth=2,
             marker="o", markersize=5, markerfacecolor=SERIES_COLOR, markeredgewidth=0, zorder=3,
             label="Trained gate" if benefit is not None else None)

    if benefit is not None:
        c_h_grid = np.linspace(r_sorted.min(), r_sorted.max(), 300)
        oracle_usage = compute_oracle_usage_curve(benefit, c_h_grid)
        ax.plot(c_h_grid, oracle_usage * 100, color=INK_MUTED, linewidth=1.5, linestyle="--", zorder=2,
                label="Oracle (exact Bayes-optimal)")
        legend = ax.legend(frameon=False, loc="best")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    ax.set_xlabel("c_h", color=INK_SECONDARY)
    ax.set_ylabel("Usage (%)", color=INK_PRIMARY)
    ax.set_title("Usage vs c_h (full sampled range)", color=INK_PRIMARY, fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved full usage-vs-c_h plot to {save_path}")


@click.command()
@click.option("--results_folder", required=True, help="Experiment result dir, e.g. results/bird_grayscale-resnet-resnet-42-128")
@click.option("--low", default=0.7, type=float, help="Lower usage bound for the zoom window (fraction, e.g. 0.7 = 70%)")
@click.option("--high", default=0.9, type=float, help="Upper usage bound for the zoom window (fraction, e.g. 0.9 = 90%)")
@click.option("--round_decimals", default=4, type=int, help="Rounding used to group near-duplicate c_h points from independent binary searches")
@click.option("--smooth_window", default=1, type=int, help="Moving-average window on the derivative curve; 1 = none")
@click.option("--kde_bandwidth", default=None, type=float, help="Bandwidth for the oracle derivative's KDE (scipy bw_method); default lets scipy pick one via Scott's rule")
def main(results_folder, low, high, round_decimals, smooth_window, kde_bandwidth):
    results_folder = os.path.normpath(results_folder)
    file_folder = os.path.join(results_folder, "files")
    image_folder = os.path.join(results_folder, "images")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(results_folder, "gate_sensitivity.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )

    r, usage, acc = load_gate_log(file_folder)
    r_sorted, usage_sorted, acc_sorted = dedupe_and_sort(r, usage, acc, round_decimals)

    try:
        benefit = load_test_benefit(os.path.join(results_folder, "latent", "test"))
    except FileNotFoundError as e:
        logger.warning(f"No oracle overlay available: {e}")
        benefit = None

    derivative = compute_usage_derivative(r_sorted, usage_sorted, smooth_window)
    plot_derivative(
        r_sorted, derivative, os.path.join(image_folder, "gate_sensitivity_derivative.png"),
        benefit=benefit, kde_bandwidth=kde_bandwidth
    )

    plot_full_curve(r_sorted, usage_sorted, os.path.join(image_folder, "gate_sensitivity_full.png"), benefit=benefit)

    zoom_range = find_zoom_range(r_sorted, usage_sorted, low, high)
    plot_zoom(r_sorted, usage_sorted, zoom_range, os.path.join(image_folder, "gate_sensitivity_zoom.png"), benefit=benefit)

    plot_isotonic_smoothing(r, usage, os.path.join(image_folder, "gate_sensitivity_isotonic.png"), benefit=benefit)

    max_usage, r_at_max = report_max_usage(r_sorted, usage_sorted)
    logger.info(
        f"Max achievable usage under the current c_h sampling grid: {max_usage * 100:.2f}% "
        f"at c_h={r_at_max:.4f} (this reflects what was sampled, not necessarily the true "
        "supremum of the underlying usage(c_h) function)."
    )
    print(f"Max achievable usage under the current c_h sampling grid: {max_usage * 100:.2f}% at c_h={r_at_max:.4f}")


if __name__ == "__main__":
    main()
