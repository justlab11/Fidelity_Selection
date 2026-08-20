import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import click

logger = logging.getLogger(__name__)

SERIES_COLOR = "#2a78d6"
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


def _style_axes(ax):
    ax.grid(True, color=GRIDLINE, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)
    ax.tick_params(colors=INK_MUTED)


def plot_derivative(r_sorted: np.ndarray, derivative: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _style_axes(ax)

    ax.plot(r_sorted, derivative, color=SERIES_COLOR, linewidth=2, marker="o", markersize=6,
             markerfacecolor=SERIES_COLOR, markeredgewidth=0, zorder=3)
    ax.axhline(0, color=AXIS_LINE, linewidth=1, zorder=1)

    ax.set_xlabel("c_h", color=INK_SECONDARY)
    ax.set_ylabel("d(usage) / d(c_h)", color=INK_PRIMARY)
    ax.set_title("Gate usage sensitivity to c_h", color=INK_PRIMARY, fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved derivative plot to {save_path}")


def plot_zoom(r_sorted: np.ndarray, usage_sorted: np.ndarray, zoom_range, save_path: str) -> None:
    r_min, r_max = zoom_range
    pad = (r_max - r_min) * 0.1 if r_max > r_min else 0.01
    mask = (r_sorted >= r_min - pad) & (r_sorted <= r_max + pad)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _style_axes(ax)

    ax.plot(r_sorted[mask], usage_sorted[mask] * 100, color=SERIES_COLOR, linewidth=2,
             marker="o", markersize=6, markerfacecolor=SERIES_COLOR, markeredgewidth=0, zorder=3)

    ax.set_xlabel("c_h", color=INK_SECONDARY)
    ax.set_ylabel("Usage (%)", color=INK_PRIMARY)
    ax.set_title("Usage vs c_h (zoomed to the 70-90% usage crossing)", color=INK_PRIMARY, fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved zoomed usage plot to {save_path}")


@click.command()
@click.option("--results_folder", required=True, help="Experiment result dir, e.g. results/bird_grayscale-resnet-resnet-42-128")
@click.option("--low", default=0.7, type=float, help="Lower usage bound for the zoom window (fraction, e.g. 0.7 = 70%)")
@click.option("--high", default=0.9, type=float, help="Upper usage bound for the zoom window (fraction, e.g. 0.9 = 90%)")
@click.option("--round_decimals", default=4, type=int, help="Rounding used to group near-duplicate c_h points from independent binary searches")
@click.option("--smooth_window", default=1, type=int, help="Moving-average window on the derivative curve; 1 = none")
def main(results_folder, low, high, round_decimals, smooth_window):
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

    derivative = compute_usage_derivative(r_sorted, usage_sorted, smooth_window)
    plot_derivative(r_sorted, derivative, os.path.join(image_folder, "gate_sensitivity_derivative.png"))

    zoom_range = find_zoom_range(r_sorted, usage_sorted, low, high)
    plot_zoom(r_sorted, usage_sorted, zoom_range, os.path.join(image_folder, "gate_sensitivity_zoom.png"))

    max_usage, r_at_max = report_max_usage(r_sorted, usage_sorted)
    logger.info(
        f"Max achievable usage under the current c_h sampling grid: {max_usage * 100:.2f}% "
        f"at c_h={r_at_max:.4f} (this reflects what was sampled, not necessarily the true "
        "supremum of the underlying usage(c_h) function)."
    )
    print(f"Max achievable usage under the current c_h sampling grid: {max_usage * 100:.2f}% at c_h={r_at_max:.4f}")


if __name__ == "__main__":
    main()
