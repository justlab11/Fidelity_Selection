import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import click

logger = logging.getLogger(__name__)

# Fixed categorical color order (never cycled) so a baseline always maps to the
# same color across every plot this script produces.
BASELINE_STYLE = {
    "FE":           {"file": "fe_results.npz",           "color": "#2a78d6"},
    "SR":           {"file": "sr_results.npz",            "color": "#eb6834"},
    "SelectiveNet": {"file": "selectivenet_results.npz",  "color": "#1baf7a"},
    "SAT":          {"file": "sat_results.npz",           "color": "#eda100"},
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"


def load_pareto_curves(file_folder: str) -> dict:
    """Loads whichever of {fe,sr,selectivenet,sat}_results.npz exist under
    file_folder. Each has 'usage'/'acc' arrays on a 0-100 scale. Missing files
    are skipped (with a warning) so partial overlays still render."""
    curves = {}

    for label, style in BASELINE_STYLE.items():
        path = os.path.join(file_folder, style["file"])
        if not os.path.exists(path):
            logger.warning(f"{label}: '{path}' not found, skipping")
            continue

        data = np.load(path)
        usage = np.asarray(data["usage"], dtype=float)
        acc = np.asarray(data["acc"], dtype=float)

        order = np.argsort(usage)
        curves[label] = (usage[order], acc[order])

    return curves


def render_pareto_plot(curves: dict, dataset_label: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.grid(True, color=GRIDLINE, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)

    for label, style in BASELINE_STYLE.items():
        if label not in curves:
            continue
        usage, acc = curves[label]
        ax.plot(
            usage, acc,
            color=style["color"], linewidth=2, marker="o", markersize=6,
            markerfacecolor=style["color"], markeredgewidth=0,
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


def plot_one_dataset(results_folder: str, output_folder: str | None) -> None:
    results_folder = os.path.normpath(results_folder)
    file_folder = os.path.join(results_folder, "files")
    dataset_label = os.path.basename(results_folder).split("-")[0]

    curves = load_pareto_curves(file_folder)

    save_folder = output_folder if output_folder else os.path.join(results_folder, "images")
    save_path = os.path.join(save_folder, "pareto.png")

    render_pareto_plot(curves, dataset_label, save_path)


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
