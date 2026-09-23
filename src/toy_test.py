"""Standalone toy_2d sanity check for the FE gate.

Builds the toy_2d hypercube dataset directly (same construction as
build_dataset's "toy_2d" case in helpers.py), trains the LF/HF MLP
classifiers, then sweeps the FE gate (MetaLossFunction) across a range of
ch cost values and renders a video of how the gate's routing decision moves
across the raw 2D LF point cloud as ch goes from expensive (mostly LF) to
free (mostly HF). Unlike the main pipeline, everything here is kept in
memory - num_dims=2 keeps the LF points themselves plottable directly, so
there's no need for save_latent/FE_Dataset's disk caching or
plot_gate_routing's supervised 2D projection.
"""
import os
import logging

import click
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datasets import HypercubeDataset
from helpers import (
    build_model, classifier_one_run, set_all_seeds, reset_all_weights,
    assemble_hf_input, compute_gate_routing_details
)
from losses import MetaLossFunction
from plot_gate_routing import LF_COLOR, HF_COLOR

logger = logging.getLogger(__name__)


def build_toy_datasets(seed: int):
    """Verbatim toy_2d construction (see build_dataset's "toy_2d" case in
    helpers.py) - kept separate/inline here so this script has no dependency
    on config.yml or the rest of the experiment pipeline."""
    set_all_seeds(seed)

    num_dims = 2
    num_clusters = 2 ** num_dims
    hf_std = np.random.uniform(0.26, 0.30, size=num_clusters)
    lf_std = np.random.uniform(0.48, 0.54, size=num_clusters)

    total_num_samples = 3000
    train_samples = int(total_num_samples * .8)
    test_samples = int(total_num_samples * .1)
    val_samples = int(total_num_samples * .1)

    train_samples_per_cluster = np.full(num_clusters, train_samples // num_clusters)
    test_samples_per_cluster = np.full(num_clusters, test_samples // num_clusters)
    val_samples_per_cluster = np.full(num_clusters, val_samples // num_clusters)

    train_ds = HypercubeDataset(
        num_dims=num_dims,
        num_samples=train_samples_per_cluster,
        hf_std=hf_std,
        lf_std=lf_std,
    )

    test_ds = HypercubeDataset(
        num_dims=num_dims,
        num_samples=test_samples_per_cluster,
        hf_std=hf_std,
        lf_std=lf_std,
    )

    val_ds = HypercubeDataset(
        num_dims=num_dims,
        num_samples=val_samples_per_cluster,
        hf_std=hf_std,
        lf_std=lf_std,
    )

    return train_ds, test_ds, val_ds


def train_classifier(model, train_dl, val_dl, fidelity, epochs, lr, device, hf_input_mode="concat"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        classifier_one_run(
            model=model, dataloader=train_dl, criterion=criterion, fidelity=fidelity,
            optimizer=optimizer, hf_input_mode=hf_input_mode
        )
        _, val_acc = classifier_one_run(
            model=model, dataloader=val_dl, criterion=criterion, fidelity=fidelity,
            hf_input_mode=hf_input_mode
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    logger.info(f"{fidelity.upper()} classifier best val accuracy: {100 * best_val_acc:.2f}%")
    return model, best_val_acc


@torch.no_grad()
def compute_gate_tensors(lf_model, hf_model, ds, hf_input_mode, device):
    """One-shot forward pass over an entire (small) split, producing the
    (lf_latent, lf_output, hf_output, label) tensors classifier_one_run's
    "gate" fidelity expects - the in-memory equivalent of save_latent +
    FE_Dataset, skipped here since the toy dataset comfortably fits in RAM."""
    lf_model.eval()
    hf_model.eval()

    lf_x = ds.lf_points.to(device, torch.float)
    hf_x = assemble_hf_input(ds.lf_points, ds.hf_points, hf_input_mode).to(device, torch.float)
    labels = ds.labels.to(device)

    lf_head = lf_model(lf_x)
    hf_head = hf_model(hf_x)

    return (
        lf_head["latent"].cpu(),
        lf_head["output"].cpu(),
        hf_head["output"].cpu(),
        labels.cpu(),
    )


def train_gate_for_r(r_val, gate_input_size, train_tensors, val_tensors, device,
                      epochs, lr, batch_size, class_weighted, grad_clip_norm):
    fe_model = build_model(model_name="mlp", input_size=gate_input_size, output_size=2, latent_size=256)
    reset_all_weights(fe_model)
    fe_model = fe_model.to(device)

    optimizer = torch.optim.Adam(fe_model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = MetaLossFunction(ch=[r_val], cw=1, device=device, class_weighted=class_weighted)

    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_tensors), batch_size=batch_size, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*val_tensors), batch_size=batch_size
    )

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in fe_model.state_dict().items()}

    for _ in range(epochs):
        classifier_one_run(
            model=fe_model, dataloader=train_dl, criterion=criterion, fidelity="gate",
            optimizer=optimizer, grad_clip_norm=grad_clip_norm
        )
        val_loss, _, _ = classifier_one_run(
            model=fe_model, dataloader=val_dl, criterion=criterion, fidelity="gate"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in fe_model.state_dict().items()}

    fe_model.load_state_dict(best_state)
    return fe_model


def render_routing_video(frames, save_path, fps):
    """frames: list of (r_val, hf_usage_pct, points_xy, choice) tuples, in
    display order. points_xy is (N, 2) raw LF coordinates, choice is (N,)
    with 0=LF/1=HF - plotted directly, no projection needed since num_dims=2."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")

    # Fixed axis limits across all frames (rather than per-frame) so the
    # view doesn't jitter/rescale as points move between LF and HF
    # coordinates from one frame to the next.
    all_points = np.concatenate([f[2] for f in frames], axis=0)
    xlim = (all_points[:, 0].min() - 0.1, all_points[:, 0].max() + 0.1)
    ylim = (all_points[:, 1].min() - 0.1, all_points[:, 1].max() + 0.1)

    def draw(idx):
        ax.clear()
        r_val, hf_usage_pct, points_xy, choice = frames[idx]

        ax.set_facecolor("#fcfcfb")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        lf_mask = choice == 0
        hf_mask = choice == 1
        ax.scatter(points_xy[lf_mask, 0], points_xy[lf_mask, 1], c=LF_COLOR, s=10, alpha=0.7, linewidths=0, label="Routed LF")
        ax.scatter(points_xy[hf_mask, 0], points_xy[hf_mask, 1], c=HF_COLOR, s=10, alpha=0.7, linewidths=0, label="Routed HF")

        ax.set_title(f"ch={r_val:.3f}  |  HF usage={hf_usage_pct:.1f}%", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        return ax.collections

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)

    try:
        anim.save(save_path, writer=animation.FFMpegWriter(fps=fps))
        logger.info(f"Saved routing video to {save_path}")
    except (FileNotFoundError, RuntimeError):
        gif_path = os.path.splitext(save_path)[0] + ".gif"
        logger.warning(f"ffmpeg not available; falling back to GIF at {gif_path}")
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
        logger.info(f"Saved routing video to {gif_path}")

    plt.close(fig)


@click.command()
@click.option("--seed", default=42, help="Random seed for dataset generation + training")
@click.option("--latent-size", default=32, help="Hidden size for the LF/HF MLPs")
@click.option("--classifier-epochs", default=200, help="Epochs to train the LF/HF classifiers")
@click.option("--classifier-lr", default=1e-3, help="LR for the LF/HF classifiers")
@click.option("--gate-epochs", default=40, help="Epochs to train each r value's gate model")
@click.option("--gate-lr", default=3e-4, help="LR for the gate model (lowered automatically if routing signal is thin)")
@click.option("--num-r", default=200, help="Number of ch values to sweep")
@click.option("--r-start", default=0.0, help="Low end of the ch sweep")
@click.option("--r-stop", default=0.5, help="High end of the ch sweep")
@click.option("--reverse/--no-reverse", default=True, help="Sweep from r-stop down to r-start (expensive HF -> free HF)")
@click.option("--fps", default=2, help="Frames per second in the output video")
@click.option("--output", default=os.path.join("results", "toy_test", "images", "gate_routing_video_200_v2.mp4"))
def main(seed, latent_size, classifier_epochs, classifier_lr, gate_epochs, gate_lr,
         num_r, r_start, r_stop, reverse, fps, output):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("BUILDING TOY_2D DATASET")
    train_ds, test_ds, val_ds = build_toy_datasets(seed)

    batch_size = 32
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    lf_input_size = train_ds.get_lf_input_size()
    hf_input_size = train_ds.get_hf_input_size()
    output_size = train_ds.get_num_classes()
    hf_input_mode = "concat"

    logger.info("TRAINING LF MODEL")
    lf_model = build_model(model_name="mlp", input_size=lf_input_size, output_size=output_size, latent_size=latent_size)
    lf_model, lf_val_acc = train_classifier(
        lf_model, train_dl, val_dl, fidelity="lf", epochs=classifier_epochs, lr=classifier_lr, device=device
    )

    logger.info("TRAINING HF MODEL")
    hf_model = build_model(
        model_name="mlp", input_size=lf_input_size + hf_input_size, output_size=output_size, latent_size=latent_size
    )
    hf_model, hf_val_acc = train_classifier(
        hf_model, train_dl, val_dl, fidelity="hf", epochs=classifier_epochs, lr=classifier_lr,
        device=device, hf_input_mode=hf_input_mode
    )

    logger.info(f"LF val accuracy: {100 * lf_val_acc:.2f}% | HF val accuracy: {100 * hf_val_acc:.2f}%")

    logger.info("PRECOMPUTING GATE TRAINING TENSORS")
    train_tensors = compute_gate_tensors(lf_model, hf_model, train_ds, hf_input_mode, device)
    val_tensors = compute_gate_tensors(lf_model, hf_model, val_ds, hf_input_mode, device)

    # Plot on the full dataset (train+val+test combined) for a denser, more
    # complete picture of the 2D plane than any single split gives alone.
    all_lf_points = torch.cat([train_ds.lf_points, val_ds.lf_points, test_ds.lf_points], dim=0).numpy()
    all_hf_points = torch.cat([train_ds.hf_points, val_ds.hf_points, test_ds.hf_points], dim=0).numpy()
    all_labels = torch.cat([train_ds.labels, val_ds.labels, test_ds.labels], dim=0)
    all_lf_x = torch.cat([train_ds.lf_points, val_ds.lf_points, test_ds.lf_points], dim=0)
    all_hf_x = torch.cat([train_ds.hf_points, val_ds.hf_points, test_ds.hf_points], dim=0)

    class _AllDS:
        pass
    all_ds = _AllDS()
    all_ds.lf_points = all_lf_x
    all_ds.hf_points = all_hf_x
    all_ds.labels = all_labels
    all_tensors = compute_gate_tensors(lf_model, hf_model, all_ds, hf_input_mode, device)
    all_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*all_tensors), batch_size=256)

    # Same imbalance mitigation main.py applies (see AdaptiveGridSearch usage
    # there) - toy_2d's LF/HF models agree often enough that an unweighted,
    # un-clipped gate can collapse to "always pick one fidelity" within a few
    # epochs regardless of cost.
    lf_correct = (train_tensors[1].argmax(dim=1) == train_tensors[3])
    hf_correct = (train_tensors[2].argmax(dim=1) == train_tensors[3])
    hf_needed_rate = (hf_correct & ~lf_correct).float().mean().item()
    logger.info(f"HF-needed rate on train set: {hf_needed_rate:.2%}")
    imbalanced_gate = hf_needed_rate < 0.15
    effective_gate_lr = 5e-5 if imbalanced_gate else gate_lr
    grad_clip_norm = 1.0 if imbalanced_gate else None

    r_values = np.linspace(r_start, r_stop, num_r)
    if reverse:
        r_values = r_values[::-1]

    gate_input_size = train_tensors[0].shape[1]

    frames = []
    swap_frames = []
    for r_val in r_values:
        logger.info(f"Training gate for ch={r_val:.4f}")
        fe_model = train_gate_for_r(
            r_val=float(r_val), gate_input_size=gate_input_size,
            train_tensors=train_tensors, val_tensors=val_tensors, device=device,
            epochs=gate_epochs, lr=effective_gate_lr, batch_size=batch_size,
            class_weighted=imbalanced_gate, grad_clip_norm=grad_clip_norm
        )

        routing = compute_gate_routing_details(fe_model, all_dl, device)
        choice = routing["choice"]
        hf_usage_pct = 100 * choice.mean()

        frames.append((float(r_val), hf_usage_pct, all_lf_points, choice))

        # Same routing choice, but HF-routed points are plotted at their
        # actual HF coordinates instead of LF coordinates - shows where the
        # gate's picks actually land in HF space as ch relaxes.
        swap_points = np.where(choice[:, None] == 1, all_hf_points, all_lf_points)
        swap_frames.append((float(r_val), hf_usage_pct, swap_points, choice))

        logger.info(f"ch={r_val:.4f} -> HF usage={hf_usage_pct:.1f}%")

    logger.info("RENDERING ROUTING VIDEO")
    render_routing_video(frames, output, fps)

    swap_output = os.path.join(
        os.path.dirname(output), os.path.splitext(os.path.basename(output))[0] + "_hf_swap" + os.path.splitext(output)[1]
    )
    logger.info("RENDERING HF-SWAP ROUTING VIDEO")
    render_routing_video(swap_frames, swap_output, fps)


if __name__ == "__main__":
    main()
