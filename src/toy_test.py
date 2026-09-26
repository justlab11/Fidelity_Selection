"""Standalone toy_2d sanity check for the FE gate.

Builds the toy_2d hypercube dataset directly (same construction as
build_dataset's "toy_2d" case in helpers.py), trains the LF/HF MLP
classifiers, then uses the same AdaptiveGridSearch bisection search main.py's
Pareto curve runs on to find a gate ch value for each of num_targets usage
levels (10%, 20%, ..., 100% by default) and renders a video/grid of how the
gate's routing decision moves across the raw 2D LF point cloud from mostly-LF
to mostly-HF. Unlike the main pipeline, everything here is kept in memory -
num_dims=2 keeps the LF points themselves plottable directly, so there's no
need for FE_Dataset's disk caching or plot_gate_routing's supervised 2D
projection (AdaptiveGridSearch still needs a small models/ folder on disk for
its own checkpointing, same as it does in the main pipeline).
"""
import os
import logging

import click
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import roc_auc_score

from datasets import HypercubeDataset
from helpers import (
    build_model, classifier_one_run, set_all_seeds,
    assemble_hf_input, compute_gate_routing_details, compute_fidelity_loss_correct,
    AdaptiveGridSearch
)

logger = logging.getLogger(__name__)

# tab10's first two colors (used for the ground-truth class labels below) are
# almost the same blue/orange as plot_gate_routing's LF_COLOR/HF_COLOR, so
# reusing those for the routing plots too would make "blue" ambiguous between
# "class 1" and "routed LF" at a glance. Routing plots get their own color
# pair instead, and HF points get a distinct marker shape (triangle vs.
# circle) everywhere - shape then means "LF vs. HF" consistently across every
# panel, while color is free to mean different things (class vs. routing
# choice) panel to panel.
ROUTED_LF_COLOR = "#6a51a3"
ROUTED_HF_COLOR = "#238b45"
LF_MARKER = "o"
HF_MARKER = "^"


class IndependentHypercubeDataset(HypercubeDataset):
    """Same hypercube-corner cluster geometry/labeling as HypercubeDataset
    (reuses its _hypercube_corners/_get_label_set/_generate_points_and_labels
    unchanged), but lf_points and hf_points are independent draws around each
    cluster's mean rather than HypercubeDataset's lf = hf + extra_noise
    coupling (extra_std = sqrt(lf_std^2 - hf_std^2)).

    That coupling has a low ceiling whenever lf_std >> hf_std: extra_std ends
    up close to lf_std itself (e.g. lf_std=0.75, hf_std=0.1 ->
    extra_std=sqrt(0.75^2-0.1^2)=0.743), so ~99% of lf_point's variance is
    noise that has nothing to do with hf_point's own (tiny-variance) draw -
    confirmed directly via the disagreement probe below (AUC ~0.6, barely
    above chance, on the "hard case" build_toy_datasets produces). This class
    is the "high ceiling" companion: LF is still noisier than HF, but it's an
    independent, moderately-noisier observation of the same cluster mean, not
    a noise-swamped function of HF's own draw - see build_high_ceiling_toy_datasets.
    """
    def __init__(self, num_dims, num_samples, hf_std, lf_std, group_classes=True):
        # Deliberately does NOT call HypercubeDataset.__init__ (that always
        # does the coupled noise-injection construction) - reuses its helper
        # methods directly instead, exactly the way HypercubeDataset itself
        # generates hf_points/labels.
        self.num_dims = num_dims
        self.num_clusters = 2 ** num_dims
        self.num_classes = self.num_clusters // 2 if group_classes else self.num_clusters
        self.group_classes = group_classes

        num_samples = np.insert(num_samples, 0, 0)

        self.hf_points, self.labels = self._generate_points_and_labels(
            num_dims, self.num_classes, num_samples, hf_std, group_classes
        )
        # Independent draw around the same per-cluster means - NOT derived
        # from self.hf_points at all, unlike HypercubeDataset.
        self.lf_points, _ = self._generate_points_and_labels(
            num_dims, self.num_classes, num_samples, lf_std, group_classes
        )

        self.hf_points = torch.from_numpy(self.hf_points.astype(np.float32))
        self.lf_points = torch.from_numpy(self.lf_points.astype(np.float32))
        self.labels = torch.from_numpy(self.labels.astype(np.int64))


def _build_split_sizes(total_num_samples, num_clusters):
    train_samples = int(total_num_samples * .8)
    test_samples = int(total_num_samples * .1)
    val_samples = int(total_num_samples * .1)
    return (
        np.full(num_clusters, train_samples // num_clusters),
        np.full(num_clusters, test_samples // num_clusters),
        np.full(num_clusters, val_samples // num_clusters),
    )


def build_toy_datasets(seed: int):
    """toy_2d "hard case" construction (see build_dataset's "toy_2d" case in
    helpers.py) - kept separate/inline here so this script has no dependency
    on config.yml or the rest of the experiment pipeline. hf_std is lowered
    well below helpers.py's toy_2d value (0.26-0.30) so HF clusters sit
    tightly on their hypercube corners - this script is a visual sanity
    check, so a more obviously-separable HF case makes the LF-vs-HF
    comparison easier to read at a glance.

    lf_std is raised well above helpers.py's toy_2d value (0.48-0.54) too -
    empirically, the original (0.08-0.12, 0.48-0.54) pairing gives
    hf_needed_rate ~24%, which sounds like plenty of routing signal but isn't:
    the gate reliably collapses to "always LF" for any c_h above ~0.5 and,
    once collapsed, more gate_epochs makes it *more* stuck (softmax
    saturation is self-reinforcing - see the consistency-review discussion),
    not less. That collapse zone swallowed several of the usage_values sweep's
    lower targets, landing on the same cached "usage=0%" point instead of
    resolving separately. Widening lf_std to 0.70-0.80 pushes hf_needed_rate
    to ~35%, which measurably shrinks (but does not eliminate) that collapse
    zone - see main.py's --gate-lr tuning for the rest of the fix.

    This is deliberately the "hard case": the disagreement probe (see
    run_disagreement_probe) confirms this construction has a low ceiling
    (AUC ~0.6) - not just a training-dynamics artifact. See
    build_high_ceiling_toy_datasets for a companion construction with a
    genuinely high ceiling, for contrast.
    """
    set_all_seeds(seed)

    num_dims = 2
    num_clusters = 2 ** num_dims
    hf_std = np.random.uniform(0.08, 0.12, size=num_clusters)
    lf_std = np.random.uniform(0.70, 0.80, size=num_clusters)

    train_n, test_n, val_n = _build_split_sizes(3000, num_clusters)

    train_ds = HypercubeDataset(num_dims=num_dims, num_samples=train_n, hf_std=hf_std, lf_std=lf_std)
    test_ds = HypercubeDataset(num_dims=num_dims, num_samples=test_n, hf_std=hf_std, lf_std=lf_std)
    val_ds = HypercubeDataset(num_dims=num_dims, num_samples=val_n, hf_std=hf_std, lf_std=lf_std)

    return train_ds, test_ds, val_ds


def build_high_ceiling_toy_datasets(seed: int):
    """toy_2d "high ceiling" companion to build_toy_datasets, for the
    appendix's "intuition" figure: LF and HF are independent draws around the
    same cluster means (IndependentHypercubeDataset) rather than
    HypercubeDataset's lf = hf + extra_noise coupling, with lf_std only
    moderately larger than hf_std (not the "hard case"'s ~7x gap) - so LF is
    still noisier than HF, but its errors aren't dominated by noise that has
    nothing to do with HF's own draw. Verify with the disagreement probe
    before relying on this for a figure - it's tuned to give a high AUC, but
    depends on the same random draw the "hard case" does.
    """
    set_all_seeds(seed)

    num_dims = 2
    num_clusters = 2 ** num_dims
    hf_std = np.random.uniform(0.15, 0.20, size=num_clusters)
    lf_std = np.random.uniform(0.35, 0.45, size=num_clusters)

    train_n, test_n, val_n = _build_split_sizes(3000, num_clusters)

    train_ds = IndependentHypercubeDataset(num_dims=num_dims, num_samples=train_n, hf_std=hf_std, lf_std=lf_std)
    test_ds = IndependentHypercubeDataset(num_dims=num_dims, num_samples=test_n, hf_std=hf_std, lf_std=lf_std)
    val_ds = IndependentHypercubeDataset(num_dims=num_dims, num_samples=val_n, hf_std=hf_std, lf_std=lf_std)

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
    (lf_latent, lf_loss, hf_loss, lf_correct, hf_correct, idx) tensors
    classifier_one_run's "gate" fidelity expects - the in-memory equivalent of
    save_latent + FE_Dataset, skipped here since the toy dataset comfortably
    fits in RAM. idx is just this split's own row order (arange), since there's
    no saved-to-disk dataset to trace back to here."""
    lf_model.eval()
    hf_model.eval()

    lf_x = ds.lf_points.to(device, torch.float)
    hf_x = assemble_hf_input(ds.lf_points, ds.hf_points, hf_input_mode).to(device, torch.float)
    labels = ds.labels.to(device)

    lf_head = lf_model(lf_x)
    hf_head = hf_model(hf_x)

    lf_loss, lf_correct = compute_fidelity_loss_correct(lf_head["output"], labels)
    hf_loss, hf_correct = compute_fidelity_loss_correct(hf_head["output"], labels)

    return (
        lf_head["latent"].cpu(),
        lf_loss.cpu(),
        hf_loss.cpu(),
        lf_correct.cpu(),
        hf_correct.cpu(),
        torch.arange(lf_x.size(0)),
    )


@torch.no_grad()
def compute_diagnostic_latents(lf_model, hf_model, ds, hf_input_mode, device):
    """Like compute_gate_tensors, but for run_disagreement_probe specifically:
    grabs LF's early_latent (CustomMLP's first hidden Linear+ReLU) alongside
    its usual final latent, plus lf_correct/hf_correct - nothing else the
    probe doesn't need. Kept separate from compute_gate_tensors rather than
    extending its return tuple, since that tuple's shape/order is relied on
    everywhere else (AdaptiveGridSearch, classifier_one_run's "gate" fidelity)."""
    lf_model.eval()
    hf_model.eval()

    lf_x = ds.lf_points.to(device, torch.float)
    hf_x = assemble_hf_input(ds.lf_points, ds.hf_points, hf_input_mode).to(device, torch.float)
    labels = ds.labels.to(device)

    lf_head = lf_model(lf_x)
    hf_head = hf_model(hf_x)

    _, lf_correct = compute_fidelity_loss_correct(lf_head["output"], labels)
    _, hf_correct = compute_fidelity_loss_correct(hf_head["output"], labels)

    return (
        lf_head["early_latent"].cpu(),
        lf_head["latent"].cpu(),
        lf_correct.cpu(),
        hf_correct.cpu(),
    )


def run_disagreement_probe(lf_latent, val_lf_latent, lf_correct, hf_correct,
                            val_lf_correct, val_hf_correct,
                            latent_name, device, epochs=100, lr=1e-3, hidden=64):
    """Standalone diagnostic, deliberately independent of MetaLossFunction:
    can a plain, class-weighted supervised classifier predict "disagreement"
    (would HF have helped) directly from a given LF latent, with no ch/
    expected-cost objective involved at all. Isolates input quality (this
    probe's ceiling) from training-dynamics issues (the real gate's
    ch-weighted objective, whose linear-in-p_d(2) cost structure + softmax's
    p*(1-p) gradient scaling is what's suspected of causing the collapse
    documented elsewhere in this file's history).

    Only ever takes an LF-derived latent (never hf_latent) - the real gate
    only has lf_latent available at routing time; computing hf_latent to
    decide whether to escalate to HF would already cost as much as just
    running HF, defeating the cascade. A latent that includes hf_latent
    would trivially get a higher AUC here but wouldn't say anything about
    the deployable gate, so it's deliberately not an option this function
    accepts.
    """
    y_train = (hf_correct & ~lf_correct).long()  # 1 = HF would have helped
    y_val = (val_hf_correct & ~val_lf_correct).long()

    train_pos_rate = y_train.float().mean().item()
    val_pos_rate = y_val.float().mean().item()
    print(f"[{latent_name}] disagreement rate: train={train_pos_rate:.2%} val={val_pos_rate:.2%}")

    probe = nn.Sequential(
        nn.Linear(lf_latent.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, 2)
    ).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-5)

    # Standard class-weighted CE - NOT the expected-cost objective, so no
    # saturation-inducing linear-in-p_d(2) structure here.
    class_weights = torch.tensor(
        [1.0, (1 - train_pos_rate) / max(train_pos_rate, 1e-6)], device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    x_train = lf_latent.to(device)
    x_val = val_lf_latent.to(device)
    y_train = y_train.to(device)

    for epoch in range(epochs):
        probe.train()
        opt.zero_grad()
        logits = probe(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        val_probs = torch.softmax(probe(x_val), dim=1)[:, 1].cpu().numpy()

    auc = roc_auc_score(y_val.numpy(), val_probs)
    print(f"[{latent_name}] held-out AUC on disagreement prediction: {auc:.3f}")

    # Calibration: is the probe actually spreading probabilities, or
    # collapsing toward 0/1 the way the real gate's softmax does?
    hist, edges = np.histogram(val_probs, bins=10, range=(0, 1))
    print(f"[{latent_name}] predicted-prob histogram (0->1): {hist.tolist()}")

    return auc, val_probs


def render_routing_video(frames, save_path, fps):
    """frames: list of (r_val, hf_usage_pct, points_xy, choice, accuracy) tuples,
    in display order. points_xy is (N, 2) raw LF coordinates, choice is (N,)
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
        r_val, hf_usage_pct, points_xy, choice, accuracy = frames[idx]

        ax.set_facecolor("#fcfcfb")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        lf_mask = choice == 0
        hf_mask = choice == 1
        ax.scatter(points_xy[lf_mask, 0], points_xy[lf_mask, 1], c=ROUTED_LF_COLOR, marker=LF_MARKER, s=10, alpha=0.7, linewidths=0, label="Routed LF")
        ax.scatter(points_xy[hf_mask, 0], points_xy[hf_mask, 1], c=ROUTED_HF_COLOR, marker=HF_MARKER, s=10, alpha=0.7, linewidths=0, label="Routed HF")

        ax.set_title(f"ch={r_val:.3f}  |  HF usage={hf_usage_pct:.1f}%  |  acc={100 * accuracy:.1f}%", fontsize=11)
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


def render_hf_usage_grid(selected_frames, selected_swap_frames, all_lf_points, all_hf_points, all_labels, save_path):
    """Top row is a ground-truth reference (not routing-dependent): LF samples
    colored by class on the top left, HF samples colored by class on the top
    right. Below that, one row per selected frame, "without swap" (LF-space
    coords) next to "with swap" (HF-routed points at their HF coords), so the
    routing rows can be read against the reference row above them."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = len(selected_frames)
    total_rows = n + 1
    fig, axes = plt.subplots(total_rows, 2, figsize=(10, 5 * total_rows), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")

    all_points = np.concatenate(
        [f[2] for f in selected_frames] + [f[2] for f in selected_swap_frames], axis=0
    )
    xlim = (all_points[:, 0].min() - 0.1, all_points[:, 0].max() + 0.1)
    ylim = (all_points[:, 1].min() - 0.1, all_points[:, 1].max() + 0.1)

    all_labels = all_labels.numpy() if torch.is_tensor(all_labels) else np.asarray(all_labels)
    classes = np.unique(all_labels)
    cmap = plt.get_cmap("tab10")

    reference_columns = [
        ("LF samples (ground-truth classes)", all_lf_points, LF_MARKER),
        ("HF samples (ground-truth classes)", all_hf_points, HF_MARKER),
    ]
    for col, (title, points_xy, marker) in enumerate(reference_columns):
        ax = axes[0, col]
        ax.set_facecolor("#fcfcfb")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        for i, cls in enumerate(classes):
            mask = all_labels == cls
            ax.scatter(points_xy[mask, 0], points_xy[mask, 1], color=cmap(i % 10), marker=marker, s=10, alpha=0.7, linewidths=0, label=f"Class {int(cls)}")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=7, frameon=False)

    columns = [
        ("Without swap", selected_frames),
        ("With HF swap", selected_swap_frames),
    ]

    for row in range(n):
        for col, (col_label, col_frames) in enumerate(columns):
            r_val, hf_usage_pct, points_xy, choice, accuracy = col_frames[row]
            ax = axes[row + 1, col]

            ax.set_facecolor("#fcfcfb")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            lf_mask = choice == 0
            hf_mask = choice == 1
            ax.scatter(points_xy[lf_mask, 0], points_xy[lf_mask, 1], c=ROUTED_LF_COLOR, marker=LF_MARKER, s=10, alpha=0.7, linewidths=0, label="Routed LF")
            ax.scatter(points_xy[hf_mask, 0], points_xy[hf_mask, 1], c=ROUTED_HF_COLOR, marker=HF_MARKER, s=10, alpha=0.7, linewidths=0, label="Routed HF")

            ax.set_title(
                f"{col_label}  |  ch={r_val:.3f}  |  HF usage={hf_usage_pct:.1f}%  |  acc={100 * accuracy:.1f}%",
                fontsize=10,
            )
            if row == 0 and col == 1:
                ax.legend(loc="upper right", fontsize=8, frameon=False)

    plt.tight_layout()
    fig.savefig(save_path)
    logger.info(f"Saved HF-usage grid to {save_path}")
    plt.close(fig)


@click.command()
@click.option("--seed", default=42, help="Random seed for dataset generation + training")
@click.option("--latent-size", default=32, help="Hidden size for the LF/HF MLPs")
@click.option("--classifier-epochs", default=200, help="Epochs to train the LF/HF classifiers")
@click.option("--classifier-lr", default=1e-3, help="LR for the LF/HF classifiers")
@click.option("--gate-epochs", default=40, help="Epochs to train each r value's gate model")
@click.option("--gate-lr", default=3e-4, help="LR for the gate model (lowered automatically if routing signal is thin)")
@click.option("--num-targets", default=10, help="Number of usage targets to search for (10 -> 10%, 20%, ..., 100%), same convention as main.py's usage_values sweep")
@click.option("--fps", default=2, help="Frames per second in the output video")
@click.option(
    "--construction", default="high-ceiling", type=click.Choice(["hard", "high-ceiling"]),
    help="'hard' (default): build_toy_datasets, lf=hf+extra_noise, confirmed low disagreement-probe ceiling. "
         "'high-ceiling': build_high_ceiling_toy_datasets, independent LF/HF draws, for the appendix intuition figure."
)
@click.option("--output", default=os.path.join("results", "toy_test", "images", "gate_routing_video_200_v2.mp4"))
def main(seed, latent_size, classifier_epochs, classifier_lr, gate_epochs, gate_lr,
         num_targets, fps, construction, output):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"BUILDING TOY_2D DATASET (construction={construction})")
    build_fn = build_toy_datasets if construction == "hard" else build_high_ceiling_toy_datasets
    train_ds, test_ds, val_ds = build_fn(seed)

    batch_size = 128
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
    test_tensors = compute_gate_tensors(lf_model, hf_model, test_ds, hf_input_mode, device)

    logger.info("RUNNING DISAGREEMENT PROBE (input-quality vs. training-dynamics diagnostic)")
    train_early, train_final, train_lf_correct, train_hf_correct = compute_diagnostic_latents(
        lf_model, hf_model, train_ds, hf_input_mode, device
    )
    val_early, val_final, val_lf_correct, val_hf_correct = compute_diagnostic_latents(
        lf_model, hf_model, val_ds, hf_input_mode, device
    )
    run_disagreement_probe(
        train_final, val_final, train_lf_correct, train_hf_correct, val_lf_correct, val_hf_correct,
        latent_name="final latent", device=device
    )
    run_disagreement_probe(
        train_early, val_early, train_lf_correct, train_hf_correct, val_lf_correct, val_hf_correct,
        latent_name="early latent", device=device
    )

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
    # epochs regardless of cost. (class_weighted per-batch reweighting used to
    # be part of this too - removed, see main.py's comment on imbalanced_gate.)
    lf_correct = train_tensors[3]
    hf_correct = train_tensors[4]
    hf_needed_rate = (hf_correct & ~lf_correct).float().mean().item()
    logger.info(f"HF-needed rate on train set: {hf_needed_rate:.2%}")
    imbalanced_gate = hf_needed_rate < 0.15
    effective_gate_lr = 5e-5 if imbalanced_gate else gate_lr
    grad_clip_norm = 1.0 if imbalanced_gate else None

    # --- NEW: global (not per-batch) disagreement upweighting via sampler ---
    disagreement = (hf_correct & ~lf_correct) | (lf_correct & ~hf_correct)
    num_disagree = disagreement.sum().item()
    num_agree = disagreement.numel() - num_disagree
    global_disagree_weight = (num_agree / num_disagree) if num_disagree > 0 else 1.0
    logger.info(f"Global disagreement weight: {global_disagree_weight:.2f} ({num_disagree} disagreement / {num_agree} agreement samples)")

    sample_weights = torch.where(
        disagreement,
        torch.full_like(disagreement, global_disagree_weight, dtype=torch.float),
        torch.ones_like(disagreement, dtype=torch.float),
    )
    gate_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    gate_input_size = train_tensors[0].shape[1]
    fe_model = build_model(model_name="mlp", input_size=gate_input_size, output_size=2, latent_size=256)
    fe_model = fe_model.to(device)

    gate_train_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_tensors), batch_size=batch_size, sampler=gate_sampler
    )
    gate_val_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*val_tensors), batch_size=batch_size)
    gate_test_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*test_tensors), batch_size=batch_size)

    model_folder = os.path.join(os.path.dirname(output), "..", "models")
    file_folder = os.path.join(os.path.dirname(output), "..", "files")
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(file_folder, exist_ok=True)

    # Same bisection search main.py's AdaptiveGridSearch runs, instead of
    # blindly training a gate at hundreds of evenly-spaced ch values (most of
    # which land on the same handful of usage levels once the gate saturates -
    # see the "stuck at 0%, then a run of near-duplicate ~40% usage" pattern
    # that motivated this change) - this targets exactly num_targets usage
    # levels directly, the same way main.py's Pareto curve does.
    search = AdaptiveGridSearch(
        fe_model=fe_model, device=device, train_dl=gate_train_dl, val_dl=gate_val_dl, test_dl=gate_test_dl,
        model_folder=model_folder, gate_epochs=gate_epochs, gate_lr=effective_gate_lr,
        gate_grad_clip_norm=grad_clip_norm, seed=seed
    )

    # Descending (100% -> 10%), not ascending: the *first* target searched
    # gets no cached points to build a bracket from (find_bracket falls back
    # to a raw [0.001, 1] bisection, whose first midpoint - 0.5005 - is deep
    # in this task's "always LF" collapse zone - see build_toy_datasets'
    # docstring). Searching the easy, high-usage targets first means the hard
    # low-usage targets inherit real nearby (r, usage) points to bracket from
    # instead of that same blind, collapse-prone fallback every time.
    usage_values = list(np.linspace(1 / num_targets, 1.0, num_targets))[::-1]
    logger.info(f"Searching gate ch values for usage targets: {[f'{u:.0%}' for u in usage_values]}")
    search.run_reruns(usage_values=usage_values, n_reruns=1)

    frames = []
    swap_frames = []
    saved_target_usage = []
    saved_ch = []
    saved_usage_pct = []
    saved_accuracy = []
    saved_choice = []
    lf_correct_all = None
    hf_correct_all = None

    # Display order stays ascending (10% -> 100%) regardless of search order.
    for diag in sorted(search.search_diagnostics, key=lambda d: d["target_usage"]):
        r_val = diag["final_r"]

        fe_model.load_state_dict(
            torch.load(os.path.join(model_folder, f"fe_model-{r_val}.pt"), weights_only=True)
        )
        fe_model = fe_model.to(device)

        routing = compute_gate_routing_details(fe_model, all_dl, device)
        choice = routing["choice"]
        hf_usage_pct = 100 * choice.mean()
        routed_correct = np.where(choice == 1, routing["hf_correct"], routing["lf_correct"])
        accuracy = routed_correct.mean()

        # lf_correct/hf_correct only depend on the fixed lf_model/hf_model, not
        # on which gate produced this target's choice - identical every loop
        # iteration, so grabbed once rather than duplicated per target.
        if lf_correct_all is None:
            lf_correct_all = routing["lf_correct"]
            hf_correct_all = routing["hf_correct"]

        frames.append((float(r_val), hf_usage_pct, all_lf_points, choice, accuracy))

        # Same routing choice, but HF-routed points are plotted at their
        # actual HF coordinates instead of LF coordinates - shows where the
        # gate's picks actually land in HF space as ch relaxes.
        swap_points = np.where(choice[:, None] == 1, all_hf_points, all_lf_points)
        swap_frames.append((float(r_val), hf_usage_pct, swap_points, choice, accuracy))

        saved_target_usage.append(diag["target_usage"])
        saved_ch.append(float(r_val))
        saved_usage_pct.append(float(hf_usage_pct))
        saved_accuracy.append(float(accuracy))
        saved_choice.append(choice)

        logger.info(
            f"target usage={diag['target_usage']:.0%} -> ch={r_val:.4f}, "
            f"actual HF usage={hf_usage_pct:.1f}%, acc={100 * accuracy:.1f}%"
        )

    logger.info("SAVING RAW SAMPLES")
    samples_path = os.path.join(file_folder, "samples.npz")
    np.savez(
        samples_path,
        construction=construction,
        seed=seed,
        lf_points=all_lf_points,          # (N, 2) float32 - LF features
        hf_points=all_hf_points,          # (N, 2) float32 - HF features
        labels=all_labels.numpy(),        # (N,) int64 - ground-truth class
        lf_correct=lf_correct_all,        # (N,) bool - fixed, independent of target
        hf_correct=hf_correct_all,        # (N,) bool - fixed, independent of target
        target_usage=np.array(saved_target_usage),  # (num_targets,) requested usage fraction
        ch=np.array(saved_ch),                        # (num_targets,) the c_h found for each target
        usage_pct=np.array(saved_usage_pct),          # (num_targets,) actual achieved HF usage %
        accuracy=np.array(saved_accuracy),            # (num_targets,) accuracy at each target
        choice=np.stack(saved_choice, axis=0),         # (num_targets, N) 0=LF/1=HF per sample per target
    )
    logger.info(f"Saved raw samples to {samples_path}")

    logger.info("RENDERING ROUTING VIDEO")
    render_routing_video(frames, output, fps)

    swap_output = os.path.join(
        os.path.dirname(output), os.path.splitext(os.path.basename(output))[0] + "_hf_swap" + os.path.splitext(output)[1]
    )
    logger.info("RENDERING HF-SWAP ROUTING VIDEO")
    render_routing_video(swap_frames, swap_output, fps)

    logger.info("RENDERING HF-USAGE GRID")
    grid_output = os.path.join(
        os.path.dirname(output), os.path.splitext(os.path.basename(output))[0] + "_hf_usage_grid.png"
    )
    render_hf_usage_grid(frames, swap_frames, all_lf_points, all_hf_points, all_labels, grid_output)


if __name__ == "__main__":
    main()
