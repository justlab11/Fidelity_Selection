"""Shared diagnostic-suite functions - per-sample routing CSV construction,
entry-usage-vs-gain analysis, never-escalated gain-distribution check, and
class/scene-level clustering check - usable from any dataset's sweep
(standalone script or main.py's run_diagnostic_suite step), not copy-pasted
per dataset.

Consolidates cub_per_sample_routing_csv.py / cub_entry_usage_5run_avg.py /
cub_never_escalated_gain_check.py / cub_class_never_escalated_check.py (each
kept, unchanged, as the historical CUB-only version - see src/historical/)
into one module, per the LogSeededGreedySearch consolidation plan. Every
function here was already generalized via an identity_fn parameter before
this move (first proven out on CUB, then reused as-is for crop/LLVIP) - this
promotion doesn't change behavior, only location.

Run from src/ (demo/validation mode - operates on CUB's already-completed
5-rerun CSVs):
    ../venv/Scripts/python.exe diagnostics.py
"""
import os
import functools

import torch  # import before pandas - see this module's DLL-load-order note below
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, chi2_contingency

from sample_identity import get_test_sample_identity

# On Windows, importing pandas before torch causes torch's c10.dll to fail to
# load (a DLL-load-order conflict, not a missing dependency - both imports
# work fine individually, and torch-then-pandas works every time). The bare
# `import torch` above, kept first, is the fix - don't reorder it below pandas.

N_RERUNS = 5
USAGE_COLS = [f"fe_{i/10:.1f}" for i in range(1, 11)]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
POINT_COLOR = "#2a78d6"
NEVER_COLOR = "#d6453d"
ENTERED_COLOR = "#2a78d6"

_DEFAULT_IDENTITY_FN = functools.partial(get_test_sample_identity, "cub")


def build_routing_dataframe(target_usage: np.ndarray, choice: np.ndarray, idx: np.ndarray,
                             filenames: list, class_names: list) -> pd.DataFrame:
    """target_usage: (10,), choice/idx: (10, N). filenames/class_names: length-N,
    in true idx order (index i describes sample idx=i). Each target's
    choice/idx arrays are scattered into the output by idx explicitly
    (col[idx] = choice), not assumed to already be in idx order - safe
    regardless of whether a given routing snapshot happens to be arange(N)
    order."""
    n = len(filenames)
    columns = {}
    for t_idx, usage in enumerate(target_usage):
        col = np.full(n, -1, dtype=int)  # -1 = sentinel, should never survive if idx covers 0..n-1
        col[idx[t_idx]] = choice[t_idx]
        assert (col >= 0).all(), f"usage={usage}: idx array doesn't cover all {n} samples"
        columns[f"fe_{usage:.1f}"] = col

    df = pd.DataFrame({"filename": filenames, "class": class_names, **columns})
    return df


def load_loss_gain(checkpoint, filenames, identity_fn=None):
    """lf_loss - hf_loss per test sample, indexed by filename (not position) -
    joined the same way the routing CSVs are, for consistency.

    identity_fn: a get_test_sample_identity()-shaped callable (no args)
    returning (filenames, classes) in idx order for this dataset - defaults
    to CUB's (sample_identity.get_test_sample_identity("cub")); pass
    functools.partial(get_test_sample_identity, "crop"/"llvip") for those.
    """
    test_latent_folder = os.path.join("results", checkpoint, "latent", "test")
    files = sorted(os.listdir(test_latent_folder))
    gain_by_idx = {}
    for fname in files:
        data = torch.load(os.path.join(test_latent_folder, fname), weights_only=True)
        gain_by_idx[int(data["idx"])] = float(data["lf_loss"]) - float(data["hf_loss"])

    if identity_fn is None:
        identity_fn = _DEFAULT_IDENTITY_FN
    id_filenames, _ = identity_fn()
    gain_by_filename = {id_filenames[idx]: g for idx, g in gain_by_idx.items()}
    return np.array([gain_by_filename[f] for f in filenames])


def per_rerun_entry_usage_and_monotonicity(df: pd.DataFrame):
    """df: one rerun's per-sample routing CSV (filename, class, fe_0.1..fe_1.0),
    in a fixed row order. Returns (entry_usage (N,) with NaN for never-entered,
    non_monotonic_mask (N,) bool, ever_hf (N,) bool)."""
    choice = df[USAGE_COLS].to_numpy()  # (N, 10), columns already in increasing-usage order
    usage_targets = np.array([float(c[3:]) for c in USAGE_COLS])

    ever_hf = choice.max(axis=1) == 1
    first_hf_idx = np.argmax(choice == 1, axis=1)
    entry_usage = np.full(len(df), np.nan)
    entry_usage[ever_hf] = usage_targets[first_hf_idx[ever_hf]]

    non_monotonic = np.zeros(len(df), dtype=bool)
    for t in range(choice.shape[1] - 1):
        non_monotonic |= (choice[:, t] == 1) & np.any(choice[:, t + 1:] == 0, axis=1)

    return entry_usage, non_monotonic, ever_hf


def _load_aligned_reruns(checkpoint, n_reruns):
    file_folder = os.path.join("results", checkpoint, "files")
    dfs = [pd.read_csv(os.path.join(file_folder, f"per_sample_routing_rerun{i}.csv")) for i in range(n_reruns)]
    canonical_filenames = dfs[0]["filename"].to_numpy()
    class_by_filename = dfs[0]["class"].to_numpy()
    aligned_dfs = []
    for i, df in enumerate(dfs):
        df_indexed = df.set_index("filename")
        missing = set(canonical_filenames) - set(df_indexed.index)
        assert not missing, f"rerun {i} CSV is missing {len(missing)} filenames present in rerun 0"
        aligned_dfs.append(df_indexed.loc[canonical_filenames].reset_index())
    return aligned_dfs, canonical_filenames, class_by_filename


def analyze_entry_usage(checkpoint, label, identity_fn=None, n_reruns=N_RERUNS):
    """Entry-usage-vs-gain analysis, averaged across n_reruns fresh reruns'
    per_sample_routing_rerun{i}.csv files. A sample's per-rerun entry-usage is
    NaN in any rerun where it's never routed to HF across all 10 targets; the
    per-sample averaged entry-usage is the mean over only the reruns where it
    DID enter (np.nanmean). A sample that never enters in ANY rerun stays NaN,
    reported separately as "never escalated in any rerun", excluded from the
    Spearman correlation."""
    print(f"\n=== {label} ({checkpoint}) ===")
    aligned_dfs, canonical_filenames, _ = _load_aligned_reruns(checkpoint, n_reruns)
    n = len(canonical_filenames)

    entry_usage_per_rerun = np.full((n_reruns, n), np.nan)
    non_monotonic_rate_per_rerun = np.zeros(n_reruns)
    never_rate_per_rerun = np.zeros(n_reruns)

    for i, df in enumerate(aligned_dfs):
        entry_usage, non_monotonic, ever_hf = per_rerun_entry_usage_and_monotonicity(df)
        entry_usage_per_rerun[i] = entry_usage
        non_monotonic_rate_per_rerun[i] = 100 * non_monotonic.mean()
        never_rate_per_rerun[i] = 100 * (~ever_hf).mean()

    print("Per-rerun non-monotonic rate (%): " + ", ".join(f"{r:.2f}" for r in non_monotonic_rate_per_rerun))
    print("Per-rerun never-escalated rate (%): " + ", ".join(f"{r:.2f}" for r in never_rate_per_rerun))
    print(f"Averaged across {n_reruns} reruns: non-monotonic={non_monotonic_rate_per_rerun.mean():.2f}%  "
          f"never-escalated={never_rate_per_rerun.mean():.2f}%")

    with np.errstate(invalid="ignore"):
        entry_usage_avg = np.nanmean(entry_usage_per_rerun, axis=0)
        entry_usage_std = np.nanstd(entry_usage_per_rerun, axis=0)
        n_reruns_entered = np.sum(~np.isnan(entry_usage_per_rerun), axis=0)

    never_in_any = np.isnan(entry_usage_avg)
    n_never_in_any = int(never_in_any.sum())
    print(f"Samples never escalated in ANY of the {n_reruns} reruns: {n_never_in_any} ({100*n_never_in_any/n:.2f}%)")

    valid_std_pp = entry_usage_std[n_reruns_entered >= 2] * 100
    if len(valid_std_pp) > 0:
        print(f"Per-sample entry-usage std across reruns (n={len(valid_std_pp)} samples entering in >=2 reruns): "
              f"mean={valid_std_pp.mean():.2f}pp  median={np.median(valid_std_pp):.2f}pp  max={valid_std_pp.max():.2f}pp")

    gain = load_loss_gain(checkpoint, canonical_filenames, identity_fn=identity_fn)

    valid = ~never_in_any
    rho, pval = spearmanr(gain[valid], entry_usage_avg[valid])
    print(f"Spearman(gain, {n_reruns}-rerun-avg entry_usage) over entered samples (n={valid.sum()}): "
          f"rho={rho:.4f}  p={pval:.2e}")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRIDLINE, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)

    ax.scatter(gain[valid], entry_usage_avg[valid] * 100, s=10, color=POINT_COLOR, alpha=0.3, linewidths=0,
               label=f"Entered samples (n={valid.sum()})")
    if n_never_in_any > 0:
        ax.scatter(gain[never_in_any], np.full(n_never_in_any, 105), s=10, color=NEVER_COLOR, alpha=0.4,
                   linewidths=0, marker="^",
                   label=f"Never escalated in any rerun (n={n_never_in_any}, plotted at y=105)")

    ax.set_xlabel("Oracle gain (lf_loss - hf_loss)", color=INK_SECONDARY)
    ax.set_ylabel(f"{n_reruns}-rerun-averaged entry usage (%)", color=INK_PRIMARY)
    ax.set_title(f"{label} - gain vs. entry-usage ({n_reruns}-rerun avg, Spearman ρ={rho:.3f})",
                 color=INK_PRIMARY, fontweight="bold")
    ax.tick_params(colors=INK_MUTED)
    legend = ax.legend(frameon=False, loc="center left")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    save_path = os.path.join("results", checkpoint, "images", "entry_usage_vs_gain_5run_avg.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {save_path}")


def analyze_entry_usage_boxplot(checkpoint, label, identity_fn=None, n_reruns=N_RERUNS):
    """Box-and-whisker version of analyze_entry_usage's scatter: one box per
    c_h value (the same per-target mean final_r values plot_usage_vs_ch_std
    plots on its own x-axis, read back from gate_search_diagnostics.csv -
    not recomputed), showing the distribution of oracle gain
    (lf_loss - hf_loss) for samples that FIRST entered HF at that exact
    target, pooled across all n_reruns reruns (so a sample entering at the
    same target in multiple reruns contributes its fixed gain value once per
    rerun - more data per box, same spirit as plot_usage_vs_ch_std averaging
    over reruns rather than picking one). A separate box holds samples never
    escalated in ANY of the n_reruns reruns (the same never_in_any
    definition analyze_entry_usage/check_never_escalated_gain use), not
    pooled per-rerun since by definition there's no entering rerun to
    attribute repeats to.

    Whiskers are fixed at the 5th/95th percentile (not matplotlib's default
    1.5*IQR), outlier markers are hidden (showfliers=False) since whis=(5,95)
    already puts 10% of each box's population outside the whiskers by
    construction - drawing all of those as individual points would be
    exactly the clutter a 5/95 whisker choice is meant to avoid.
    """
    file_folder = os.path.join("results", checkpoint, "files")
    aligned_dfs, canonical_filenames, _ = _load_aligned_reruns(checkpoint, n_reruns)
    n = len(canonical_filenames)

    entry_usage_per_rerun = np.full((n_reruns, n), np.nan)
    for i, df in enumerate(aligned_dfs):
        entry_usage, _, _ = per_rerun_entry_usage_and_monotonicity(df)
        entry_usage_per_rerun[i] = entry_usage

    with np.errstate(invalid="ignore"):
        entry_usage_avg = np.nanmean(entry_usage_per_rerun, axis=0)
    never_in_any = np.isnan(entry_usage_avg)

    gain = load_loss_gain(checkpoint, canonical_filenames, identity_fn=identity_fn)

    # Per-target mean c_h (final_r) - same values plot_usage_vs_ch_std.png
    # itself plots - read back from disk rather than recomputed, kept only
    # to confirm which target each box corresponds to (no longer the axis
    # label itself - see below). Sorted by target_usage ascending, not by
    # c_h: usage(c_h) is the thing with real paper-reader-facing meaning, and
    # sorting on it directly is exact regardless of whether c_h's own
    # ordering lines up pointwise (it usually does, since usage falls as
    # c_h rises, but isn't guaranteed to given reruns/ties).
    search_diag = pd.read_csv(os.path.join(file_folder, "gate_search_diagnostics.csv"))
    ch_by_target = search_diag.groupby("target_usage")["final_r"].mean()
    targets_sorted = np.sort(ch_by_target.index.to_numpy())

    # Targets can tie on the same converged c_h (e.g. once a checkpoint hits
    # a usage ceiling, every higher target reuses the same r≈0 evaluation) -
    # "first entry" logic then credits every such sample to the *lowest* of
    # the tied targets (see per_rerun_entry_usage_and_monotonicity's
    # argmax(choice==1)), leaving the other tied targets with zero new
    # entries. Skipped here rather than drawn as empty boxes - an empty box
    # isn't a distribution, and a run of blank columns reads as missing data
    # rather than the real "ceiling" signal it actually is.
    box_data = []
    box_labels = []
    box_colors = []
    for target in targets_sorted:
        mask_any_rerun = (entry_usage_per_rerun == target)  # (n_reruns, n) bool
        gains_this_box = np.tile(gain, (n_reruns, 1))[mask_any_rerun]
        if gains_this_box.size == 0:
            continue
        box_data.append(gains_this_box)
        box_labels.append(f"{100*target:.0f}")
        box_colors.append(ENTERED_COLOR)

    box_data.append(gain[never_in_any])
    box_labels.append("Never\nescalated")
    box_colors.append(NEVER_COLOR)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)

    positions = np.arange(1, len(box_data) + 1)
    bp = ax.boxplot(
        box_data, positions=positions, whis=(5, 95), showfliers=False,
        patch_artist=True, widths=0.6,
        medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
        whiskerprops=dict(color=INK_SECONDARY, linewidth=1.2),
        capprops=dict(color=INK_SECONDARY, linewidth=1.2),
        boxprops=dict(linewidth=1.0, edgecolor=INK_SECONDARY),
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.axhline(0, color=INK_MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(box_labels, color=INK_SECONDARY)
    ax.set_xlabel("Target usage (%, ascending)", color=INK_SECONDARY)
    ax.set_ylabel("Oracle gain (lf_loss - hf_loss)", color=INK_PRIMARY)
    ax.set_title(f"{label} - gain by entry usage ({n_reruns}-rerun pooled, 5th-95th pct whiskers)",
                 color=INK_PRIMARY, fontweight="bold")
    ax.tick_params(colors=INK_MUTED)

    fig.tight_layout()
    save_path = os.path.join("results", checkpoint, "images", "entry_usage_vs_gain_boxplot_by_ch.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # bbox_inches="tight": the bold title can run wider than the fixed
    # figsize at some label counts (e.g. long dataset labels) - this expands
    # the saved canvas to fit everything instead of clipping it.
    fig.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def check_never_escalated_gain(checkpoint, label, identity_fn=None, n_reruns=N_RERUNS):
    """Checks whether samples never escalated in any of n_reruns reruns are
    concentrated toward negative/low lf_loss - hf_loss gain (the gate
    correctly avoiding HF-harmful samples) or spread similarly to entered
    samples (never-escalated just reflecting general low priority)."""
    aligned_dfs, canonical_filenames, _ = _load_aligned_reruns(checkpoint, n_reruns)
    n = len(canonical_filenames)
    entry_usage_per_rerun = np.full((n_reruns, n), np.nan)
    for i, df in enumerate(aligned_dfs):
        entry_usage, _, _ = per_rerun_entry_usage_and_monotonicity(df)
        entry_usage_per_rerun[i] = entry_usage

    with np.errstate(invalid="ignore"):
        entry_usage_avg = np.nanmean(entry_usage_per_rerun, axis=0)
    never_in_any = np.isnan(entry_usage_avg)

    gain = load_loss_gain(checkpoint, canonical_filenames, identity_fn=identity_fn)
    gain_never = gain[never_in_any]
    gain_entered = gain[~never_in_any]

    print(f"=== {label} ({checkpoint}) ===")
    print(f"Never escalated in any rerun: n={len(gain_never)}")
    if len(gain_never) == 0:
        print("  (empty - this checkpoint's gate escalates every test sample in at least one of "
              f"the {n_reruns} reruns, so there's no never-escalated group to compare against. "
              "Not an error - this can happen when a checkpoint reaches a clean 100% usage ceiling.)")
    else:
        print(f"  mean={gain_never.mean():.4f}  median={np.median(gain_never):.4f}  "
              f"std={gain_never.std():.4f}  min={gain_never.min():.4f}  max={gain_never.max():.4f}")
        print(f"  fraction with gain > 0 (HF genuinely better): {100*(gain_never > 0).mean():.2f}%")
        print(f"  fraction with gain < 0 (LF genuinely better): {100*(gain_never < 0).mean():.2f}%")

    print(f"\nEntered in >=1 rerun: n={len(gain_entered)}")
    print(f"  mean={gain_entered.mean():.4f}  median={np.median(gain_entered):.4f}  "
          f"std={gain_entered.std():.4f}  min={gain_entered.min():.4f}  max={gain_entered.max():.4f}")
    print(f"  fraction with gain > 0 (HF genuinely better): {100*(gain_entered > 0).mean():.2f}%")
    print(f"  fraction with gain < 0 (LF genuinely better): {100*(gain_entered < 0).mean():.2f}%")

    if len(gain_never) == 0:
        print("\nSkipping Mann-Whitney test and never-vs-entered histogram - no never-escalated group.")
    else:
        stat, pval = mannwhitneyu(gain_never, gain_entered, alternative="two-sided")
        print(f"\nMann-Whitney U test (never-escalated vs. entered gain distributions): "
              f"U={stat:.1f}  p={pval:.2e}")

        diff = gain_entered.mean() - gain_never.mean()
        print(f"\nMean gain difference (entered - never-escalated): {diff:+.4f}")
        if diff > 0 and pval < 0.05:
            print("-> Never-escalated samples DO skew toward lower gain than entered samples "
                  "(statistically significant) - consistent with the gate avoiding HF-harmful "
                  "samples, though check the histogram below for how much the distributions actually overlap.")
        else:
            print("-> No significant / no lower-gain skew detected for never-escalated samples.")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRIDLINE, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)

    bins = np.linspace(min(gain.min(), -5), max(gain.max(), 5), 60)
    ax.hist(gain_entered, bins=bins, density=True, color=ENTERED_COLOR, alpha=0.5,
            label=f"Entered ≥ 1 rerun (n={len(gain_entered)}, mean={gain_entered.mean():.2f})")
    ax.axvline(gain_entered.mean(), color=ENTERED_COLOR, linewidth=1.5, linestyle="--")
    if len(gain_never) > 0:
        ax.hist(gain_never, bins=bins, density=True, color=NEVER_COLOR, alpha=0.5,
                label=f"Never escalated (n={len(gain_never)}, mean={gain_never.mean():.2f})")
        ax.axvline(gain_never.mean(), color=NEVER_COLOR, linewidth=1.5, linestyle="--")
    ax.axvline(0, color=INK_MUTED, linewidth=1, linestyle=":")

    ax.set_xlabel("Oracle gain (lf_loss - hf_loss)", color=INK_SECONDARY)
    ax.set_ylabel("Density", color=INK_PRIMARY)
    ax.set_title(f"{label} - gain distribution: never-escalated vs. entered", color=INK_PRIMARY, fontweight="bold")
    ax.tick_params(colors=INK_MUTED)
    legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    save_path = os.path.join("results", checkpoint, "images", "never_escalated_gain_distribution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved {save_path}")


def check_class_clustering(checkpoint, label, identity_fn=None, n_reruns=N_RERUNS):
    """Checks whether never-escalated status (never in ANY of n_reruns
    reruns - a genuine per-sample label) clusters by class/scene, via a
    chi-square test of independence, plus a class-level correlation between
    mean gain and never-escalated rate."""
    file_folder = os.path.join("results", checkpoint, "files")
    aligned_dfs, canonical_filenames, class_by_filename = _load_aligned_reruns(checkpoint, n_reruns)
    n = len(canonical_filenames)
    entry_usage_per_rerun = np.full((n_reruns, n), np.nan)
    for i, df in enumerate(aligned_dfs):
        entry_usage, _, _ = per_rerun_entry_usage_and_monotonicity(df)
        entry_usage_per_rerun[i] = entry_usage

    with np.errstate(invalid="ignore"):
        entry_usage_avg = np.nanmean(entry_usage_per_rerun, axis=0)
    never_in_any = np.isnan(entry_usage_avg)
    gain = load_loss_gain(checkpoint, canonical_filenames, identity_fn=identity_fn)

    baseline_rate = 100 * never_in_any.mean()
    print(f"=== {label} ({checkpoint}) ===")
    print(f"Dataset-wide never-escalated rate (per-sample, never in ANY of {n_reruns} reruns): "
          f"{baseline_rate:.2f}% ({never_in_any.sum()}/{n})")

    df = pd.DataFrame({
        "class": class_by_filename, "gain": gain, "never_escalated": never_in_any,
    })

    per_class = df.groupby("class").agg(
        n=("gain", "size"),
        never_rate_pct=("never_escalated", lambda s: 100 * s.mean()),
        mean_gain=("gain", "mean"),
        median_gain=("gain", "median"),
    ).reset_index()
    per_class["deviation_pp"] = per_class["never_rate_pct"] - baseline_rate

    out_path = os.path.join(file_folder, "class_never_escalated_summary.csv")
    per_class.sort_values("deviation_pp", ascending=False).to_csv(out_path, index=False)
    print(f"Full {len(per_class)}-class table saved to {out_path}")
    print(f"Class sample counts: min={per_class['n'].min()}  median={per_class['n'].median():.0f}  "
          f"max={per_class['n'].max()}")

    top_n = 15
    print(f"\n--- Top {top_n} classes by HIGHEST never-escalated rate (most above baseline) ---")
    top = per_class.sort_values("deviation_pp", ascending=False).head(top_n)
    for _, row in top.iterrows():
        print(f"  {str(row['class']):<45s} n={row['n']:3.0f}  never_rate={row['never_rate_pct']:6.2f}%  "
              f"dev={row['deviation_pp']:+6.2f}pp  mean_gain={row['mean_gain']:+.3f}")

    print(f"\n--- Top {top_n} classes by LOWEST never-escalated rate (most below baseline) ---")
    bottom = per_class.sort_values("deviation_pp", ascending=True).head(top_n)
    for _, row in bottom.iterrows():
        print(f"  {str(row['class']):<45s} n={row['n']:3.0f}  never_rate={row['never_rate_pct']:6.2f}%  "
              f"dev={row['deviation_pp']:+6.2f}pp  mean_gain={row['mean_gain']:+.3f}")

    if df["never_escalated"].nunique() < 2:
        print("\nChi-square test skipped: never_escalated has no variation at all "
              f"(every sample is {df['never_escalated'].iloc[0]}) - this checkpoint has no "
              "never-escalated samples to test for class clustering (consistent with it reaching "
              f"a clean 100% usage ceiling across all {n_reruns} reruns).")
    else:
        contingency = pd.crosstab(df["class"], df["never_escalated"])
        for col in (True, False):
            if col not in contingency.columns:
                contingency[col] = 0
        chi2, pval, dof, expected = chi2_contingency(contingency)
        min_expected = expected.min()
        print(f"\nChi-square test (never_escalated ~ class): chi2={chi2:.2f}  dof={dof}  p={pval:.2e}")
        print(f"  min expected cell count={min_expected:.2f} "
              f"({'OK' if min_expected >= 5 else 'WARNING: below the usual >=5 rule of thumb for chi-square validity'})")

    if per_class["never_rate_pct"].nunique() < 2:
        print(f"\nClass-level correlation skipped: never_rate_pct is constant across all {len(per_class)} classes "
              f"(={per_class['never_rate_pct'].iloc[0]:.2f}%) - no variation to correlate against mean_gain.")
    else:
        rho_s, p_s = spearmanr(per_class["mean_gain"], per_class["never_rate_pct"])
        rho_p, p_p = pearsonr(per_class["mean_gain"], per_class["never_rate_pct"])
        print(f"\nClass-level correlation (mean_gain vs. never_rate_pct), n={len(per_class)} classes:")
        print(f"  Spearman rho={rho_s:.4f}  p={p_s:.2e}")
        print(f"  Pearson  r  ={rho_p:.4f}  p={p_p:.2e}")

    print("\n--- Classes sorted by mean_gain (lowest 10) vs. their never_rate ---")
    low_gain = per_class.sort_values("mean_gain").head(10)
    for _, row in low_gain.iterrows():
        print(f"  {str(row['class']):<45s} n={row['n']:3.0f}  mean_gain={row['mean_gain']:+.3f}  "
              f"never_rate={row['never_rate_pct']:6.2f}%")
    print("--- Classes sorted by mean_gain (highest 10) vs. their never_rate ---")
    high_gain = per_class.sort_values("mean_gain", ascending=False).head(10)
    for _, row in high_gain.iterrows():
        print(f"  {str(row['class']):<45s} n={row['n']:3.0f}  mean_gain={row['mean_gain']:+.3f}  "
              f"never_rate={row['never_rate_pct']:6.2f}%")


def run_full_diagnostic_suite(checkpoint, label, identity_fn=None, n_reruns=N_RERUNS):
    """Runs all three diagnostics in sequence - the single entry point
    main.py's run_diagnostic_suite config flag calls."""
    analyze_entry_usage(checkpoint, label, identity_fn=identity_fn, n_reruns=n_reruns)
    check_never_escalated_gain(checkpoint, label, identity_fn=identity_fn, n_reruns=n_reruns)
    check_class_clustering(checkpoint, label, identity_fn=identity_fn, n_reruns=n_reruns)


if __name__ == "__main__":
    run_full_diagnostic_suite("bird_grayscale-resnet-resnet-42-256", "CUB resnet/resnet (gray/color)")
