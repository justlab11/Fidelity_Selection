import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolov5.utils.general import non_max_suppression, xywhn2xyxy
from yolov5.utils.metrics import box_iou

from datasets import LLVIPDataset
from models import build_yolov5

IMG_SIZE = 640
IOU_MATCH_THRESH = 0.5
CONF_THRES = 0.25
IOU_NMS_THRES = 0.45
BATCH_SIZE = 32

OUT_DIR = "results/llvip_analysis"
IMAGES_DIR = os.path.join(OUT_DIR, "images")
FILES_DIR = os.path.join(OUT_DIR, "files")


def match_detections(gt_xyxy, det_xyxy, det_conf, iou_thresh=IOU_MATCH_THRESH):
    """Greedy IoU matching: highest-confidence detections claim the
    best-IoU unclaimed ground-truth box first (standard COCO/VOC-style
    matching). Returns (tp_mask over detections, n_fn)."""
    n_det = det_xyxy.shape[0]
    n_gt = gt_xyxy.shape[0]
    tp = torch.zeros(n_det, dtype=torch.bool)

    if n_gt == 0 or n_det == 0:
        return tp, n_gt

    order = torch.argsort(det_conf, descending=True)
    claimed = torch.zeros(n_gt, dtype=torch.bool)
    ious = box_iou(det_xyxy, gt_xyxy)  # (n_det, n_gt)

    for i in order.tolist():
        row = ious[i].clone()
        row[claimed] = -1
        best_iou, best_j = row.max(dim=0)
        if best_iou >= iou_thresh:
            tp[i] = True
            claimed[best_j] = True

    n_fn = int((~claimed).sum())
    return tp, n_fn


def per_image_metrics(gt_boxes_norm, dets, img_size=IMG_SIZE):
    """gt_boxes_norm: (n, 5) [cls, xc, yc, w, h] normalized, real rows only
    (already filtered of -1 padding). dets: (n_det, 6) [x1,y1,x2,y2,conf,cls]
    tensor from NMS (may be empty). Both already share the same 640x640
    letterboxed coordinate frame, so no unletterboxing is needed here.

    Returns (metrics dict, per-detection confidences, per-detection tp flags).
    """
    n_gt = gt_boxes_norm.shape[0]
    gt_xyxy = xywhn2xyxy(gt_boxes_norm[:, 1:], w=img_size, h=img_size) if n_gt > 0 else torch.zeros((0, 4))

    if dets is None or dets.shape[0] == 0:
        det_xyxy = torch.zeros((0, 4))
        det_conf = torch.zeros((0,))
    else:
        det_xyxy = dets[:, :4].cpu()
        det_conf = dets[:, 4].cpu()

    tp_mask, n_fn = match_detections(gt_xyxy, det_xyxy, det_conf)
    n_tp = int(tp_mask.sum())
    n_det = det_xyxy.shape[0]
    n_fp = n_det - n_tp

    precision = n_tp / n_det if n_det > 0 else 1.0
    recall = n_tp / n_gt if n_gt > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "count": n_det, "tp": n_tp, "fp": n_fp, "fn": n_fn,
        "precision": precision, "recall": recall, "f1": f1,
    }
    return metrics, det_conf.tolist(), tp_mask.tolist()


def run_analysis(device):
    lf_model = build_yolov5("../models/yolov5_visible.pt", device=device).eval()
    hf_model = build_yolov5("../models/yolov5_infrared.pt", device=device).eval()
    print(f"Loaded LF (visible): stride={lf_model.stride} names={lf_model.names}")
    print(f"Loaded HF (infrared): stride={hf_model.stride} names={hf_model.names}")

    ds = LLVIPDataset(root="../data/LLVIP", split="test")
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test set size: {len(ds)}")

    records = []
    conf_pool = {"lf_tp": [], "lf_fp": [], "hf_tp": [], "hf_fp": []}

    idx = 0
    for lf_imgs, hf_imgs, labels in tqdm(dl, desc="Running inference"):
        with torch.no_grad():
            lf_preds = lf_model(lf_imgs.to(device))["output"]
            hf_preds = hf_model(hf_imgs.to(device))["output"]

        lf_dets_batch = non_max_suppression(lf_preds, conf_thres=CONF_THRES, iou_thres=IOU_NMS_THRES)
        hf_dets_batch = non_max_suppression(hf_preds, conf_thres=CONF_THRES, iou_thres=IOU_NMS_THRES)

        for b in range(labels.shape[0]):
            fname = ds.basenames[idx]
            gt_boxes = labels[b][labels[b][:, 0] >= 0]

            lf_metrics, lf_confs, lf_tp_flags = per_image_metrics(gt_boxes, lf_dets_batch[b])
            hf_metrics, hf_confs, hf_tp_flags = per_image_metrics(gt_boxes, hf_dets_batch[b])

            for conf, is_tp in zip(lf_confs, lf_tp_flags):
                conf_pool["lf_tp" if is_tp else "lf_fp"].append(conf)
            for conf, is_tp in zip(hf_confs, hf_tp_flags):
                conf_pool["hf_tp" if is_tp else "hf_fp"].append(conf)

            records.append({
                "fname": fname,
                "gt_count": int(gt_boxes.shape[0]),
                "lf_count": lf_metrics["count"], "hf_count": hf_metrics["count"],
                "lf_tp": lf_metrics["tp"], "lf_fp": lf_metrics["fp"], "lf_fn": lf_metrics["fn"],
                "hf_tp": hf_metrics["tp"], "hf_fp": hf_metrics["fp"], "hf_fn": hf_metrics["fn"],
                "lf_precision": lf_metrics["precision"], "lf_recall": lf_metrics["recall"], "lf_f1": lf_metrics["f1"],
                "hf_precision": hf_metrics["precision"], "hf_recall": hf_metrics["recall"], "hf_f1": hf_metrics["f1"],
            })
            idx += 1

    return pd.DataFrame(records), conf_pool


def summarize(df, conf_pool):
    print("\n=== Summary ===")
    print(f"Images analyzed: {len(df)}")
    print(f"Mean GT count/image: {df['gt_count'].mean():.2f}")
    print(f"Mean LF detections/image: {df['lf_count'].mean():.2f}  |  Mean HF detections/image: {df['hf_count'].mean():.2f}")
    print(f"Mean LF recall: {df['lf_recall'].mean():.3f}  |  Mean HF recall: {df['hf_recall'].mean():.3f}")
    print(f"Mean LF precision: {df['lf_precision'].mean():.3f}  |  Mean HF precision: {df['hf_precision'].mean():.3f}")
    print(f"Mean LF F1: {df['lf_f1'].mean():.3f}  |  Mean HF F1: {df['hf_f1'].mean():.3f}")

    print("\nConfidence by group (mean ± std, n):")
    for key, vals in conf_pool.items():
        if vals:
            arr = np.array(vals)
            print(f"  {key}: {arr.mean():.3f} ± {arr.std():.3f}  (n={len(arr)})")
        else:
            print(f"  {key}: (no detections)")

    lf_missed = df["lf_recall"] < 1.0
    hf_recovers = lf_missed & (df["hf_recall"] > df["lf_recall"])
    hf_no_help = lf_missed & (df["hf_recall"] <= df["lf_recall"])
    lf_sufficient = ~lf_missed
    hf_worse = df["hf_recall"] < df["lf_recall"]

    print("\nHF-needed partition (detection-task analogue of hf_needed = hf_correct & ~lf_correct):")
    print(f"  LF already found everyone (lf_recall == 1): {int(lf_sufficient.sum())} ({100*lf_sufficient.mean():.1f}%)")
    print(f"  HF recovers a miss LF made:                 {int(hf_recovers.sum())} ({100*hf_recovers.mean():.1f}%)")
    print(f"  HF doesn't help despite LF miss:            {int(hf_no_help.sum())} ({100*hf_no_help.mean():.1f}%)")
    print(f"  HF actually worse than LF:                  {int(hf_worse.sum())} ({100*hf_worse.mean():.1f}%)")

    return {
        "lf_sufficient": int(lf_sufficient.sum()),
        "hf_recovers": int(hf_recovers.sum()),
        "hf_no_help": int(hf_no_help.sum()),
        "hf_worse": int(hf_worse.sum()),
    }


def make_plots(df, conf_pool, buckets):
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # 1. detection count difference histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    diff = df["hf_count"] - df["lf_count"]
    ax.hist(diff, bins=range(int(diff.min()), int(diff.max()) + 2), color="#eb6834", edgecolor="white")
    ax.set_xlabel("HF detections - LF detections (per image)")
    ax.set_ylabel("Count")
    ax.set_title("Detection count difference: HF (infrared) - LF (visible)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "count_diff_histogram.png"), dpi=150)
    plt.close(fig)

    # 2. confidence by group: mean +/- std bar chart, split TP vs FP
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["lf_tp", "lf_fp", "hf_tp", "hf_fp"]
    means = [np.mean(conf_pool[k]) if conf_pool[k] else 0 for k in labels]
    stds = [np.std(conf_pool[k]) if conf_pool[k] else 0 for k in labels]
    colors = ["#2a78d6", "#2a78d6", "#eb6834", "#eb6834"]
    ax.bar(labels, means, yerr=stds, color=colors, alpha=0.6, capsize=5)
    ax.set_ylabel("Confidence")
    ax.set_title("Detection confidence by model and correctness")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "confidence_by_group.png"), dpi=150)
    plt.close(fig)

    # 3. recall histograms, LF vs HF overlaid
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(df["lf_recall"], bins=bins, alpha=0.5, label="LF (visible)", color="#2a78d6")
    ax.hist(df["hf_recall"], bins=bins, alpha=0.5, label="HF (infrared)", color="#eb6834")
    ax.set_xlabel("Per-image recall")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Per-image recall distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "recall_histogram.png"), dpi=150)
    plt.close(fig)

    # 4. F1 histograms, LF vs HF overlaid
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["lf_f1"], bins=bins, alpha=0.5, label="LF (visible)", color="#2a78d6")
    ax.hist(df["hf_f1"], bins=bins, alpha=0.5, label="HF (infrared)", color="#eb6834")
    ax.set_xlabel("Per-image F1")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Per-image F1 distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "f1_histogram.png"), dpi=150)
    plt.close(fig)

    # 5. hf-needed partition bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    bucket_labels = ["LF sufficient", "HF recovers miss", "HF no help", "HF worse"]
    bucket_values = [buckets["lf_sufficient"], buckets["hf_recovers"], buckets["hf_no_help"], buckets["hf_worse"]]
    ax.bar(bucket_labels, bucket_values, color=["#898781", "#2a9d59", "#eb6834", "#c0392b"])
    ax.set_ylabel("Image count")
    ax.set_title("HF-needed partition")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "hf_needed_partition.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved plots to {IMAGES_DIR}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df, conf_pool = run_analysis(device)

    os.makedirs(FILES_DIR, exist_ok=True)
    csv_path = os.path.join(FILES_DIR, "per_image_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved per-image metrics to {csv_path}")

    buckets = summarize(df, conf_pool)
    make_plots(df, conf_pool, buckets)


if __name__ == "__main__":
    main()
