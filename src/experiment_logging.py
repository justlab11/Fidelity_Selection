import csv
import os
import time
import logging

import torch

logger = logging.getLogger(__name__)


def count_parameters(model, trainable_only: bool = True) -> int:
    """Trainable-only is the right "cost realism" signal for models this
    pipeline actually trains (resnet/vit can be partially frozen via
    freeze_body()) - but a YOLO checkpoint is always fully frozen here (loaded
    pretrained, never trained), so trainable_only would report 0 regardless of
    its real size, badly understating its actual inference cost. Pass
    trainable_only=False for those.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def reset_peak_memory(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_mb(device) -> float:
    if torch.device(device).type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1e6


def measure_inference_latency(forward_fn, dataloader, device, num_batches: int = 20) -> float:
    """Average inference latency in ms/sample, timed over up to num_batches batches.
    forward_fn(batch) must return the model's output tensor (already moved to device)."""
    is_cuda = torch.device(device).type == "cuda"
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            if is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = forward_fn(batch)
            if is_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            batch_size = output.size(0) if hasattr(output, "size") else len(batch[0])
            total_time += elapsed
            total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return (total_time / total_samples) * 1000


class EpochMetricsLogger:
    """Appends one CSV row per epoch; (re)writes the header on the first call each
    construction, so a fresh run overwrites any log left by a previous run of the
    same experiment folder (matching experiment.log's own filemode='w' convention)."""
    def __init__(self, file_path: str, fieldnames: list):
        self.file_path = file_path
        self.fieldnames = fieldnames
        self._header_written = False

    def log_epoch(self, **row) -> None:
        mode = "w" if not self._header_written else "a"
        with open(self.file_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)
