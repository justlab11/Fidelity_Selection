import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import JaccardIndex
from typing import *
import json
import click
import os.path as path
import shutil
import logging
import os
import time

from datasets import *
from models import build_unet, CustomResNet18, LatentCNNHead, CustomMLP
from helpers import *
from comparisons import *
from losses import MetaLossFunction
from experiment_logging import (
    count_parameters, reset_peak_memory, get_peak_memory_mb,
    measure_inference_latency, format_duration, EpochMetricsLogger
)
from plot_gate_sensitivity import (
    load_gate_log, load_test_benefit, dedupe_and_sort, compute_usage_derivative,
    plot_derivative, find_zoom_range, plot_zoom, plot_full_curve, report_max_usage,
    plot_isotonic_smoothing
)
from plot_gate_routing import (
    load_routing_snapshots, compute_oracle_curve, plot_routing_projection_grid, plot_routing_table_grid
)
from plot_pareto_curves import load_pareto_curves, render_pareto_plot

from custom_types import ConfigOptions

@click.command()
@click.option("--config_file", default="../config.yml")
def main(config_file):
    # ================================================================
    # INITIALIZATION
    # config, folders, logging, seed, dataset/dataloaders, model building
    # ================================================================
    experiment_start_time = time.perf_counter()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load config
    config: ConfigOptions = load_yaml_options(config_file)
    seed: int = config.random_seed
    latent_size: int = config.latent_size
    dataset_name: str = config.dataset.name
    lf_model_name: str = config.classifier_training.lf_model
    hf_model_name: str = config.classifier_training.hf_model
    run_comparisons: bool = config.run_comparisons
    train_body: bool = config.train_body if dataset_name != "crop" else True
    hf_input_mode: str = config.classifier_training.hf_input_mode

    # yolov5's own module-level logging setup (triggered the first time it's
    # imported anywhere - e.g. by LLVIPDataset or build_model's "yolo" case,
    # both deferred imports) calls logging.config.dictConfig(), which
    # unconditionally closes every currently registered logging handler
    # process-wide (a documented dictConfig quirk: disable_existing_loggers
    # only protects loggers from being disabled, not handlers from being
    # closed). If that first import happens *after* our own logging.basicConfig
    # below, it silently kills the file handler and every log message from
    # then on goes nowhere - no exception, no warning, the run just stops
    # appearing in experiment.log while continuing to execute. Importing
    # yolov5 here, before our own basicConfig, makes its one-time logging setup
    # happen first, so ours is what's left standing afterward.
    if lf_model_name == "yolo" or hf_model_name == "yolo":
        import yolov5  # noqa: F401

    # create folders for the dataset
    folder_name: str = os.path.join("results", f"{dataset_name}-{lf_model_name}-{hf_model_name}-{seed}-{latent_size}")

    print(f"Results for this experiment located at '{os.path.abspath(folder_name)}'")
    print("Please see 'experiments.log' for the logs")

    model_folder: str = os.path.join(folder_name, "models")
    file_folder: str = os.path.join(folder_name, "files")
    image_folder: str = os.path.join(folder_name, "images")
    latent_folder: str = os.path.join(folder_name, "latent")

    folders = [model_folder, file_folder, image_folder, latent_folder]

    # if we are training the body, then we can't save the representations to files and use
    # them for the head to save time. we only make body_folder if we are not training it. 
    if not train_body:
        body_folder: str = os.path.join(folder_name, "body")
        folders.append(body_folder)

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # make sure to keep a copy of the config_file for any accidents
    shutil.copyfile(config_file, os.path.join(file_folder, os.path.basename(config_file)))

    # initialize logger 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(folder_name, "experiment.log"),
        filemode='w'
    )

    logger = logging.getLogger(__name__)
    logger.info("SETTINGS")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Folder: {folder_name}")
    logger.info(f"Dataset: {dataset_name.capitalize()}")
    logger.info(f"LF Model: {lf_model_name.capitalize()}")
    logger.info(f"HF Model: {hf_model_name.capitalize()}")
    logger.info(f"Latent Size: {latent_size}")
    logger.info(f"Run Comparisons: {run_comparisons}")

    logger.info(f"SETTING SEED")
    set_all_seeds(seed=seed)

    logger.info(f"BUILDING DATASET & DATALOADERS")
    train_ds, test_ds, val_ds = build_dataset(
        dataset_name=dataset_name,
        seed=seed,
        folder=config.dataset.folder
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=config.classifier_training.batch_size,
        shuffle=True
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=config.classifier_training.batch_size,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.classifier_training.batch_size,
    )
    train_size = len(train_ds)
    test_size = len(test_ds)
    val_size = len(val_ds)

    logger.info(f"Train Dataset Size: {train_size:,}")
    logger.info(f"Test Dataset Size: {test_size:,}")
    logger.info(f"Validation Dataset Size: {val_size:,}")

    # incrementally filled in as each phase completes, written to summary.json at
    # the end (and at the early "not run_comparisons" exit, with whatever's filled so far)
    run_summary: Dict = {
        "dataset": dataset_name,
        "lf_model": lf_model_name,
        "hf_model": hf_model_name,
        "seed": seed,
        "latent_size": latent_size,
        "dataset_sizes": {"train": train_size, "test": test_size, "val": val_size},
        "timing": {},
        "peak_gpu_memory_mb": {},
        "cost_realism": {},
        "final_accuracy": {},
        "result_files": []
    }

    logger.info(f"BUILDING MODELS")
    lf_input_size: int = train_ds.get_lf_input_size()
    hf_input_size: int = train_ds.get_hf_input_size()
    output_size: int = train_ds.get_num_classes()

    # YOLO isn't trained fresh like the other model types - the checkpoint itself
    # is what gets built (see helpers.build_model's "yolo" case), so its weights
    # path has to be resolved before building rather than loaded via
    # load_state_dict afterward. ClassifierSettings' validator already guarantees
    # trained_lf_model/trained_hf_model is set whenever lf_model/hf_model is "yolo".
    lf_weights_path: str | None = os.path.abspath(config.classifier_training.trained_lf_model) \
        if lf_model_name == "yolo" else None
    hf_weights_path: str | None = os.path.abspath(config.classifier_training.trained_hf_model) \
        if hf_model_name == "yolo" else None

    lf_model: nn.Module = build_model(
        model_name=lf_model_name,
        input_size=lf_input_size,
        output_size=output_size,
        latent_size=latent_size,
        weights_path=lf_weights_path,
        device=DEVICE
    )

    # HF model's input size depends on hf_input_mode: "concat" feeds it the LF+HF
    # channels concatenated (see classifier_one_run's "hf" fidelity and the
    # SelectiveNet/SAT cascades), so its input size is the sum; "hf_only" feeds it
    # just the HF channels, matching checkpoints pretrained on HF-only features.
    # (Irrelevant for "yolo" - build_model ignores input_size for that case.)
    hf_model_input_size: int = lf_input_size + hf_input_size if hf_input_mode == "concat" else hf_input_size

    hf_model: nn.Module = build_model(
        model_name=hf_model_name,
        input_size=hf_model_input_size,
        output_size=output_size,
        latent_size=latent_size,
        weights_path=hf_weights_path,
        device=DEVICE
    )

    lf_param_count = count_parameters(lf_model, trainable_only=lf_model_name != "yolo")
    hf_param_count = count_parameters(hf_model, trainable_only=hf_model_name != "yolo")
    logger.info(f"LF model parameters: {lf_param_count:,}")
    logger.info(f"HF model parameters: {hf_param_count:,}")
    run_summary["cost_realism"]["lf"] = {"num_parameters": lf_param_count}
    run_summary["cost_realism"]["hf"] = {"num_parameters": hf_param_count}

    # ================================================================
    # LF & HF MODEL TRAINING
    # ================================================================
    if not train_body:
        logger.info(f"Parameter 'train_body' was set to False, saving the body outputs for faster running")
        
        train_body_folder: str = os.path.join(body_folder, "train")
        save_body(
            lf_model=lf_model,
            hf_model=hf_model,
            dataloader=train_dl,
            save_folder=train_body_folder,
            device=DEVICE,
            hf_input_mode=hf_input_mode
        )

        test_body_folder: str = os.path.join(body_folder, "test")
        save_body(
            lf_model=lf_model,
            hf_model=hf_model,
            dataloader=test_dl,
            save_folder=test_body_folder,
            device=DEVICE,
            hf_input_mode=hf_input_mode
        )

        val_body_folder: str = os.path.join(body_folder, "val")
        save_body(
            lf_model=lf_model,
            hf_model=hf_model,
            dataloader=val_dl,
            save_folder=val_body_folder,
            device=DEVICE,
            hf_input_mode=hf_input_mode
        )
        
        folder_size: float = get_folder_size(body_folder)/1e6
        logger.info(f"Total size of {body_folder}: {folder_size:,} MB")

        train_ds = BodyDataset(
            folder_path=train_body_folder
        )

        test_ds = BodyDataset(
            folder_path=test_body_folder
        )

        val_ds = BodyDataset(
            folder_path=val_body_folder
        )

        train_dl: DataLoader = DataLoader(
            train_ds,
            batch_size=config.classifier_training.batch_size,
            shuffle=True
        )

        test_dl: DataLoader = DataLoader(
            test_ds,
            batch_size=config.classifier_training.batch_size,
        )

        val_dl: DataLoader = DataLoader(
            val_ds,
            batch_size=config.classifier_training.batch_size,
        )

    logger.info(f"TRAINING HF AND LF MODELS")
    lf_model: nn.Module = lf_model.to(DEVICE)
    hf_model: nn.Module = hf_model.to(DEVICE)

    lf_optimizer = torch.optim.Adam(lf_model.parameters(), lr=config.classifier_training.lr, weight_decay=1e-5)
    hf_optimizer = torch.optim.Adam(hf_model.parameters(), lr=config.classifier_training.lr, weight_decay=1e-5)

    lf_state_dict: Dict | None  = None
    hf_state_dict: Dict | None  = None

    lf_best_acc: float = 0.0
    hf_best_acc: float = 0.0

    lf_model_file = os.path.join(model_folder, "lf_model.pt")
    hf_model_file = os.path.join(model_folder, "hf_model.pt")

    # if the user provides models, use theirs
    train_lf_model = True
    trained_lf_path = config.classifier_training.trained_lf_model

    if lf_model_name == "yolo":
        # Already loaded straight from the checkpoint via build_model above (not
        # via load_state_dict - the yolov5 checkpoint format doesn't match a plain
        # state_dict), so just persist its weights so the reload later in this
        # function behaves the same as every other model case.
        train_lf_model = False
        torch.save(lf_model.state_dict(), lf_model_file)
        logger.info("LF model is YOLO; using pretrained checkpoint, skipping LF train")
    elif trained_lf_path is not None:
        try:
            trained_lf_path = os.path.abspath(trained_lf_path)
            lf_model.load_state_dict(
                torch.load(
                    trained_lf_path,
                    weights_only=True
            ))
            train_lf_model = False
            torch.save(lf_model.state_dict(), lf_model_file)
            logger.info("Pretrained LF model found; skipping LF train")
        except:
            logger.info("Pretrained LF model failed to load; training LF model")
    else:
        logger.info("Pretrained LF model not provided; training LF model")

    train_hf_model = True
    trained_hf_path = config.classifier_training.trained_hf_model

    if hf_model_name == "yolo":
        train_hf_model = False
        torch.save(hf_model.state_dict(), hf_model_file)
        logger.info("HF model is YOLO; using pretrained checkpoint, skipping HF train")
    elif trained_hf_path is not None:
        try:
            trained_hf_path = os.path.abspath(trained_hf_path)
            hf_model.load_state_dict(
                torch.load(
                    trained_hf_path,
                    weights_only=True
            ))
            train_hf_model = False
            torch.save(hf_model.state_dict(), hf_model_file)
            logger.info("Pretrained HF model found; skipping HF train")
        except RuntimeError as e:
            if "size mismatch" in str(e) or "shape" in str(e).lower():
                logger.warning(
                    f"Pretrained HF model failed to load, likely due to a hf_input_mode mismatch "
                    f"(current setting: '{hf_input_mode}', expected input size: {hf_model_input_size}). "
                    f"Check whether this checkpoint was trained with a different hf_input_mode."
                )
            else:
                logger.info("Pretrained HF model failed to load; training HF model")
            train_hf_model = True
        except Exception:
            logger.info("Pretrained HF model failed to load; training HF model")
            train_hf_model = True
    else:
        logger.info("Pretrained HF model not provided; training HF model")

    num_classifier_epochs: int = config.classifier_training.epochs
    # dont retrain if user provided models
    if not train_lf_model and not train_hf_model:
        num_classifier_epochs = 0
        logger.info("Both models provided; skipping classifier training")

    reset_peak_memory(DEVICE)
    lf_train_time_sec: float = 0.0
    hf_train_time_sec: float = 0.0

    epoch_metric_fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_sec"]
    lf_epoch_logger = EpochMetricsLogger(os.path.join(file_folder, "lf_epoch_metrics.csv"), epoch_metric_fields) \
        if train_lf_model else None
    hf_epoch_logger = EpochMetricsLogger(os.path.join(file_folder, "hf_epoch_metrics.csv"), epoch_metric_fields) \
        if train_hf_model else None

    for epoch in range(num_classifier_epochs):
        logger.info(f"Epoch {epoch+1} Summary:")
        if train_lf_model:
            lf_epoch_start = time.perf_counter()

            lf_train_loss, lf_train_acc = classifier_one_run(
                model=lf_model,
                dataloader=train_dl,
                criterion=nn.CrossEntropyLoss(),
                fidelity="lf",
                train_body=train_body,
                optimizer=lf_optimizer
            )

            lf_val_loss, lf_val_acc = classifier_one_run(
                model=lf_model,
                dataloader=val_dl,
                criterion=nn.CrossEntropyLoss(),
                fidelity="lf",
                train_body=train_body,
            )

            lf_epoch_time_sec = time.perf_counter() - lf_epoch_start
            lf_train_time_sec += lf_epoch_time_sec

            if lf_val_acc > lf_best_acc:
                lf_best_acc = lf_val_acc
                lf_state_dict = lf_model.state_dict()
                torch.save(lf_state_dict, lf_model_file)

            lf_epoch_logger.log_epoch(
                epoch=epoch+1, train_loss=lf_train_loss, train_acc=lf_train_acc,
                val_loss=lf_val_loss, val_acc=lf_val_acc, epoch_time_sec=lf_epoch_time_sec
            )

            logger.info(f"\tLF Train  Loss: {lf_train_loss:.4f} | Accuracy: {100 * lf_train_acc:.2f}%")
            logger.info(f"\tLF Val    Loss: {lf_val_loss:.4f}   | Accuracy: {100 * lf_val_acc:.2f}%")

        if train_hf_model:
            hf_epoch_start = time.perf_counter()

            hf_train_loss, hf_train_acc = classifier_one_run(
                model=hf_model,
                dataloader=train_dl,
                criterion=nn.CrossEntropyLoss(),
                fidelity="hf",
                train_body=train_body,
                optimizer=hf_optimizer,
                hf_input_mode=hf_input_mode
            )

            hf_val_loss, hf_val_acc = classifier_one_run(
                model=hf_model,
                dataloader=val_dl,
                criterion=nn.CrossEntropyLoss(),
                fidelity="hf",
                train_body=train_body,
                hf_input_mode=hf_input_mode
            )

            hf_epoch_time_sec = time.perf_counter() - hf_epoch_start
            hf_train_time_sec += hf_epoch_time_sec

            if hf_val_acc > hf_best_acc:
                hf_best_acc = hf_val_acc
                hf_state_dict = hf_model.state_dict()
                torch.save(hf_state_dict, hf_model_file)

            hf_epoch_logger.log_epoch(
                epoch=epoch+1, train_loss=hf_train_loss, train_acc=hf_train_acc,
                val_loss=hf_val_loss, val_acc=hf_val_acc, epoch_time_sec=hf_epoch_time_sec
            )

            logger.info(f"\tHF Train  Loss: {hf_train_loss:.4f} | Accuracy: {100 * hf_train_acc:.2f}%")
            logger.info(f"\tHF Val    Loss: {hf_val_loss:.4f}   | Accuracy: {100 * hf_val_acc:.2f}%")

    classifier_peak_mem_mb = get_peak_memory_mb(DEVICE)
    logger.info(f"LF train time: {format_duration(lf_train_time_sec)}")
    logger.info(f"HF train time: {format_duration(hf_train_time_sec)}")
    logger.info(f"Classifier phase peak GPU memory: {classifier_peak_mem_mb:.1f} MB")

    run_summary["timing"]["lf_train_sec"] = lf_train_time_sec
    run_summary["timing"]["hf_train_sec"] = hf_train_time_sec
    run_summary["peak_gpu_memory_mb"]["classifier_training_phase"] = classifier_peak_mem_mb

    lf_model.load_state_dict(torch.load(lf_model_file, weights_only=True))
    hf_model.load_state_dict(torch.load(hf_model_file, weights_only=True))

    lf_test_loss, lf_test_acc = classifier_one_run(
        model=lf_model,
        dataloader=test_dl,
        criterion=nn.CrossEntropyLoss(),
        fidelity="lf",
        train_body=train_body,
    )

    hf_test_loss, hf_test_acc = classifier_one_run(
        model=hf_model,
        dataloader=test_dl,
        criterion=nn.CrossEntropyLoss(),
        fidelity="hf",
        train_body=train_body,
        hf_input_mode=hf_input_mode
    )

    # Also reported against val here (not just test) so a val/test gap is visible
    # up front - for datasets like LLVIP, where val is carved out of the train
    # pool rather than being a truly independent split, val can look far rosier
    # than test and that's worth catching immediately, not discovering later via
    # the gate search quietly optimizing against an inflated signal.
    lf_val_loss, lf_val_acc = classifier_one_run(
        model=lf_model,
        dataloader=val_dl,
        criterion=nn.CrossEntropyLoss(),
        fidelity="lf",
        train_body=train_body,
    )

    hf_val_loss, hf_val_acc = classifier_one_run(
        model=hf_model,
        dataloader=val_dl,
        criterion=nn.CrossEntropyLoss(),
        fidelity="hf",
        train_body=train_body,
        hf_input_mode=hf_input_mode
    )

    logger.info("Final Test Summary")
    logger.info(f"\tLF Test   Loss: {lf_test_loss:.4f}  | Accuracy: {100 * lf_test_acc:.2f}%")
    logger.info(f"\tHF Test   Loss: {hf_test_loss:.4f}  | Accuracy: {100 * hf_test_acc:.2f}%")
    logger.info("Final Val Summary")
    logger.info(f"\tLF Val    Loss: {lf_val_loss:.4f}  | Accuracy: {100 * lf_val_acc:.2f}%")
    logger.info(f"\tHF Val    Loss: {hf_val_loss:.4f}  | Accuracy: {100 * hf_val_acc:.2f}%")

    # snapshot the cascade (raw/body) datasets & dataloaders before train_dl/test_dl/val_dl
    # get overwritten below with FE_Dataset loaders (precomputed latents) for FE/SR — the
    # SelectiveNet/SAT baselines need real forward passes through lf_model/hf_model.
    cascade_train_ds = train_ds
    cascade_test_ds = test_ds
    cascade_val_ds = val_ds
    cascade_train_dl = train_dl
    cascade_test_dl = test_dl
    cascade_val_dl = val_dl

    def lf_forward(batch):
        data = batch[0].to(DEVICE, torch.float)
        return (lf_model(data) if train_body else lf_model.head(data))["output"]

    def hf_forward(batch):
        data = assemble_hf_input(batch[0], batch[1], hf_input_mode, hf_model=hf_model).to(DEVICE, torch.float)
        return (hf_model(data) if train_body else hf_model.head(data))["output"]

    lf_inference_latency_ms = measure_inference_latency(lf_forward, cascade_test_dl, DEVICE)
    hf_inference_latency_ms = measure_inference_latency(hf_forward, cascade_test_dl, DEVICE)
    logger.info(f"LF inference latency: {lf_inference_latency_ms:.3f} ms/sample")
    logger.info(f"HF inference latency: {hf_inference_latency_ms:.3f} ms/sample")

    run_summary["cost_realism"]["lf"]["inference_latency_ms_per_sample"] = lf_inference_latency_ms
    run_summary["cost_realism"]["hf"]["inference_latency_ms_per_sample"] = hf_inference_latency_ms
    run_summary["final_accuracy"]["lf_test_acc"] = lf_test_acc
    run_summary["final_accuracy"]["hf_test_acc"] = hf_test_acc
    run_summary["final_accuracy"]["lf_val_acc"] = lf_val_acc
    run_summary["final_accuracy"]["hf_val_acc"] = hf_val_acc

    # ================================================================
    # FE MODEL TRAINING
    # save LF/HF latents, then AdaptiveGridSearch over usage_values
    # ================================================================
    logger.info("SAVING LATENT REPRESENTATIONS")

    # UNet exposes both its encoder bottleneck ("body", e.g. 1024x14x14) and
    # its decoder's near-final feature map at full input resolution
    # ("latent", 64x224x224 - kept spatial for a future per-pixel/region-
    # level gate, see UNet.forward's comment in models.py). Saving "latent"
    # per sample across crop's ~90k-sample splits needs ~1.6TB of disk, so
    # this currently uses the much smaller bottleneck instead. Switch back to
    # "latent" here (and see save_latent's docstring) to bring that future
    # work back.
    gate_latent_key = "body" if dataset_name == "crop" else "latent"

    # save_latent saves each sample's "idx" alongside its latent/loss/correctness
    # (see save_latent's docstring) so a routed sample can be traced back to the
    # original dataset item later - IndexedDataset is what actually supplies that
    # idx, wrapping the raw cascade datasets rather than mutating cascade_train_dl
    # etc. themselves (those are reused unwrapped elsewhere, e.g. SelectiveNet's
    # cascade below). shuffling doesn't matter here - every sample lands in its
    # own idx-named file regardless of batch order.
    latent_train_dl = DataLoader(IndexedDataset(cascade_train_ds), batch_size=config.classifier_training.batch_size)
    latent_test_dl = DataLoader(IndexedDataset(cascade_test_ds), batch_size=config.classifier_training.batch_size)
    latent_val_dl = DataLoader(IndexedDataset(cascade_val_ds), batch_size=config.classifier_training.batch_size)

    train_latent_folder: str = os.path.join(latent_folder, "train")
    save_latent(
        lf_model=lf_model,
        hf_model=hf_model,
        dataloader=latent_train_dl,
        save_folder=train_latent_folder,
        train_body=train_body,
        device=DEVICE,
        hf_input_mode=hf_input_mode,
        latent_key=gate_latent_key
    )

    test_latent_folder: str = os.path.join(latent_folder, "test")
    save_latent(
        lf_model=lf_model,
        hf_model=hf_model,
        dataloader=latent_test_dl,
        save_folder=test_latent_folder,
        train_body=train_body,
        device=DEVICE,
        hf_input_mode=hf_input_mode,
        latent_key=gate_latent_key
    )

    val_latent_folder: str = os.path.join(latent_folder, "val")
    save_latent(
        lf_model=lf_model,
        hf_model=hf_model,
        dataloader=latent_val_dl,
        save_folder=val_latent_folder,
        train_body=train_body,
        device=DEVICE,
        hf_input_mode=hf_input_mode,
        latent_key=gate_latent_key
    )
    folder_size: float = get_folder_size(latent_folder)/1e6
    logger.info(f"Total size of {latent_folder}: {folder_size:,} MB")

    train_ds: FE_Dataset = FE_Dataset(
        folder_path=train_latent_folder
    )

    test_ds: FE_Dataset = FE_Dataset(
        folder_path=test_latent_folder
    )

    val_ds: FE_Dataset = FE_Dataset(
        folder_path=val_latent_folder
    )

    # FE_Dataset falls back to per-sample torch.load() from disk whenever a split
    # doesn't fit in RAM (see FE_Dataset.__init__) - for a dataset the size of
    # crop's train split, that's ~98% of an epoch's wall time (measured: ~190s of
    # disk I/O vs. ~2.6s of actual gate-model compute per epoch), all serialized
    # in the main process under the default num_workers=0. Parallelizing those
    # reads across worker processes is the fix - this dataloader gets re-iterated
    # every gate epoch, across every bisection trial, across every rerun, so it's
    # worth paying for. persistent_workers avoids respawning that whole worker
    # pool at the start of each of those re-iterations; pin_memory speeds up the
    # host->GPU transfer. Harmless (if less impactful) for test/val too, which
    # usually fit in RAM already.
    fe_dataloader_workers = 8

    train_dl: DataLoader = DataLoader(
        train_ds,
        batch_size=config.fe_training.batch_size,
        shuffle=True,
        num_workers=fe_dataloader_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    test_dl: DataLoader = DataLoader(
        test_ds,
        batch_size=config.fe_training.batch_size,
        num_workers=fe_dataloader_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    val_dl: DataLoader = DataLoader(
        val_ds,
        batch_size=config.fe_training.batch_size,
        num_workers=fe_dataloader_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    # cnn_head is for a spatial (C, H, W) lf_latent - crop's UNet bottleneck
    # (see gate_latent_key above) and YOLO's raw (unpooled) backbone feature
    # map - vs. mlp for a flat (D,) latent (resnet/vit/mlp). Note LatentCNNHead
    # itself still global-average-pools before its routing decision either
    # way, so this is about which features feed that decision, not (yet)
    # about making the decision itself per-pixel/region.
    is_spatial_latent = (dataset_name == "crop" or lf_model_name == "yolo")

    # Either way, size the gate model off the *actual* saved latent, not
    # config.latent_size - that assumption only holds for resnet/vit/mlp, which
    # explicitly project their backbone output down to latent_size via a
    # latent_rep layer. UNet's decoder output and YOLO's raw backbone feature
    # are each a fixed channel count of their own, independent of config.latent_size.
    lf_latent_channels: int = train_ds[0][0].shape[0]

    if is_spatial_latent:
        fe_model: nn.Module = build_model(
            model_name="cnn_head",
            input_size=lf_latent_channels,
            output_size=2,
            latent_size=None
        )
    else:
        fe_model: nn.Module = build_model(
            model_name="mlp",
            input_size=lf_latent_channels,
            output_size=2,
            latent_size=256
        )

    fe_param_count = count_parameters(fe_model)
    logger.info(f"Gate/FE model parameters: {fe_param_count:,}")
    run_summary["cost_realism"]["gate"] = {"num_parameters": fe_param_count}

    logger.info("RUNNING DEFAULT FE")

    # Measure how much routing signal actually exists before deciding whether
    # to apply the imbalance mitigations below - originally these were gated on
    # is_yolo_gate (LLVIP/YOLO's LF/HF disagree on only ~5% of samples), but
    # the same collapse - an unweighted, un-clipped gate hits softmax
    # saturation within a handful of epochs and gets stuck always picking one
    # fidelity, at every cost value - shows up on any dataset whose LF/HF
    # models are this correlated, toy_2d included (~7% here, just from how
    # close its two fidelity's accuracies happen to land). Measuring it
    # directly generalizes the mitigation instead of hardcoding it to one
    # dataset name.
    hf_needed_count = 0
    total_count = 0
    for _, _, _, lf_correct, hf_correct, _ in train_dl:
        hf_needed_count += (hf_correct & ~lf_correct).sum().item()
        total_count += lf_correct.numel()
    hf_needed_rate = hf_needed_count / total_count
    logger.info(f"HF-needed rate on train set: {hf_needed_rate:.2%} ({hf_needed_count:,}/{total_count:,})")

    # Below this, an unweighted/un-clipped gate reliably collapses within a
    # handful of epochs regardless of cost (observed on both LLVIP/YOLO's ~5%
    # rate and toy_2d's ~7%). class_weighted per-batch loss reweighting used to
    # be the other half of this mitigation, but was removed - at low
    # hf_needed_rate a 128-sample batch sees on the order of one disagreement
    # sample, so its reweight factor swings wildly batch to batch (0x some
    # batches, 100x+ on others), which turned out to destabilize gate training
    # more than the imbalance itself did.
    imbalanced_gate = hf_needed_rate < 0.15

    adaptive_search: AdaptiveGridSearch = AdaptiveGridSearch(
        fe_model=fe_model,
        device=DEVICE,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        model_folder=model_folder,
        # A lower LR + gradient clipping guards against the gate's routing
        # logits blowing up into softmax saturation within the first epoch or
        # two (observed on this task's thin, imbalanced routing signal).
        gate_epochs=config.fe_training.epochs,
        gate_lr=5e-5 if imbalanced_gate else 3e-4,
        gate_grad_clip_norm=1.0 if imbalanced_gate else None,
        seed=seed
    )

    reset_peak_memory(DEVICE)
    gate_search_start = time.perf_counter()

    usage_values: List[float] = [i/10 for i in range(1, 11)]
    fe_reruns: int = config.fe_training.reruns
    logger.info(f"FE reruns: {fe_reruns}")

    fe_usage_runs, fe_acc_runs = adaptive_search.run_reruns(
        usage_values=usage_values,
        n_reruns=fe_reruns
    )

    fe_usage_vals = fe_usage_runs.mean(axis=0)
    fe_acc_vals = fe_acc_runs.mean(axis=0)
    fe_usage_std = fe_usage_runs.std(axis=0)
    fe_acc_std = fe_acc_runs.std(axis=0)

    gate_search_time_sec = time.perf_counter() - gate_search_start
    gate_peak_mem_mb = get_peak_memory_mb(DEVICE)
    logger.info(f"Gate search total time: {format_duration(gate_search_time_sec)}")
    logger.info(f"Gate search phase peak GPU memory: {gate_peak_mem_mb:.1f} MB")

    run_summary["timing"]["gate_search_total_sec"] = gate_search_time_sec
    run_summary["peak_gpu_memory_mb"]["gate_search_phase"] = gate_peak_mem_mb

    adaptive_search.save_evaluated_points(os.path.join(file_folder, "gate_search_log.npz"))
    adaptive_search.save_epoch_log(os.path.join(file_folder, "gate_epoch_metrics.csv"))
    adaptive_search.save_search_diagnostics(os.path.join(file_folder, "gate_search_diagnostics.csv"))
    adaptive_search.save_routing_snapshots(os.path.join(file_folder, "gate_routing_snapshots.npz"))

    fe_data_path: str = os.path.join(file_folder, "fe_results.npz")
    np.savez(
        fe_data_path,
        usage=fe_usage_vals,
        acc=fe_acc_vals,
        usage_std=fe_usage_std,
        acc_std=fe_acc_std,
        usage_runs=fe_usage_runs,
        acc_runs=fe_acc_runs
    )
    run_summary["result_files"].append("fe_results.npz")

    # ================================================================
    # FE EVALUATION PLOTS
    # gate (c_h) sensitivity — only depends on the FE search above, so this
    # runs even when run_comparisons is False
    # ================================================================
    logger.info("PLOTTING FE GATE SENSITIVITY")

    gate_r, gate_usage, gate_acc = load_gate_log(file_folder)
    gate_r_sorted, gate_usage_sorted, gate_acc_sorted = dedupe_and_sort(gate_r, gate_usage, gate_acc)

    # Exact Bayes-optimal usage(c_h) curve, computed straight from the frozen
    # LF/HF models' own test-set losses (test_latent_folder, already on disk
    # from save_latent above) - no training/noise involved, unlike every other
    # series on these plots. Lets the gate's own noisy estimates be judged
    # against ground truth instead of only against each other.
    try:
        gate_benefit = load_test_benefit(test_latent_folder)
    except FileNotFoundError as e:
        logger.warning(f"No oracle overlay available for gate sensitivity plots: {e}")
        gate_benefit = None

    gate_derivative = compute_usage_derivative(gate_r_sorted, gate_usage_sorted)
    plot_derivative(
        gate_r_sorted, gate_derivative,
        os.path.join(image_folder, "gate_sensitivity_derivative.png"),
        benefit=gate_benefit
    )

    plot_full_curve(
        gate_r_sorted, gate_usage_sorted,
        os.path.join(image_folder, "gate_sensitivity_full.png"),
        benefit=gate_benefit
    )

    gate_zoom_range = find_zoom_range(gate_r_sorted, gate_usage_sorted, low=0.7, high=0.9)
    plot_zoom(
        gate_r_sorted, gate_usage_sorted, gate_zoom_range,
        os.path.join(image_folder, "gate_sensitivity_zoom.png"),
        benefit=gate_benefit
    )

    # Raw (undeduped) points, deliberately - the scatter is meant to show the
    # actual per-run training noise, not the averaged-per-c_h view above.
    plot_isotonic_smoothing(
        gate_r, gate_usage,
        os.path.join(image_folder, "gate_sensitivity_isotonic.png"),
        benefit=gate_benefit
    )

    logger.info("PLOTTING GATE ROUTING (projection + confusion tables)")
    routing_snapshots = load_routing_snapshots(file_folder)
    for rerun_idx in range(fe_reruns):
        plot_routing_projection_grid(
            routing_snapshots, rerun_idx,
            os.path.join(image_folder, f"gate_routing_projection_rerun{rerun_idx}.png")
        )
        plot_routing_table_grid(
            routing_snapshots, rerun_idx,
            os.path.join(image_folder, f"gate_routing_table_rerun{rerun_idx}.png")
        )

    gate_max_usage, gate_r_at_max = report_max_usage(gate_r_sorted, gate_usage_sorted)
    logger.info(
        f"Max achievable usage under the current c_h sampling grid: {gate_max_usage * 100:.2f}% "
        f"at c_h={gate_r_at_max:.4f}"
    )

    # ================================================================
    # FE + SOFTMAX RESPONSE
    # "just the FE model" (above) uses each r's gate with its own hard argmax
    # routing decision. This reuses the SAME already-trained fe_model-{r}.pt
    # checkpoints, but sweeps the gate's own continuous confidence over a
    # threshold grid instead - see AdaptiveGridSearch.run_fe_sr_grid. Runs
    # regardless of run_comparisons - it's a natural extension of the FE
    # results above, not one of the classification-shaped SelectiveNet/SAT/SR
    # baselines below (which YOLO can't run at all).
    # ================================================================
    logger.info("RUNNING FE + SOFTMAX RESPONSE")

    fe_sr_threshold_grid: List[float] = list(np.linspace(0.0, 1.0, 21))
    fe_sr = adaptive_search.run_fe_sr_grid(
        usage_values=usage_values,
        threshold_grid=fe_sr_threshold_grid
    )

    fe_sr_data_path: str = os.path.join(file_folder, "fe_sr_results.npz")
    np.savez(
        fe_sr_data_path,
        usage=fe_sr["usage"],
        acc=fe_sr["acc"],
        target_usage=fe_sr["target_usage"],
        chosen_r=fe_sr["chosen_r"],
        chosen_threshold=fe_sr["chosen_threshold"]
    )
    run_summary["result_files"].append("fe_sr_results.npz")

    fe_sr_grid_path: str = os.path.join(file_folder, "fe_sr_grid.npz")
    np.savez(
        fe_sr_grid_path,
        r_grid=fe_sr["r_grid"],
        threshold_grid=fe_sr["threshold_grid"],
        val_usage_grid=fe_sr["val_usage_grid"],
        val_acc_grid=fe_sr["val_acc_grid"],
        test_usage_grid=fe_sr["test_usage_grid"],
        test_acc_grid=fe_sr["test_acc_grid"]
    )
    run_summary["result_files"].append("fe_sr_grid.npz")

    # ends the script if you don't want comparisons
    if not run_comparisons:
        logger.info("PLOTTING PARETO CURVES")

        # lf_correct/hf_correct only depend on the LF/HF models' own
        # predictions on the fixed test set, not on which gate model produced
        # a given snapshot — so any single snapshot's arrays (here, the
        # first) give the ground truth.
        oracle_acc = compute_oracle_curve(
            routing_snapshots["lf_correct"][0], routing_snapshots["hf_correct"][0], usage_values
        )
        oracle = (np.array(usage_values) * 100, oracle_acc)

        pareto_curves = load_pareto_curves(file_folder)
        render_pareto_plot(
            pareto_curves, dataset_label=dataset_name,
            save_path=os.path.join(image_folder, "pareto.png"),
            oracle=oracle
        )

        run_summary["timing"]["total_experiment_sec"] = time.perf_counter() - experiment_start_time
        with open(os.path.join(folder_name, "summary.json"), "w") as f:
            json.dump(run_summary, f, indent=2)
        logger.info("EXPERIMENT FINISHED")
        return

    # ================================================================
    # SOFTMAX RESPONSE (SR) BASELINE
    # ================================================================
    logger.info("RUNNING DEFAULT SOFTMAX RESPONSE")

    softmax_response: SoftmaxResponseMethod = SoftmaxResponseMethod(
        lf_model=lf_model,
        val_dl=cascade_val_dl,
        test_dl=cascade_test_dl,
        device=DEVICE,
        train_body=train_body,
    )

    sr_acc_vals, sr_usage_vals = softmax_response.get_softmax_thresholds(
        usage_list=usage_values
    )

    thresholds = np.array(list(sr_acc_vals.keys()))
    sr_acc_vals = np.array(list(sr_acc_vals.values()))
    sr_usage_vals = np.array(list(sr_usage_vals.values()))

    sr_data_path: str = os.path.join(file_folder, "sr_results.npz")
    np.savez(
        sr_data_path,
        usage=sr_usage_vals,
        acc=sr_acc_vals,
        thresholds=thresholds
    )
    run_summary["result_files"].append("sr_results.npz")

    # ================================================================
    # SELECTIVENET MODEL TRAINING
    # one model per c in usage_values, then a (c, threshold) grid search
    # ================================================================
    num_baseline_epochs: int = config.classifier_training.epochs
    threshold_grid: List[float] = list(np.linspace(0.0, 1.0, 21))
    c_grid: List[float] = usage_values

    logger.info("RUNNING SELECTIVENET (c GRID)")

    n_c = len(c_grid)
    n_t = len(threshold_grid)

    # rows = c (train-time coverage penalty), cols = threshold (inference-time cutoff)
    selnet_val_usage_grid = np.zeros((n_c, n_t))
    selnet_val_acc_grid = np.zeros((n_c, n_t))
    selnet_test_usage_grid = np.zeros((n_c, n_t))
    selnet_test_acc_grid = np.zeros((n_c, n_t))

    selnet_param_count = None
    reset_peak_memory(DEVICE)
    selnet_train_start = time.perf_counter()

    for c_idx, c_val in enumerate(c_grid):
        logger.info(f"SelectiveNet c={c_val:.2f}")

        selnet_model: nn.Module = build_model(
            model_name=lf_model_name,
            input_size=lf_input_size,
            output_size=output_size,
            latent_size=latent_size
        ).to(DEVICE)

        if selnet_param_count is None:
            selnet_param_count = count_parameters(selnet_model)
            logger.info(f"SelectiveNet model parameters: {selnet_param_count:,}")
            run_summary["cost_realism"]["selectivenet"] = {"num_parameters": selnet_param_count}

        selnet_optimizer = torch.optim.Adam(selnet_model.parameters(), lr=config.classifier_training.lr, weight_decay=1e-5)
        selnet_method: SelectiveNetMethod = SelectiveNetMethod(model_folder=model_folder, c=c_val)
        selnet_model_file = os.path.join(model_folder, f"selnet_model_c{c_val:.2f}.pt")
        selnet_best_acc: float = 0.0

        selnet_epoch_logger = EpochMetricsLogger(
            os.path.join(file_folder, f"selnet_epoch_metrics_c{c_val:.2f}.csv"),
            ["epoch", "train_loss", "train_acc", "train_usage", "val_loss", "val_acc", "val_usage", "epoch_time_sec"]
        )

        for epoch in range(40):
            selnet_epoch_start = time.perf_counter()

            selnet_train_metrics = selnet_method.one_run(
                model=selnet_model,
                dataloader=cascade_train_dl,
                hf_model=hf_model,
                threshold=0.5,
                train_body=train_body,
                optimizer=selnet_optimizer,
                hf_input_mode=hf_input_mode
            )

            selnet_val_metrics = selnet_method.one_run(
                model=selnet_model,
                dataloader=cascade_val_dl,
                hf_model=hf_model,
                threshold=0.5,
                train_body=train_body,
                hf_input_mode=hf_input_mode
            )

            selnet_epoch_time_sec = time.perf_counter() - selnet_epoch_start

            if selnet_val_metrics["accuracy"] > selnet_best_acc:
                selnet_best_acc = selnet_val_metrics["accuracy"]
                torch.save(selnet_model.state_dict(), selnet_model_file)

            selnet_epoch_logger.log_epoch(
                epoch=epoch+1,
                train_loss=selnet_train_metrics["loss"], train_acc=selnet_train_metrics["accuracy"],
                train_usage=selnet_train_metrics["usage"],
                val_loss=selnet_val_metrics["loss"], val_acc=selnet_val_metrics["accuracy"],
                val_usage=selnet_val_metrics["usage"], epoch_time_sec=selnet_epoch_time_sec
            )

            logger.info(f"SelectiveNet c={c_val:.2f} Epoch {epoch+1} Summary:")
            logger.info(f"\tTrain Loss: {selnet_train_metrics['loss']:.4f} | Accuracy: {100 * selnet_train_metrics['accuracy']:.2f}% | Usage: {100 * selnet_train_metrics['usage']:.2f}%")
            logger.info(f"\tVal   Loss: {selnet_val_metrics['loss']:.4f}   | Accuracy: {100 * selnet_val_metrics['accuracy']:.2f}%   | Usage: {100 * selnet_val_metrics['usage']:.2f}%")

        selnet_model.load_state_dict(torch.load(selnet_model_file, weights_only=True))

        for t_idx, threshold in enumerate(threshold_grid):
            selnet_val_grid_metrics = selnet_method.one_run(
                model=selnet_model,
                dataloader=cascade_val_dl,
                hf_model=hf_model,
                threshold=float(threshold),
                train_body=train_body,
                hf_input_mode=hf_input_mode
            )

            selnet_test_grid_metrics = selnet_method.one_run(
                model=selnet_model,
                dataloader=cascade_test_dl,
                hf_model=hf_model,
                threshold=float(threshold),
                train_body=train_body,
                hf_input_mode=hf_input_mode
            )

            selnet_val_usage_grid[c_idx, t_idx] = selnet_val_grid_metrics["usage"]
            selnet_val_acc_grid[c_idx, t_idx] = selnet_val_grid_metrics["accuracy"]
            selnet_test_usage_grid[c_idx, t_idx] = selnet_test_grid_metrics["usage"]
            selnet_test_acc_grid[c_idx, t_idx] = selnet_test_grid_metrics["accuracy"]

    selnet_train_time_sec = time.perf_counter() - selnet_train_start
    selnet_peak_mem_mb = get_peak_memory_mb(DEVICE)
    logger.info(f"SelectiveNet train time (all c values): {format_duration(selnet_train_time_sec)}")
    logger.info(f"SelectiveNet phase peak GPU memory: {selnet_peak_mem_mb:.1f} MB")

    run_summary["timing"]["selectivenet_train_sec"] = selnet_train_time_sec
    run_summary["peak_gpu_memory_mb"]["selectivenet_phase"] = selnet_peak_mem_mb

    # For each target usage, pick the (c, threshold) cell whose val usage is the
    # largest one still <= target (closest without going over the budget). If no
    # cell meets the budget, fall back to whichever cell's val usage is closest
    # to the target overall, and warn since the budget was violated.
    selnet_usage_vals: List[float] = []
    selnet_acc_vals: List[float] = []
    selnet_chosen_c: List[float] = []
    selnet_chosen_threshold: List[float] = []

    flat_val_usage = selnet_val_usage_grid.ravel()

    for target_usage in usage_values:
        under_budget = np.where(flat_val_usage <= target_usage)[0]

        if under_budget.size > 0:
            best_flat_idx = under_budget[np.argmax(flat_val_usage[under_budget])]
        else:
            logger.warning(
                f"No (c, threshold) combo has val usage <= {target_usage}; "
                f"falling back to the closest val usage overall"
            )
            best_flat_idx = np.argmin(np.abs(flat_val_usage - target_usage))

        c_idx, t_idx = np.unravel_index(best_flat_idx, selnet_val_usage_grid.shape)

        selnet_usage_vals.append(100 * selnet_test_usage_grid[c_idx, t_idx])
        selnet_acc_vals.append(100 * selnet_test_acc_grid[c_idx, t_idx])
        selnet_chosen_c.append(c_grid[c_idx])
        selnet_chosen_threshold.append(threshold_grid[t_idx])

    selectivenet_data_path: str = os.path.join(file_folder, "selectivenet_results.npz")
    np.savez(
        selectivenet_data_path,
        usage=np.array(selnet_usage_vals),
        acc=np.array(selnet_acc_vals),
        target_usage=np.array(usage_values),
        chosen_c=np.array(selnet_chosen_c),
        chosen_threshold=np.array(selnet_chosen_threshold)
    )
    run_summary["result_files"].append("selectivenet_results.npz")

    selectivenet_grid_path: str = os.path.join(file_folder, "selectivenet_grid.npz")
    np.savez(
        selectivenet_grid_path,
        c_grid=np.array(c_grid),
        threshold_grid=np.array(threshold_grid),
        val_usage_grid=selnet_val_usage_grid,
        val_acc_grid=selnet_val_acc_grid,
        test_usage_grid=selnet_test_usage_grid,
        test_acc_grid=selnet_test_acc_grid
    )
    run_summary["result_files"].append("selectivenet_grid.npz")

    # ================================================================
    # SAT MODEL TRAINING
    # a single model, then a threshold grid search (mirrors SelectiveNet's
    # selection logic but with no c dimension)
    # ================================================================
    logger.info("RUNNING DEFAULT SELF-ADAPTIVE TRAINING (SAT)")
    sat_model: nn.Module = build_model(
        model_name=lf_model_name,
        input_size=lf_input_size,
        output_size=output_size + 1,
        latent_size=latent_size
    ).to(DEVICE)

    sat_param_count = count_parameters(sat_model)
    logger.info(f"SAT model parameters: {sat_param_count:,}")
    run_summary["cost_realism"]["sat"] = {"num_parameters": sat_param_count}

    sat_optimizer = torch.optim.Adam(sat_model.parameters(), lr=config.classifier_training.lr, weight_decay=1e-5)
    sat_method: SelfAdaptiveTrainingMethod = SelfAdaptiveTrainingMethod(
        num_train_samples=len(cascade_train_ds),
        num_classes=output_size,
        alpha=0.99,
        warmup_epochs=0,
        is_segmentation=(dataset_name == "crop"),
        device=DEVICE
    )

    sat_train_dl: DataLoader = DataLoader(
        IndexedDataset(cascade_train_ds),
        batch_size=config.classifier_training.batch_size,
        shuffle=True
    )

    sat_val_dl: DataLoader = DataLoader(
        IndexedDataset(cascade_val_ds),
        batch_size=config.classifier_training.batch_size,
    )

    sat_test_dl: DataLoader = DataLoader(
        IndexedDataset(cascade_test_ds),
        batch_size=config.classifier_training.batch_size,
    )

    sat_method.initialize_targets(sat_train_dl)

    sat_model_file = os.path.join(model_folder, "sat_model.pt")
    sat_best_acc: float = 0.0

    reset_peak_memory(DEVICE)
    sat_train_start = time.perf_counter()
    sat_epoch_logger = EpochMetricsLogger(
        os.path.join(file_folder, "sat_epoch_metrics.csv"),
        ["epoch", "train_loss", "train_acc", "train_usage", "val_loss", "val_acc", "val_usage", "epoch_time_sec"]
    )

    for epoch in range(num_baseline_epochs):
        sat_epoch_start = time.perf_counter()

        sat_train_metrics = sat_method.one_run(
            model=sat_model,
            dataloader=sat_train_dl,
            hf_model=hf_model,
            tau=0.5,
            train_body=train_body,
            optimizer=sat_optimizer,
            epoch=epoch,
            hf_input_mode=hf_input_mode
        )

        sat_val_metrics = sat_method.one_run(
            model=sat_model,
            dataloader=sat_val_dl,
            hf_model=hf_model,
            tau=0.5,
            train_body=train_body,
            epoch=epoch,
            hf_input_mode=hf_input_mode
        )

        sat_epoch_time_sec = time.perf_counter() - sat_epoch_start

        if sat_val_metrics["accuracy"] > sat_best_acc:
            sat_best_acc = sat_val_metrics["accuracy"]
            torch.save(sat_model.state_dict(), sat_model_file)

        sat_epoch_logger.log_epoch(
            epoch=epoch+1,
            train_loss=sat_train_metrics["loss"], train_acc=sat_train_metrics["accuracy"],
            train_usage=sat_train_metrics["usage"],
            val_loss=sat_val_metrics["loss"], val_acc=sat_val_metrics["accuracy"],
            val_usage=sat_val_metrics["usage"], epoch_time_sec=sat_epoch_time_sec
        )

        logger.info(f"SAT Epoch {epoch+1} Summary:")
        logger.info(f"\tTrain Loss: {sat_train_metrics['loss']:.4f} | Accuracy: {100 * sat_train_metrics['accuracy']:.2f}% | Usage: {100 * sat_train_metrics['usage']:.2f}%")
        logger.info(f"\tVal   Loss: {sat_val_metrics['loss']:.4f}   | Accuracy: {100 * sat_val_metrics['accuracy']:.2f}%   | Usage: {100 * sat_val_metrics['usage']:.2f}%")

    sat_train_time_sec = time.perf_counter() - sat_train_start
    sat_peak_mem_mb = get_peak_memory_mb(DEVICE)
    logger.info(f"SAT train time: {format_duration(sat_train_time_sec)}")
    logger.info(f"SAT phase peak GPU memory: {sat_peak_mem_mb:.1f} MB")

    run_summary["timing"]["sat_train_sec"] = sat_train_time_sec
    run_summary["peak_gpu_memory_mb"]["sat_phase"] = sat_peak_mem_mb

    sat_model.load_state_dict(torch.load(sat_model_file, weights_only=True))

    sat_val_usage_grid: List[float] = []
    sat_val_acc_grid: List[float] = []
    sat_test_usage_grid: List[float] = []
    sat_test_acc_grid: List[float] = []

    for tau in threshold_grid:
        sat_val_metrics = sat_method.one_run(
            model=sat_model,
            dataloader=sat_val_dl,
            hf_model=hf_model,
            tau=float(tau),
            train_body=train_body,
            epoch=num_baseline_epochs,
            hf_input_mode=hf_input_mode
        )
        sat_test_metrics = sat_method.one_run(
            model=sat_model,
            dataloader=sat_test_dl,
            hf_model=hf_model,
            tau=float(tau),
            train_body=train_body,
            epoch=num_baseline_epochs,
            hf_input_mode=hf_input_mode
        )

        sat_val_usage_grid.append(sat_val_metrics["usage"])
        sat_val_acc_grid.append(sat_val_metrics["accuracy"])
        sat_test_usage_grid.append(sat_test_metrics["usage"])
        sat_test_acc_grid.append(sat_test_metrics["accuracy"])

    sat_val_usage_grid = np.array(sat_val_usage_grid)
    sat_val_acc_grid = np.array(sat_val_acc_grid)
    sat_test_usage_grid = np.array(sat_test_usage_grid)
    sat_test_acc_grid = np.array(sat_test_acc_grid)

    # For each target usage, pick the threshold whose val usage is the largest
    # one still <= target (closest without going over the budget) — same
    # selection rule already used for SelectiveNet's (c, threshold) grid.
    sat_usage_vals: List[float] = []
    sat_acc_vals: List[float] = []
    sat_chosen_threshold: List[float] = []

    for target_usage in usage_values:
        under_budget = np.where(sat_val_usage_grid <= target_usage)[0]

        if under_budget.size > 0:
            best_idx = under_budget[np.argmax(sat_val_usage_grid[under_budget])]
        else:
            logger.warning(
                f"No threshold has val usage <= {target_usage}; "
                f"falling back to the closest val usage overall"
            )
            best_idx = np.argmin(np.abs(sat_val_usage_grid - target_usage))

        sat_usage_vals.append(100 * sat_test_usage_grid[best_idx])
        sat_acc_vals.append(100 * sat_test_acc_grid[best_idx])
        sat_chosen_threshold.append(threshold_grid[best_idx])

    sat_data_path: str = os.path.join(file_folder, "sat_results.npz")
    np.savez(
        sat_data_path,
        usage=np.array(sat_usage_vals),
        acc=np.array(sat_acc_vals),
        target_usage=np.array(usage_values),
        chosen_threshold=np.array(sat_chosen_threshold)
    )
    run_summary["result_files"].append("sat_results.npz")

    sat_grid_path: str = os.path.join(file_folder, "sat_grid.npz")
    np.savez(
        sat_grid_path,
        threshold_grid=np.array(threshold_grid),
        val_usage_grid=sat_val_usage_grid,
        val_acc_grid=sat_val_acc_grid,
        test_usage_grid=sat_test_usage_grid,
        test_acc_grid=sat_test_acc_grid
    )
    run_summary["result_files"].append("sat_grid.npz")

    # ================================================================
    # FINAL PLOTS
    # Pareto overlay of every baseline that produced a *_results.npz
    # ================================================================
    logger.info("PLOTTING PARETO CURVES")

    # lf_correct/hf_correct only depend on the LF/HF models' own predictions on
    # the fixed test set, not on which gate model produced a given snapshot —
    # so any single snapshot's arrays (here, the first) give the ground truth.
    oracle_acc = compute_oracle_curve(
        routing_snapshots["lf_correct"][0], routing_snapshots["hf_correct"][0], usage_values
    )
    oracle = (np.array(usage_values) * 100, oracle_acc)

    pareto_curves = load_pareto_curves(file_folder)
    render_pareto_plot(
        pareto_curves, dataset_label=dataset_name,
        save_path=os.path.join(image_folder, "pareto.png"),
        oracle=oracle
    )

    run_summary["timing"]["total_experiment_sec"] = time.perf_counter() - experiment_start_time
    logger.info(f"Total experiment time: {format_duration(run_summary['timing']['total_experiment_sec'])}")
    with open(os.path.join(folder_name, "summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    logger.info("EXPERIMENT FINISHED")

if __name__ == "__main__":
    main()
