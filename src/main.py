import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
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
import functools
import gc

from datasets import *
from models import build_unet, CustomResNet18, LatentCNNHead, CustomMLP
from helpers import *
from comparisons import *
from losses import MetaLossFunction
from sample_identity import get_test_sample_identity
from diagnostics import build_routing_dataframe, run_full_diagnostic_suite
from experiment_logging import (
    count_parameters, reset_peak_memory, get_peak_memory_mb,
    measure_inference_latency, format_duration, EpochMetricsLogger
)
from plot_gate_sensitivity import (
    load_gate_log, load_test_benefit, dedupe_and_sort, compute_usage_derivative,
    plot_derivative, find_zoom_range, plot_zoom, plot_full_curve, report_max_usage,
    plot_isotonic_smoothing, plot_usage_vs_ch_std
)
from plot_gate_routing import (
    load_routing_snapshots, compute_dense_oracle_curve, plot_routing_projection_grid, plot_routing_table_grid
)
from plot_pareto_curves import load_pareto_curves, render_pareto_plot, render_pareto_scatter_plot
from plot_selective_comparison import collect_comparison_rows, save_comparison_data, render_all_comparison_plots

from custom_types import ConfigOptions

@click.command()
@click.option("--config_file", default="../config.yml")
@click.option(
    "--comparisons_only", is_flag=True, default=False,
    help="Skip LF/HF training and the FE gate search entirely, reusing an existing "
         "experiment's checkpoints/results folder (same dataset/lf_model/hf_model/seed/"
         "latent_size as config_file) - only runs SR/SelectiveNet/SAT and the comparison "
         "plots/data, appending them to that folder. The experiment must already have "
         "completed at least LF/HF training (models/lf_model.pt and hf_model.pt present)."
)
def main(config_file, comparisons_only):
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

    if comparisons_only:
        # The whole point of this mode is to run the comparison baselines,
        # regardless of what the config file itself says.
        config.run_comparisons = True
        run_comparisons = True

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
        # Compat shim: huggingface_hub 0.36.2 removed huggingface_hub.utils._errors,
        # which the installed yolov5 package still imports from (its download-from-hub
        # fallback path, hit even when loading a purely local checkpoint, since
        # attempt_load tries the hub path first). Only _errors needs shimming - do NOT
        # also shim huggingface_hub.utils._validators, which is a real, fully-functional
        # module (validate_hf_hub_args, HFValidationError) that huggingface_hub's own
        # internal imports still need; replacing it breaks those instead of fixing
        # anything. Every standalone LLVIP script in this repo carries this same shim -
        # main.py never needed it before now because this is its first time loading a
        # YOLO checkpoint directly rather than through one of those scripts.
        import sys
        import types
        from huggingface_hub import errors as _hf_errors
        _errors_shim = types.ModuleType("huggingface_hub.utils._errors")
        _errors_shim.RepositoryNotFoundError = _hf_errors.RepositoryNotFoundError
        sys.modules["huggingface_hub.utils._errors"] = _errors_shim

        import yolov5  # noqa: F401

    # create folders for the dataset
    folder_name: str = os.path.join("results", f"{dataset_name}-{lf_model_name}-{hf_model_name}-{seed}-{latent_size}")

    print(f"Results for this experiment located at '{os.path.abspath(folder_name)}'")
    print("Please see 'experiments.log' for the logs")

    model_folder: str = os.path.join(folder_name, "models")
    file_folder: str = os.path.join(folder_name, "files")
    image_folder: str = os.path.join(folder_name, "images")
    latent_folder: str = os.path.join(folder_name, "latent")

    if comparisons_only:
        lf_checkpoint = os.path.join(model_folder, "lf_model.pt")
        hf_checkpoint = os.path.join(model_folder, "hf_model.pt")
        if not (os.path.isfile(lf_checkpoint) and os.path.isfile(hf_checkpoint)):
            raise FileNotFoundError(
                f"--comparisons_only requires an existing, already-trained experiment at "
                f"'{folder_name}' (same dataset/lf_model/hf_model/seed/latent_size as "
                f"{config_file!r}) - expected both '{lf_checkpoint}' and '{hf_checkpoint}' "
                f"to exist, found: lf={os.path.isfile(lf_checkpoint)} hf={os.path.isfile(hf_checkpoint)}."
            )
        # Reuse THIS experiment's own checkpoints specifically (not whatever
        # config.classifier_training.trained_lf_model/trained_hf_model says,
        # which may be unset - e.g. when the original run trained LF/HF from
        # scratch, those weights only ever landed here) - this is what lets
        # the existing "Pretrained {LF,HF} model found; skipping train" path
        # below skip training with no further changes needed.
        config.classifier_training.trained_lf_model = lf_checkpoint
        config.classifier_training.trained_hf_model = hf_checkpoint

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

    # initialize logger - 'a' (append) in comparisons_only mode so the
    # original run's log (LF/HF training, gate search) isn't erased; 'w'
    # otherwise, same as before.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(folder_name, "experiment.log"),
        filemode='a' if comparisons_only else 'w'
    )

    logger = logging.getLogger(__name__)
    if comparisons_only:
        logger.info("=" * 70)
        logger.info("COMPARISONS-ONLY APPEND RUN (--comparisons_only)")
        logger.info("=" * 70)
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

    if comparisons_only:
        # Merge onto the original run's summary.json instead of starting
        # fresh - the FE/gate-search phase's timing/result_files/etc. only
        # exist there (this run never recomputes them), and the final write
        # below would otherwise silently erase that history.
        existing_summary_path = os.path.join(folder_name, "summary.json")
        if os.path.isfile(existing_summary_path):
            with open(existing_summary_path) as f:
                existing_summary = json.load(f)
            for key in ("timing", "peak_gpu_memory_mb", "cost_realism", "final_accuracy"):
                run_summary[key] = {**existing_summary.get(key, {}), **run_summary[key]}
            run_summary["result_files"] = list(dict.fromkeys(
                existing_summary.get("result_files", []) + run_summary["result_files"]
            ))

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

    # -inf (not 0.0): guarantees the first epoch's checkpoint always gets
    # saved even if val accuracy never climbs above exactly 0% for the whole
    # run (a real, observed failure mode with very small epoch budgets) -
    # otherwise the strict ">" comparison below never fires, no checkpoint
    # file is ever written, and the load_state_dict() after the epoch loop
    # crashes with FileNotFoundError instead of training proceeding with a
    # merely-bad model.
    lf_best_acc: float = float('-inf')
    hf_best_acc: float = float('-inf')

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

    # Reported against train too, so it's obvious up front how much LF/HF are
    # overfitting (train acc vs val/test acc) without having to dig through
    # the cached latents after the fact.
    lf_train_loss, lf_train_acc = classifier_one_run(
        model=lf_model,
        dataloader=train_dl,
        criterion=nn.CrossEntropyLoss(),
        fidelity="lf",
        train_body=train_body,
    )

    hf_train_loss, hf_train_acc = classifier_one_run(
        model=hf_model,
        dataloader=train_dl,
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
    logger.info("Final Train Summary")
    logger.info(f"\tLF Train  Loss: {lf_train_loss:.4f}  | Accuracy: {100 * lf_train_acc:.2f}%")
    logger.info(f"\tHF Train  Loss: {hf_train_loss:.4f}  | Accuracy: {100 * hf_train_acc:.2f}%")

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
    run_summary["final_accuracy"]["lf_train_acc"] = lf_train_acc
    run_summary["final_accuracy"]["hf_train_acc"] = hf_train_acc

    # Hoisted above the (possibly-skipped, see comparisons_only below) FE
    # training/gate-search block - the comparison baselines need this too,
    # and it's just a literal, not a product of anything in that block.
    usage_values: List[float] = [i/10 for i in range(1, 11)]

    # Skipped entirely in --comparisons_only mode (latent caching, the FE
    # gate search, gate-sensitivity/routing plots, and the FE+SR grid all
    # reuse what the original run already produced on disk - see
    # COMPARISON BASELINES SETUP below for what actually still runs).
    if not comparisons_only:
        # ================================================================
        # FE MODEL TRAINING
        # save LF/HF latents, then LogSeededGreedySearch over usage_values
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

        # Continuous IoU-and-miss-based detection loss (helpers.compute_yolo_detection_loss),
        # replacing the old binary 1-correct signal, at the threshold validated against the
        # old boolean accuracy (agreement within <0.5pp - see llvip_validate_continuous_loss.py,
        # now in src/historical/). Only meaningful for YOLO (the only model type whose raw
        # output isn't directly comparable to a label via CE) - a no-op flag for every other
        # model type, where save_latent never looks at it.
        is_yolo_gate = lf_model_name == "yolo" or hf_model_name == "yolo"

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
            latent_key=gate_latent_key,
            yolo_continuous_loss=is_yolo_gate
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
            latent_key=gate_latent_key,
            yolo_continuous_loss=is_yolo_gate
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
            latent_key=gate_latent_key,
            yolo_continuous_loss=is_yolo_gate
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
        # crop's train split *used to be* (full spatial latents), that was ~98% of an
        # epoch's wall time (measured: ~190s of disk I/O vs. ~2.6s of actual gate-model
        # compute per epoch), all serialized in the main process under the default
        # num_workers=0. Parallelizing those reads across worker processes was the fix.
        # Now that save_latent pools spatial latents down to (C,) before caching (crop's
        # whole train split is ~530MB, not ~60GB), FE_Dataset's own RAM-budget check
        # (self._in_memory) almost always caches the split entirely in memory instead -
        # at that point, spawning/pickling-to 8 persistent worker processes to serve
        # batches that are already sitting in the main process's RAM is pure overhead
        # (observed to stall for 20+ minutes with no progress on crop's pooled latents,
        # instead of the ~1.7s/epoch a single-process in-memory DataLoader gets - see
        # results/crop_pooled_gate_experiment). Only pay for worker parallelism when a
        # split actually fell back to disk.
        fe_dataloader_workers = 0 if train_ds._in_memory else 8
        if fe_dataloader_workers == 0:
            logger.info("FE_Dataset splits fit in RAM - using num_workers=0 for gate dataloaders")

        # is_spatial_latent flags datasets whose lf_latent USED TO BE spatial
        # before save_latent (helpers.py) started pooling it down to (C,) on
        # disk - crop's UNet bottleneck and YOLO's raw backbone feature map, vs.
        # an already-flat (D,) latent (resnet/vit/mlp). Those datasets get
        # PooledGateMLP (a plain 2-layer MLP) instead of LatentCNNHead - a side
        # experiment (results/crop_pooled_gate_experiment) found the gate loses
        # nothing from only ever seeing the pooled vector (LatentCNNHead pooled
        # internally anyway), while training ~25x faster on ~195x smaller cached
        # latents and reaching meaningfully higher usage with no collapse step.
        is_spatial_latent = (dataset_name == "crop" or lf_model_name == "yolo")

        # Either way, size the gate model off the *actual* saved latent, not
        # config.latent_size - that assumption only holds for resnet/vit/mlp, which
        # explicitly project their backbone output down to latent_size via a
        # latent_rep layer. UNet's decoder output and YOLO's raw backbone feature
        # are each a fixed channel count of their own, independent of config.latent_size.
        lf_latent_channels: int = train_ds[0][0].shape[0]

        if is_spatial_latent:
            fe_model: nn.Module = build_model(
                model_name="pooled_gate",
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

        # Cached-FE-latent gate dataloaders plus the global (dataset-wide, fixed)
        # disagreement reweighting via WeightedRandomSampler - promoted to
        # helpers.build_gate_dataloaders_with_reweighting (previously inlined
        # here and separately copy-pasted into every per-dataset 5-rerun sweep
        # script). Measures hf_needed_rate from one plain pass over train_ds
        # before building the real sampler-weighted train_dl - see that
        # function's docstring for the full rationale (originally documented
        # inline here, now there since this logic no longer lives only in
        # main.py).
        train_dl, val_dl, test_dl, hf_needed_rate = build_gate_dataloaders_with_reweighting(
            train_ds, val_ds, test_ds, config.fe_training.batch_size, workers=fe_dataloader_workers,
            max_weight=config.fe_training.disagreement_weight_cap
        )
        logger.info(f"HF-needed rate on train set: {hf_needed_rate:.2%}")
        if config.fe_training.disagreement_weight_cap is not None:
            logger.info(f"Disagreement oversampling weight capped at {config.fe_training.disagreement_weight_cap}")

        # Below this, an unweighted/un-clipped gate reliably collapses within a
        # handful of epochs regardless of cost (observed on both LLVIP/YOLO's ~5%
        # rate and toy_2d's ~7%).
        imbalanced_gate = hf_needed_rate < 0.15

        # config.fe_training.gate_lr overrides the imbalanced_gate-based default
        # below when explicitly set - see FESettings' docstring on that field for
        # the validated (CUB-only) finding behind this override.
        gate_lr = config.fe_training.gate_lr if config.fe_training.gate_lr is not None else (5e-5 if imbalanced_gate else 3e-4)

        # Per-pixel accuracy lookup for segmentation only (crop) - lets
        # LogSeededGreedySearch report the FE/oracle curves' accuracy on the same
        # per-pixel metric as this run's own "Final Test Summary" above, instead
        # of compute_fidelity_loss_correct's per-image-majority "*_correct" flag.
        # None for every other dataset (classification), where the two are
        # identical anyway - see load_pixel_acc_lookup/compute_fidelity_loss_correct.
        val_pixel_acc = load_pixel_acc_lookup(val_latent_folder) if dataset_name == "crop" else None
        test_pixel_acc = load_pixel_acc_lookup(test_latent_folder) if dataset_name == "crop" else None

        adaptive_search: LogSeededGreedySearch = LogSeededGreedySearch(
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
            gate_lr=gate_lr,
            gate_grad_clip_norm=1.0 if imbalanced_gate else None,
            seed=seed,
            log_seed_points=config.fe_training.log_seed_points,
            refinement_budget=config.fe_training.refinement_budget,
            gap_tolerance=config.fe_training.gap_tolerance,
            wide_bracket_ratio=config.fe_training.wide_bracket_ratio,
            val_pixel_acc=val_pixel_acc,
            test_pixel_acc=test_pixel_acc
        )

        reset_peak_memory(DEVICE)
        gate_search_start = time.perf_counter()

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

        # Post-sweep step (default on - see FESettings.run_diagnostic_suite):
        # per-rerun per-sample routing CSVs (filename/class, fe_0.1..fe_1.0) built
        # straight from adaptive_search.routing_snapshots (already holds every
        # (rerun, target_usage) combination run_reruns just populated - no
        # retraining, no re-routing), then the three diagnostics.py analyses
        # (entry-usage-vs-gain, never-escalated gain distribution, class/scene
        # clustering) - the same suite previously only available via standalone
        # per-dataset scripts (cub_log_seeded_greedy_5run.py etc.), now wired
        # into main.py directly.
        sample_identity_dataset = {
            "bird_grayscale": "cub", "bird_color": "cub", "crop": "crop", "llvip": "llvip",
        }.get(dataset_name)

        if config.fe_training.run_diagnostic_suite and sample_identity_dataset is None:
            logger.warning(
                f"run_diagnostic_suite is True but dataset_name={dataset_name!r} has no registered "
                f"sample_identity mapping (only bird_grayscale/bird_color/crop/llvip are supported) - "
                f"skipping the diagnostic suite for this run. Set run_diagnostic_suite: False explicitly "
                f"for this dataset's config to suppress this warning, or add a mapping in sample_identity.py."
            )
        elif config.fe_training.run_diagnostic_suite:
            logger.info("RUNNING DIAGNOSTIC SUITE (per-sample CSVs + entry-usage/never-escalated/class-clustering)")

            identity_fn = functools.partial(get_test_sample_identity, sample_identity_dataset)
            filenames, class_names = identity_fn()

            target_usage_all = np.array([s["target_usage"] for s in adaptive_search.routing_snapshots])
            rerun_all = np.array([s["rerun"] for s in adaptive_search.routing_snapshots])
            for rerun_idx in range(fe_reruns):
                this_rerun = np.where(rerun_all == rerun_idx)[0]
                order = this_rerun[np.argsort(target_usage_all[this_rerun])]
                target_usage_sorted = target_usage_all[order]
                choice_sorted = np.stack([adaptive_search.routing_snapshots[i]["choice"] for i in order], axis=0)
                idx_sorted = np.stack([adaptive_search.routing_snapshots[i]["idx"] for i in order], axis=0)

                df = build_routing_dataframe(target_usage_sorted, choice_sorted, idx_sorted, filenames, class_names)
                csv_path = os.path.join(file_folder, f"per_sample_routing_rerun{rerun_idx}.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {csv_path}  shape={df.shape}")

            run_full_diagnostic_suite(
                checkpoint=os.path.basename(folder_name),
                label=f"{dataset_name} ({lf_model_name}/{hf_model_name})",
                identity_fn=identity_fn,
                n_reruns=fe_reruns,
            )

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

        # Mean +/- std band across reruns - only meaningful with >1 rerun; at
        # reruns=1 every point's std is exactly 0, so this is skipped rather than
        # rendering a degenerate zero-width band. Per-rerun individual plots are
        # deliberately NOT generated here - each rerun's own (r, usage) pairs
        # already live in adaptive_search.search_diagnostics/gate_search_log.npz
        # if ever needed, without a pile of near-duplicate image files.
        if fe_reruns > 1:
            plot_usage_vs_ch_std(
                adaptive_search.search_diagnostics,
                os.path.join(image_folder, "usage_vs_ch_std.png"),
                dataset_label=dataset_name
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
        # threshold grid instead - see LogSeededGreedySearch.run_fe_sr_grid. Runs
        # regardless of run_comparisons - it's a natural extension of the FE
        # results above, not one of the classification-shaped SelectiveNet/SAT/SR
        # baselines below (which YOLO can't run at all).
        #
        # Scoped per-rerun (run_fe_sr_grid's model_folder override, reading
        # model_folder/rerun_{i}/ - see LogSeededGreedySearch.run_reruns) rather
        # than one call pooling every rerun's checkpoints together - gives FE+SR
        # its own std across reruns in fe_results_5runs.npz-style
        # usage_std/acc_std, the same treatment FE already gets, instead of a
        # single curve with no spread. At fe_reruns=1 this is exactly one call
        # scoped to model_folder/rerun_0/ (or model_folder itself - run_reruns
        # only creates the rerun_0/ nesting when n_reruns > 1), identical to the
        # old pooled behavior in that case.
        # ================================================================
        logger.info("RUNNING FE + SOFTMAX RESPONSE")

        fe_sr_threshold_grid: List[float] = list(np.linspace(0.0, 1.0, 21))
        fe_sr_usage_runs = np.zeros((fe_reruns, len(usage_values)))
        fe_sr_acc_runs = np.zeros((fe_reruns, len(usage_values)))

        for rerun_idx in range(fe_reruns):
            rerun_model_folder = os.path.join(model_folder, f"rerun_{rerun_idx}") if fe_reruns > 1 else model_folder
            fe_sr = adaptive_search.run_fe_sr_grid(
                usage_values=usage_values,
                threshold_grid=fe_sr_threshold_grid,
                model_folder=rerun_model_folder
            )
            fe_sr_usage_runs[rerun_idx] = fe_sr["usage"]
            fe_sr_acc_runs[rerun_idx] = fe_sr["acc"]

            fe_sr_grid_path: str = os.path.join(file_folder, f"fe_sr_grid_rerun{rerun_idx}.npz")
            np.savez(
                fe_sr_grid_path,
                r_grid=fe_sr["r_grid"],
                threshold_grid=fe_sr["threshold_grid"],
                val_usage_grid=fe_sr["val_usage_grid"],
                val_acc_grid=fe_sr["val_acc_grid"],
                test_usage_grid=fe_sr["test_usage_grid"],
                test_acc_grid=fe_sr["test_acc_grid"]
            )
            run_summary["result_files"].append(f"fe_sr_grid_rerun{rerun_idx}.npz")

        fe_sr_usage_vals = fe_sr_usage_runs.mean(axis=0)
        fe_sr_acc_vals = fe_sr_acc_runs.mean(axis=0)
        fe_sr_usage_std = fe_sr_usage_runs.std(axis=0)
        fe_sr_acc_std = fe_sr_acc_runs.std(axis=0)

        fe_sr_data_path: str = os.path.join(file_folder, "fe_sr_results.npz")
        np.savez(
            fe_sr_data_path,
            usage=fe_sr_usage_vals,
            acc=fe_sr_acc_vals,
            usage_std=fe_sr_usage_std,
            acc_std=fe_sr_acc_std,
            usage_runs=fe_sr_usage_runs,
            acc_runs=fe_sr_acc_runs,
            target_usage=fe_sr["target_usage"],
            chosen_r=fe_sr["chosen_r"],
            chosen_threshold=fe_sr["chosen_threshold"]
        )
        run_summary["result_files"].append("fe_sr_results.npz")

        # ends the script if you don't want comparisons
        if not run_comparisons:
            logger.info("PLOTTING PARETO CURVES")

            # lf_correct/hf_correct only depend on the LF/HF models' own
            # predictions on the fixed test set, not on which gate model produced
            # a given snapshot — so any single snapshot's arrays (here, the
            # first) give the ground truth. lf_pixel_acc/hf_pixel_acc (only
            # present for segmentation - see LogSeededGreedySearch.evaluate_fe_model)
            # score the oracle on true per-pixel accuracy instead of falling back
            # to the per-image-majority correct/incorrect flag. Dense (every
            # possible usage level, not just the 10 usage_values targets) -
            # see compute_dense_oracle_curve's docstring.
            pareto_lf_correct = routing_snapshots["lf_correct"][0]
            pareto_hf_correct = routing_snapshots["hf_correct"][0]
            pareto_lf_pixel_acc = routing_snapshots.get("lf_pixel_acc", [None])[0]
            pareto_hf_pixel_acc = routing_snapshots.get("hf_pixel_acc", [None])[0]

            oracle = compute_dense_oracle_curve(
                pareto_lf_correct, pareto_hf_correct,
                lf_pixel_acc=pareto_lf_pixel_acc, hf_pixel_acc=pareto_hf_pixel_acc
            )

            pareto_curves = load_pareto_curves(file_folder)
            render_pareto_plot(
                pareto_curves, dataset_label=dataset_name,
                save_path=os.path.join(image_folder, "pareto.png"),
                oracle=oracle, oracle_label="Oracle (budget-constrained)"
            )

            # Second, separate plot: unconnected scatter points, the uncapped
            # oracle (can decline past its peak - see compute_dense_oracle_curve's
            # clip_negative_gain docstring), and a dotted "random routing"
            # reference line between LF-alone and HF-alone accuracy.
            oracle_uncapped = compute_dense_oracle_curve(
                pareto_lf_correct, pareto_hf_correct,
                lf_pixel_acc=pareto_lf_pixel_acc, hf_pixel_acc=pareto_hf_pixel_acc,
                clip_negative_gain=False
            )
            lf_acc_pct = 100 * (pareto_lf_pixel_acc.mean() if pareto_lf_pixel_acc is not None else pareto_lf_correct.astype(float).mean())
            hf_acc_pct = 100 * (pareto_hf_pixel_acc.mean() if pareto_hf_pixel_acc is not None else pareto_hf_correct.astype(float).mean())
            render_pareto_scatter_plot(
                pareto_curves, dataset_label=dataset_name,
                save_path=os.path.join(image_folder, "pareto_scatter.png"),
                oracle=oracle_uncapped, oracle_label="Oracle (budget-constrained)",
                random_baseline=(lf_acc_pct, hf_acc_pct)
            )

            run_summary["timing"]["total_experiment_sec"] = time.perf_counter() - experiment_start_time
            with open(os.path.join(folder_name, "summary.json"), "w") as f:
                json.dump(run_summary, f, indent=2)
            logger.info("EXPERIMENT FINISHED")
            return

    # ================================================================
    # COMPARISON BASELINES SETUP
    # epoch budgets + model-sharing disclosure, resolved once up front so SR/
    # SelectiveNet/SAT's sections below can all reference the same values and
    # so a reader sees the full picture before any individual method's log
    # lines start.
    # ================================================================
    num_baseline_epochs: int = config.classifier_training.epochs
    # SelectiveNet trains one model per c - was hardcoded to 40 regardless of
    # dataset/model; now overridable via config.selectivenet_training.epochs,
    # falling back to that same 40 when unset.
    num_selnet_epochs: int = (
        config.selectivenet_training.epochs if config.selectivenet_training.epochs is not None else 40
    )
    sat_alpha = 0.99
    # EMA memory ~= 1/(1-alpha) epochs (one update per sample per epoch) -
    # below that, the abstain-probability targets haven't had time to diverge
    # much from their one-hot init, independent of anything else about the
    # run. config.sat_training.epochs overrides classifier_training.epochs
    # for SAT specifically; falls back to the shared value when unset so
    # every existing config keeps its current behavior unless it opts in.
    num_sat_epochs: int = config.sat_training.epochs if config.sat_training.epochs is not None else num_baseline_epochs
    sat_ema_memory_epochs = 1 / (1 - sat_alpha)
    if num_sat_epochs < sat_ema_memory_epochs:
        logger.warning(
            f"SAT training for {num_sat_epochs} epochs, but alpha={sat_alpha} gives an EMA memory of "
            f"~{sat_ema_memory_epochs:.0f} epochs - the abstain head may be undertrained. Set "
            f"sat_training.epochs explicitly to raise this independently of classifier_training.epochs."
        )

    logger.info("COMPARISON BASELINES: epoch budgets and model-sharing")
    logger.info(f"\tSR: no training (threshold-only on the frozen, already-trained lf_model/hf_model)")
    logger.info(f"\tSelectiveNet: {num_selnet_epochs} epochs per c value ({len(usage_values)} c values trained)")
    logger.info(f"\tSAT: {num_sat_epochs} epochs (single model)")
    logger.info(
        "\tSR reuses the shared lf_model directly for its 'kept' predictions. SelectiveNet and SAT each "
        "train their own classifier on lf_data from scratch instead - that classifier IS their 'LF' role, "
        "not a wrapper around the project's separately-trained lf_model - so their usage=0 accuracy can "
        "differ substantially from lf_model's own accuracy (see each method's usage=0 accuracy log line "
        "below)."
    )

    def log_usage_mismatch(method_label: str, target_usage: float, realized_usage_pct: float, tol_pp: float = 5.0):
        """Common >5pp target-vs-realized usage check, shared by SR/
        SelectiveNet(native)/SelectiveNet(calibrated)/SAT. For SelectiveNet's
        native (fixed-threshold-0.5) variant this is expected to fire
        routinely - it's reported as a finding about the method, not treated
        as a bug to go fix."""
        err_pp = abs(realized_usage_pct - 100 * target_usage)
        if err_pp > tol_pp:
            logger.warning(
                f"{method_label}: target usage {100*target_usage:.2f}% missed by {err_pp:.2f}pp "
                f"(realized {realized_usage_pct:.2f}%)"
            )

    def save_incremental(path: str, arrays: dict):
        """Rewrites path's full .npz from arrays (each value a list/array -
        the full accumulated-so-far sequence, not just the newest point).
        Called after every single usage target in SR/SelectiveNet/SAT's loops
        below, not just once at the end - so a crash or kill partway through
        a long run (e.g. SelectiveNet retraining 10 full models) still
        leaves every target computed up to that point safely on disk,
        instead of losing the entire method's results because the one
        np.savez call never ran. Cheap: these arrays are at most
        len(usage_values) long."""
        np.savez(path, **{k: np.array(v) for k, v in arrays.items()})

    # ================================================================
    # SOFTMAX RESPONSE (SR) BASELINE
    # ================================================================
    logger.info("RUNNING DEFAULT SOFTMAX RESPONSE")

    softmax_response: SoftmaxResponseMethod = SoftmaxResponseMethod(
        lf_model=lf_model,
        hf_model=hf_model,
        val_dl=cascade_val_dl,
        test_dl=cascade_test_dl,
        device=DEVICE,
        train_body=train_body,
        hf_input_mode=hf_input_mode,
    )

    # get_softmax_thresholds returns dicts keyed by target usage (not by
    # threshold - several usage targets can land on the same threshold once
    # confidence saturates, which used to silently collide in a
    # threshold-keyed dict). acc_vals is cascade accuracy (LF on admitted +
    # HF on escalated), the same metric SelectiveNet/SAT report below.
    sr_acc_vals, sr_usage_vals = softmax_response.get_softmax_thresholds(
        usage_list=usage_values
    )

    target_usage_arr = np.array(list(sr_acc_vals.keys()))
    sr_selective_acc_vals = np.array([softmax_response.last_selective_acc[u] for u in target_usage_arr])
    thresholds = np.array([softmax_response.last_thresholds[u] for u in target_usage_arr])
    sr_val_usage_vals = np.array([softmax_response.last_val_usage[u] for u in target_usage_arr])
    sr_acc_vals = np.array(list(sr_acc_vals.values()))
    sr_usage_vals = np.array(list(sr_usage_vals.values()))

    for i, target_usage in enumerate(target_usage_arr):
        log_usage_mismatch("SR", float(target_usage), float(sr_usage_vals[i]))

    # SR's usage=0 accuracy is exactly lf_test_acc (already computed in the
    # Final Test Summary above) - usage=0 means nobody escalated, i.e. pure
    # LF predictions, and SR is the one baseline that reuses the shared
    # lf_model directly (no separate classifier of its own).
    logger.info(f"SR usage=0 accuracy (shared lf_model's own full-test accuracy): {100*lf_test_acc:.2f}%")

    sr_data_path: str = os.path.join(file_folder, "sr_results.npz")
    np.savez(
        sr_data_path,
        usage=sr_usage_vals,
        val_usage=sr_val_usage_vals,
        acc=sr_acc_vals,  # cascade accuracy
        selective_acc=sr_selective_acc_vals,  # LF-only accuracy on admitted samples
        target_usage=target_usage_arr,
        thresholds=thresholds,
        usage0_acc=100 * lf_test_acc
    )
    run_summary["result_files"].append("sr_results.npz")

    # ================================================================
    # SELECTIVENET MODEL TRAINING
    # one model per c (= 1 - usage target), checkpointed on val accuracy at
    # the fixed threshold=0.5 used during training - exactly as before. Both
    # variants below reuse that SAME trained model; no extra training:
    #   - native (primary/paper-faithful): fixed threshold=0.5, no post-hoc
    #     calibration - whatever usage that produces is reported as-is.
    #   - calibrated (secondary): threshold set per target usage from the
    #     target_usage-quantile of val's selection scores g, mirroring SAT's
    #     calibrate_tau.
    # Replaces the old (c, threshold) grid search entirely - that grid
    # existed to approximate per-target threshold calibration one point at a
    # time; get_selection_probs/calibrate_threshold now do that exactly in
    # closed form, the same way SAT's calibrate_tau already did.
    # ================================================================
    c_grid: List[float] = [1 - u for u in usage_values]  # c is target COVERAGE, not usage

    logger.info("RUNNING SELECTIVENET (native + calibrated)")

    # Native kept as "selectivenet_results.npz" (unchanged filename) since
    # it's the primary/paper-faithful result - existing tooling that only
    # knows about this one file still gets the right series.
    selectivenet_native_path: str = os.path.join(file_folder, "selectivenet_results.npz")
    selectivenet_calibrated_path: str = os.path.join(file_folder, "selectivenet_calibrated_results.npz")

    selnet_param_count = None
    reset_peak_memory(DEVICE)
    selnet_train_start = time.perf_counter()

    selnet_native_usage_vals: List[float] = []
    selnet_native_val_usage_vals: List[float] = []
    selnet_native_acc_vals: List[float] = []
    selnet_native_selacc_vals: List[float] = []

    selnet_calib_usage_vals: List[float] = []
    selnet_calib_val_usage_vals: List[float] = []
    selnet_calib_acc_vals: List[float] = []
    selnet_calib_selacc_vals: List[float] = []
    selnet_calib_thresholds: List[float] = []

    selnet_usage0_acc: float | None = None  # set + logged once, on the first (smallest-target_usage) iteration

    for target_usage, c_val in zip(usage_values, c_grid):
        logger.info(f"SelectiveNet target usage={100*target_usage:.2f}% (c={c_val:.2f})")

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
        # -inf, not 0.0 - see lf_best_acc's comment above for why.
        selnet_best_acc: float = float('-inf')

        selnet_epoch_logger = EpochMetricsLogger(
            os.path.join(file_folder, f"selnet_epoch_metrics_c{c_val:.2f}.csv"),
            ["epoch", "train_loss", "train_acc", "train_usage", "val_loss", "val_acc", "val_usage", "epoch_time_sec"]
        )

        for epoch in range(num_selnet_epochs):
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

            # Checkpoint selection criterion: val accuracy AT THE FIXED
            # threshold=0.5 used for training (not val usage, not a
            # threshold sweep) - unchanged from before. Noted explicitly
            # here since it's a real, somewhat arbitrary choice: a model
            # could have higher val accuracy at a different threshold and
            # still lose this comparison.
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

        logger.info(
            f"SelectiveNet c={c_val:.2f}: checkpoint selected by val cascade accuracy at the fixed "
            f"threshold=0.5 ({100*selnet_best_acc:.2f}%)"
        )
        selnet_model.load_state_dict(torch.load(selnet_model_file, weights_only=True))

        # usage=0 accuracy (this model's own full-test accuracy, forced via
        # threshold=0.0 - every select_prob is >0, so this keeps everyone
        # regardless of saturation) - from the smallest-target_usage c-model,
        # the actually-trained operating point closest to usage=0 (0.0 isn't
        # itself in usage_values). Computed first (not after the loop) so
        # it's already available for this same iteration's incremental saves.
        if target_usage == min(usage_values):
            zero_test = selnet_method.one_run(model=selnet_model, dataloader=cascade_test_dl, hf_model=hf_model,
                                               threshold=0.0, train_body=train_body, hf_input_mode=hf_input_mode)
            selnet_usage0_acc = 100 * zero_test["accuracy"]
            logger.info(
                f"SelectiveNet usage=0 accuracy (own model from target_usage={min(usage_values):.2f}'s c, "
                f"forced to keep everyone): {selnet_usage0_acc:.2f}%"
            )

        # --- Native: fixed threshold=0.5, no calibration (primary result) ---
        native_val = selnet_method.one_run(model=selnet_model, dataloader=cascade_val_dl, hf_model=hf_model,
                                            threshold=0.5, train_body=train_body, hf_input_mode=hf_input_mode)
        native_test = selnet_method.one_run(model=selnet_model, dataloader=cascade_test_dl, hf_model=hf_model,
                                             threshold=0.5, train_body=train_body, hf_input_mode=hf_input_mode)
        logger.info(
            f"SelectiveNet (native) target usage {100*target_usage:.2f}% -> threshold=0.5 (fixed) | "
            f"realized val usage {100*native_val['usage']:.2f}% | realized test usage {100*native_test['usage']:.2f}% | "
            f"cascade acc {100*native_test['accuracy']:.2f}% | selective acc {100*native_test['selective_accuracy']:.2f}%"
        )
        # Native's own usage is whatever the fixed threshold happens to produce
        # for this c - a real mismatch here is an expected property of the
        # method, not a bug (see docstring/comment above and item 1 in the task).
        log_usage_mismatch("SelectiveNet (native)", target_usage, 100 * native_test["usage"])
        selnet_native_usage_vals.append(100 * native_test["usage"])
        selnet_native_val_usage_vals.append(100 * native_val["usage"])
        selnet_native_acc_vals.append(100 * native_test["accuracy"])
        selnet_native_selacc_vals.append(100 * native_test["selective_accuracy"])

        # Rewritten after every target (not just once at the end) - see
        # save_incremental's docstring.
        save_incremental(selectivenet_native_path, {
            "usage": selnet_native_usage_vals,
            "val_usage": selnet_native_val_usage_vals,
            "acc": selnet_native_acc_vals,
            "selective_acc": selnet_native_selacc_vals,
            "target_usage": usage_values[:len(selnet_native_usage_vals)],
            "usage0_acc": selnet_usage0_acc,
        })

        # --- Calibrated: val-quantile threshold for this exact target usage ---
        val_g = selnet_method.get_selection_probs(selnet_model, cascade_val_dl, train_body=train_body)
        calib_threshold = selnet_method.calibrate_threshold(val_g, target_usage)
        calib_val = selnet_method.one_run(model=selnet_model, dataloader=cascade_val_dl, hf_model=hf_model,
                                           threshold=calib_threshold, train_body=train_body, hf_input_mode=hf_input_mode)
        calib_test = selnet_method.one_run(model=selnet_model, dataloader=cascade_test_dl, hf_model=hf_model,
                                            threshold=calib_threshold, train_body=train_body, hf_input_mode=hf_input_mode)
        logger.info(
            f"SelectiveNet (calibrated) target usage {100*target_usage:.2f}% -> "
            f"val-calibrated threshold={calib_threshold:.4f} | "
            f"realized val usage {100*calib_val['usage']:.2f}% | realized test usage {100*calib_test['usage']:.2f}% | "
            f"cascade acc {100*calib_test['accuracy']:.2f}% | selective acc {100*calib_test['selective_accuracy']:.2f}%"
        )
        log_usage_mismatch("SelectiveNet (calibrated)", target_usage, 100 * calib_test["usage"])
        selnet_calib_usage_vals.append(100 * calib_test["usage"])
        selnet_calib_val_usage_vals.append(100 * calib_val["usage"])
        selnet_calib_acc_vals.append(100 * calib_test["accuracy"])
        selnet_calib_selacc_vals.append(100 * calib_test["selective_accuracy"])
        selnet_calib_thresholds.append(calib_threshold)

        # Rewritten after every target - see save_incremental's docstring.
        save_incremental(selectivenet_calibrated_path, {
            "usage": selnet_calib_usage_vals,
            "val_usage": selnet_calib_val_usage_vals,
            "acc": selnet_calib_acc_vals,
            "selective_acc": selnet_calib_selacc_vals,
            "target_usage": usage_values[:len(selnet_calib_usage_vals)],
            "chosen_threshold": selnet_calib_thresholds,
            "usage0_acc": selnet_usage0_acc,
        })

        # GPU memory cleanup, once per c-value - without this, a growing
        # CUDA allocator reservation across the 10 fresh ResNet18+optimizer
        # builds in this loop causes a real, observed escalating slowdown
        # (25min -> 42min per c-value in one run, 19min -> 81min in a
        # concurrent one) and eventually near-exhausts a 24GB GPU (measured
        # at 23.6/24.5GB mid-run). del + empty_cache() + gc.collect() forces
        # the allocator to release its reserved-but-unused blocks back to
        # the driver at each iteration boundary instead of letting them
        # accumulate.
        del selnet_model, selnet_optimizer
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    selnet_train_time_sec = time.perf_counter() - selnet_train_start
    selnet_peak_mem_mb = get_peak_memory_mb(DEVICE)
    logger.info(f"SelectiveNet train time (all c values): {format_duration(selnet_train_time_sec)}")
    logger.info(f"SelectiveNet phase peak GPU memory: {selnet_peak_mem_mb:.1f} MB")

    run_summary["timing"]["selectivenet_train_sec"] = selnet_train_time_sec
    run_summary["peak_gpu_memory_mb"]["selectivenet_phase"] = selnet_peak_mem_mb

    # Both files are already fully written and up to date - the in-loop
    # save_incremental calls above wrote the complete arrays on the last
    # (and every prior) iteration, so there's nothing left to save here.
    run_summary["result_files"].append("selectivenet_results.npz")
    run_summary["result_files"].append("selectivenet_calibrated_results.npz")

    # ================================================================
    # SAT MODEL TRAINING
    # a single model, then a threshold grid search (mirrors SelectiveNet's
    # selection logic but with no c dimension)
    # ================================================================
    logger.info("RUNNING DEFAULT SELF-ADAPTIVE TRAINING (SAT)")

    sat_alpha = 0.99
    # EMA memory ~= 1/(1-alpha) epochs (one update per sample per epoch) -
    # below that, the abstain-probability targets haven't had time to diverge
    # much from their one-hot init, independent of anything else about the
    # run. config.sat_training.epochs overrides classifier_training.epochs
    # for SAT specifically; falls back to the shared value when unset so
    # every existing config keeps its current behavior unless it opts in.
    num_sat_epochs: int = config.sat_training.epochs if config.sat_training.epochs is not None else num_baseline_epochs
    sat_ema_memory_epochs = 1 / (1 - sat_alpha)
    if num_sat_epochs < sat_ema_memory_epochs:
        logger.warning(
            f"SAT training for {num_sat_epochs} epochs, but alpha={sat_alpha} gives an EMA memory of "
            f"~{sat_ema_memory_epochs:.0f} epochs - the abstain head may be undertrained. Set "
            f"sat_training.epochs explicitly to raise this independently of classifier_training.epochs."
        )

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
        alpha=sat_alpha,
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
    # -inf, not 0.0 - see lf_best_acc's comment above for why.
    sat_best_acc: float = float('-inf')

    reset_peak_memory(DEVICE)
    sat_train_start = time.perf_counter()
    sat_epoch_logger = EpochMetricsLogger(
        os.path.join(file_folder, "sat_epoch_metrics.csv"),
        ["epoch", "train_loss", "train_acc", "train_usage", "val_loss", "val_acc", "val_usage", "epoch_time_sec"]
    )

    for epoch in range(num_sat_epochs):
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

    # Per-usage-target tau calibration (quantile of val's abstain-probability
    # distribution), replacing a coarse 21-point threshold_grid sweep + nearest-
    # match selection (SelectiveNet's approach, reused here previously) with an
    # exact val-quantile calibration - SAT's tau is a single scalar cutoff over
    # one abstain-probability distribution, so (unlike SelectiveNet's joint
    # (c, threshold) grid, which has a real second training-time dimension) a
    # grid search over it was only approximating what a quantile gives exactly.
    sat_val_abstain_probs = sat_method.get_abstain_probs(
        model=sat_model, dataloader=sat_val_dl, train_body=train_body
    )

    # usage=0 accuracy: SAT's own model (not the shared lf_model), forced via
    # tau=+inf to keep everyone (calibrate_tau(..., 0) already returns +inf).
    # Computed up front (cheap - one eval pass, no retraining) so it's
    # available for the incremental saves inside the loop below, not just
    # the final one.
    sat_zero_tau = sat_method.calibrate_tau(sat_val_abstain_probs, 0.0)
    sat_zero_test = sat_method.one_run(
        model=sat_model, dataloader=sat_test_dl, hf_model=hf_model, tau=sat_zero_tau,
        train_body=train_body, epoch=num_sat_epochs, hf_input_mode=hf_input_mode
    )
    sat_usage0_acc = 100 * sat_zero_test["accuracy"]
    logger.info(f"SAT usage=0 accuracy (own model, forced to keep everyone): {sat_usage0_acc:.2f}%")

    sat_data_path: str = os.path.join(file_folder, "sat_results.npz")
    sat_usage_vals: List[float] = []
    sat_val_usage_vals: List[float] = []
    sat_acc_vals: List[float] = []
    sat_selacc_vals: List[float] = []
    sat_chosen_tau: List[float] = []

    for target_usage in usage_values:
        tau = sat_method.calibrate_tau(sat_val_abstain_probs, target_usage)

        sat_val_metrics_at_tau = sat_method.one_run(
            model=sat_model,
            dataloader=sat_val_dl,
            hf_model=hf_model,
            tau=tau,
            train_body=train_body,
            epoch=num_sat_epochs,
            hf_input_mode=hf_input_mode
        )

        sat_test_metrics = sat_method.one_run(
            model=sat_model,
            dataloader=sat_test_dl,
            hf_model=hf_model,
            tau=tau,
            train_body=train_body,
            epoch=num_sat_epochs,
            hf_input_mode=hf_input_mode
        )

        logger.info(
            f"SAT target usage {100*target_usage:.2f}% -> val-calibrated tau={tau:.4f} | "
            f"realized val usage {100*sat_val_metrics_at_tau['usage']:.2f}% | "
            f"realized test usage {100*sat_test_metrics['usage']:.2f}% | "
            f"cascade acc {100*sat_test_metrics['accuracy']:.2f}% | "
            f"selective acc {100*sat_test_metrics['selective_accuracy']:.2f}%"
        )
        log_usage_mismatch("SAT", target_usage, 100 * sat_test_metrics["usage"])

        sat_usage_vals.append(100 * sat_test_metrics["usage"])
        sat_val_usage_vals.append(100 * sat_val_metrics_at_tau["usage"])
        sat_acc_vals.append(100 * sat_test_metrics["accuracy"])
        sat_selacc_vals.append(100 * sat_test_metrics["selective_accuracy"])
        sat_chosen_tau.append(tau)

        # Rewritten after every target (not just once at the end) - see
        # save_incremental's docstring.
        save_incremental(sat_data_path, {
            "usage": sat_usage_vals,
            "val_usage": sat_val_usage_vals,
            "acc": sat_acc_vals,
            "selective_acc": sat_selacc_vals,
            "target_usage": usage_values[:len(sat_usage_vals)],
            "chosen_tau": sat_chosen_tau,
            "usage0_acc": sat_usage0_acc,
        })

    run_summary["result_files"].append("sat_results.npz")

    # Consolidated target-vs-realized usage table across all three baselines,
    # now that SR/SelectiveNet/SAT have all run - makes it possible to see at
    # a glance how tightly each method actually hits the usage_values grid it
    # was calibrated against, rather than hunting through three separate
    # per-method log sections.
    logger.info("TARGET VS REALIZED USAGE (all baselines)")
    logger.info(
        f"{'target%':>8} | {'SR%':>8} | {'SelNet(native)%':>16} | {'SelNet(calib)%':>16} | {'SAT%':>8}"
    )
    for i, target_usage in enumerate(usage_values):
        logger.info(
            f"{100*target_usage:8.2f} | {sr_usage_vals[i]:8.2f} | "
            f"{selnet_native_usage_vals[i]:16.2f} | {selnet_calib_usage_vals[i]:16.2f} | {sat_usage_vals[i]:8.2f}"
        )
    logger.info(
        f"Usage=0 accuracy (own model): SR={100*lf_test_acc:.2f}% | "
        f"SelectiveNet={selnet_usage0_acc:.2f}% | SAT={sat_usage0_acc:.2f}%"
    )

    # ================================================================
    # COMPARISON DATA + PLOTS
    # Long-format per-seed CSV/npz/JSON (plot_selective_comparison.py) so
    # replot_pareto.py can remake the plots below without rerunning anything,
    # plus the four-series scatter/line/native-vs-calibrated plots themselves.
    # ================================================================
    comparison_rows = collect_comparison_rows(file_folder, seed=seed)
    save_comparison_data(comparison_rows, file_folder)
    render_all_comparison_plots(comparison_rows, dataset_label=dataset_name, image_folder=image_folder)

    # ================================================================
    # FINAL PLOTS
    # Pareto overlay of every baseline that produced a *_results.npz
    # ================================================================
    logger.info("PLOTTING PARETO CURVES")

    # In the normal flow, routing_snapshots was already loaded above (inside
    # the now-possibly-skipped FE block, for the gate-routing projection/
    # table plots). In --comparisons_only mode that block never ran, so
    # reload it here unconditionally instead - cheap, idempotent, and the
    # file is already on disk from the original run either way.
    routing_snapshots = load_routing_snapshots(file_folder)

    # lf_correct/hf_correct only depend on the LF/HF models' own predictions on
    # the fixed test set, not on which gate model produced a given snapshot —
    # so any single snapshot's arrays (here, the first) give the ground truth.
    # Dense (every possible usage level) - see compute_dense_oracle_curve's docstring.
    pareto_lf_correct = routing_snapshots["lf_correct"][0]
    pareto_hf_correct = routing_snapshots["hf_correct"][0]
    pareto_lf_pixel_acc = routing_snapshots.get("lf_pixel_acc", [None])[0]
    pareto_hf_pixel_acc = routing_snapshots.get("hf_pixel_acc", [None])[0]

    oracle = compute_dense_oracle_curve(
        pareto_lf_correct, pareto_hf_correct,
        lf_pixel_acc=pareto_lf_pixel_acc, hf_pixel_acc=pareto_hf_pixel_acc
    )

    pareto_curves = load_pareto_curves(file_folder)
    render_pareto_plot(
        pareto_curves, dataset_label=dataset_name,
        save_path=os.path.join(image_folder, "pareto.png"),
        oracle=oracle, oracle_label="Oracle (budget-constrained)"
    )

    # Second, separate plot: unconnected scatter points, the uncapped oracle
    # (can decline past its peak - see compute_dense_oracle_curve's
    # clip_negative_gain docstring), and a dotted "random routing" reference
    # line between LF-alone and HF-alone accuracy.
    oracle_uncapped = compute_dense_oracle_curve(
        pareto_lf_correct, pareto_hf_correct,
        lf_pixel_acc=pareto_lf_pixel_acc, hf_pixel_acc=pareto_hf_pixel_acc,
        clip_negative_gain=False
    )
    lf_acc_pct = 100 * (pareto_lf_pixel_acc.mean() if pareto_lf_pixel_acc is not None else pareto_lf_correct.astype(float).mean())
    hf_acc_pct = 100 * (pareto_hf_pixel_acc.mean() if pareto_hf_pixel_acc is not None else pareto_hf_correct.astype(float).mean())
    render_pareto_scatter_plot(
        pareto_curves, dataset_label=dataset_name,
        save_path=os.path.join(image_folder, "pareto_scatter.png"),
        oracle=oracle_uncapped, oracle_label="Oracle (budget-constrained)",
        random_baseline=(lf_acc_pct, hf_acc_pct)
    )

    run_summary["timing"]["total_experiment_sec"] = time.perf_counter() - experiment_start_time
    logger.info(f"Total experiment time: {format_duration(run_summary['timing']['total_experiment_sec'])}")
    with open(os.path.join(folder_name, "summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    logger.info("EXPERIMENT FINISHED")

if __name__ == "__main__":
    main()
