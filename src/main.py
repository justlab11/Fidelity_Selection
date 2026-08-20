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

from custom_types import ConfigOptions

@click.command()
@click.option("--config_file", default="../config.yml")
def main(config_file):
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

    lf_model: nn.Module = build_model(
        model_name=lf_model_name,
        input_size=lf_input_size,
        output_size=output_size,
        latent_size=latent_size
    )

    hf_model: nn.Module = build_model(
        model_name=hf_model_name,
        # HF model consumes the LF+HF channels concatenated (see classifier_one_run's
        # "hf" fidelity and the SelectiveNet/SAT cascades), so its input size is the sum,
        # not hf_input_size alone — get_hf_input_size() reports the HF-only channel count.
        input_size=lf_input_size + hf_input_size,
        output_size=output_size,
        latent_size=latent_size
    )

    lf_param_count = count_parameters(lf_model)
    hf_param_count = count_parameters(hf_model)
    logger.info(f"LF model parameters: {lf_param_count:,}")
    logger.info(f"HF model parameters: {hf_param_count:,}")
    run_summary["cost_realism"]["lf"] = {"num_parameters": lf_param_count}
    run_summary["cost_realism"]["hf"] = {"num_parameters": hf_param_count}

    if not train_body:
        logger.info(f"Parameter 'train_body' was set to False, saving the body outputs for faster running")
        
        train_body_folder: str = os.path.join(body_folder, "train")
        save_body(
            lf_model=lf_model,
            hf_model=hf_model,
            dataloader=train_dl,
            save_folder=train_body_folder,
            device=DEVICE
        )

        test_body_folder: str = os.path.join(body_folder, "test")
        save_body(
            lf_model=lf_model,
            hf_model=hf_model,
            dataloader=test_dl,
            save_folder=test_body_folder,
            device=DEVICE
        )

        val_body_folder: str = os.path.join(body_folder, "val")
        save_body(
            lf_model=lf_model,
            hf_model=hf_model,
            dataloader=val_dl,
            save_folder=val_body_folder,
            device=DEVICE
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

    lf_optimizer = torch.optim.Adam(lf_model.parameters(), lr=1e-5, weight_decay=1e-5)
    hf_optimizer = torch.optim.Adam(hf_model.parameters(), lr=1e-5, weight_decay=1e-5)

    lf_state_dict: Dict | None  = None
    hf_state_dict: Dict | None  = None

    lf_best_acc: float = 0.0
    hf_best_acc: float = 0.0

    lf_model_file = os.path.join(model_folder, "lf_model.pt")
    hf_model_file = os.path.join(model_folder, "hf_model.pt")

    # if the user provides models, use theirs
    train_lf_model = True
    trained_lf_path = config.classifier_training.trained_lf_model

    if trained_lf_path is not None:
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

    if trained_hf_path is not None:
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
        except:
            logger.info("Pretrained HF model failed to load; training HF model")
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
                optimizer=hf_optimizer
            )

            hf_val_loss, hf_val_acc = classifier_one_run(
                model=hf_model,
                dataloader=val_dl,
                criterion=nn.CrossEntropyLoss(),
                fidelity="hf",
                train_body=train_body,
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
    )

    logger.info("Final Test Summary")
    logger.info(f"\tLF Test   Loss: {lf_test_loss:.4f}  | Accuracy: {100 * lf_test_acc:.2f}%")
    logger.info(f"\tHF Test   Loss: {hf_test_loss:.4f}  | Accuracy: {100 * hf_test_acc:.2f}%")

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
        data = torch.cat([batch[0], batch[1]], dim=1).to(DEVICE, torch.float)
        return (hf_model(data) if train_body else hf_model.head(data))["output"]

    lf_inference_latency_ms = measure_inference_latency(lf_forward, cascade_test_dl, DEVICE)
    hf_inference_latency_ms = measure_inference_latency(hf_forward, cascade_test_dl, DEVICE)
    logger.info(f"LF inference latency: {lf_inference_latency_ms:.3f} ms/sample")
    logger.info(f"HF inference latency: {hf_inference_latency_ms:.3f} ms/sample")

    run_summary["cost_realism"]["lf"]["inference_latency_ms_per_sample"] = lf_inference_latency_ms
    run_summary["cost_realism"]["hf"]["inference_latency_ms_per_sample"] = hf_inference_latency_ms
    run_summary["final_accuracy"]["lf_test_acc"] = lf_test_acc
    run_summary["final_accuracy"]["hf_test_acc"] = hf_test_acc

    logger.info("SAVING LATENT REPRESENTATIONS")
    train_latent_folder: str = os.path.join(latent_folder, "train")
    save_latent(
        lf_model=lf_model,
        hf_model=hf_model,
        dataloader=train_dl,
        save_folder=train_latent_folder,
        train_body=train_body
    )

    test_latent_folder: str = os.path.join(latent_folder, "test")
    save_latent(
        lf_model=lf_model,
        hf_model=hf_model,
        dataloader=test_dl,
        save_folder=test_latent_folder,
        train_body=train_body
    )

    val_latent_folder: str = os.path.join(latent_folder, "val")
    save_latent(
        lf_model=lf_model,
        hf_model=hf_model,
        dataloader=val_dl,
        save_folder=val_latent_folder,
        train_body=train_body
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

    train_dl: DataLoader = DataLoader(
        train_ds,
        batch_size=config.fe_training.batch_size,
        shuffle=True
    )

    test_dl: DataLoader = DataLoader(
        test_ds,
        batch_size=config.fe_training.batch_size,
    )

    val_dl: DataLoader = DataLoader(
        val_ds,
        batch_size=config.fe_training.batch_size,
    )

    if dataset_name != "crop":
        fe_model: nn.Module = build_model(
            model_name="mlp",
            input_size=latent_size,
            output_size=2,
            latent_size=128
        )
    else:
        fe_model: nn.Module = build_model(
            model_name="cnn_head",
            input_size=latent_size,
            output_size=2,
            latent_size=None
        )

    fe_param_count = count_parameters(fe_model)
    logger.info(f"Gate/FE model parameters: {fe_param_count:,}")
    run_summary["cost_realism"]["gate"] = {"num_parameters": fe_param_count}

    logger.info("RUNNING DEFAULT FE")
    adaptive_search: AdaptiveGridSearch = AdaptiveGridSearch(
        fe_model=fe_model,
        device=DEVICE,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        model_folder=model_folder
    )

    reset_peak_memory(DEVICE)
    gate_search_start = time.perf_counter()

    usage_values: List[float] = [i/10 for i in range(1, 11)]

    fe_usage_vals: List[float] = []
    fe_acc_vals: List[float] = []

    for usage in usage_values:
        _, test_acc, test_use = adaptive_search.find_r_for_target(
            usage=usage
        )

        fe_usage_vals.append(test_use)
        fe_acc_vals.append(test_acc)

    gate_search_time_sec = time.perf_counter() - gate_search_start
    gate_peak_mem_mb = get_peak_memory_mb(DEVICE)
    logger.info(f"Gate search total time: {format_duration(gate_search_time_sec)}")
    logger.info(f"Gate search phase peak GPU memory: {gate_peak_mem_mb:.1f} MB")

    run_summary["timing"]["gate_search_total_sec"] = gate_search_time_sec
    run_summary["peak_gpu_memory_mb"]["gate_search_phase"] = gate_peak_mem_mb

    adaptive_search.save_evaluated_points(os.path.join(file_folder, "gate_search_log.npz"))
    adaptive_search.save_epoch_log(os.path.join(file_folder, "gate_epoch_metrics.csv"))
    adaptive_search.save_search_diagnostics(os.path.join(file_folder, "gate_search_diagnostics.csv"))

    fe_usage_vals = np.array(fe_usage_vals)
    fe_acc_vals = np.array(fe_acc_vals)

    fe_data_path: str = os.path.join(file_folder, "fe_results.npz")
    np.savez(
        fe_data_path,
        usage=fe_usage_vals,
        acc=fe_acc_vals
    )
    run_summary["result_files"].append("fe_results.npz")

    # ends the script if you don't want comparisons
    if not run_comparisons:
        run_summary["timing"]["total_experiment_sec"] = time.perf_counter() - experiment_start_time
        with open(os.path.join(folder_name, "summary.json"), "w") as f:
            json.dump(run_summary, f, indent=2)
        logger.info("EXPERIMENT FINISHED")
        return

    logger.info("RUNNING DEFAULT SOFTMAX RESPONSE")

    softmax_response: SoftmaxResponseMethod = SoftmaxResponseMethod(
        val_dl=val_dl,
        test_dl=test_dl,
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

    num_baseline_epochs: int = config.classifier_training.epochs
    threshold_grid: List[float] = list(np.linspace(0.0, 1.0, 21))

    logger.info("RUNNING DEFAULT SELECTIVENET")
    selnet_model: nn.Module = build_model(
        model_name=lf_model_name,
        input_size=lf_input_size,
        output_size=output_size,
        latent_size=latent_size
    ).to(DEVICE)

    selnet_param_count = count_parameters(selnet_model)
    logger.info(f"SelectiveNet model parameters: {selnet_param_count:,}")
    run_summary["cost_realism"]["selectivenet"] = {"num_parameters": selnet_param_count}

    selnet_optimizer = torch.optim.Adam(selnet_model.parameters(), lr=1e-3, weight_decay=1e-5)
    selnet_method: SelectiveNetMethod = SelectiveNetMethod(model_folder=model_folder)
    selnet_model_file = os.path.join(model_folder, "selnet_model.pt")
    selnet_best_acc: float = 0.0

    reset_peak_memory(DEVICE)
    selnet_train_start = time.perf_counter()
    selnet_epoch_logger = EpochMetricsLogger(
        os.path.join(file_folder, "selnet_epoch_metrics.csv"),
        ["epoch", "train_loss", "train_acc", "train_usage", "val_loss", "val_acc", "val_usage", "epoch_time_sec"]
    )

    for epoch in range(num_baseline_epochs):
        selnet_epoch_start = time.perf_counter()

        selnet_train_metrics = selnet_method.one_run(
            model=selnet_model,
            dataloader=cascade_train_dl,
            hf_model=hf_model,
            threshold=0.5,
            train_body=train_body,
            optimizer=selnet_optimizer
        )

        selnet_val_metrics = selnet_method.one_run(
            model=selnet_model,
            dataloader=cascade_val_dl,
            hf_model=hf_model,
            threshold=0.5,
            train_body=train_body,
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

        logger.info(f"SelectiveNet Epoch {epoch+1} Summary:")
        logger.info(f"\tTrain Loss: {selnet_train_metrics['loss']:.4f} | Accuracy: {100 * selnet_train_metrics['accuracy']:.2f}% | Usage: {100 * selnet_train_metrics['usage']:.2f}%")
        logger.info(f"\tVal   Loss: {selnet_val_metrics['loss']:.4f}   | Accuracy: {100 * selnet_val_metrics['accuracy']:.2f}%   | Usage: {100 * selnet_val_metrics['usage']:.2f}%")

    selnet_train_time_sec = time.perf_counter() - selnet_train_start
    selnet_peak_mem_mb = get_peak_memory_mb(DEVICE)
    logger.info(f"SelectiveNet train time: {format_duration(selnet_train_time_sec)}")
    logger.info(f"SelectiveNet phase peak GPU memory: {selnet_peak_mem_mb:.1f} MB")

    run_summary["timing"]["selectivenet_train_sec"] = selnet_train_time_sec
    run_summary["peak_gpu_memory_mb"]["selectivenet_phase"] = selnet_peak_mem_mb

    selnet_model.load_state_dict(torch.load(selnet_model_file, weights_only=True))

    selnet_usage_vals: List[float] = []
    selnet_acc_vals: List[float] = []

    for threshold in threshold_grid:
        selnet_test_metrics = selnet_method.one_run(
            model=selnet_model,
            dataloader=cascade_test_dl,
            hf_model=hf_model,
            threshold=float(threshold),
            train_body=train_body,
        )

        selnet_usage_vals.append(100 * selnet_test_metrics["usage"])
        selnet_acc_vals.append(100 * selnet_test_metrics["accuracy"])

    selectivenet_data_path: str = os.path.join(file_folder, "selectivenet_results.npz")
    np.savez(
        selectivenet_data_path,
        usage=np.array(selnet_usage_vals),
        acc=np.array(selnet_acc_vals),
        thresholds=np.array(threshold_grid)
    )
    run_summary["result_files"].append("selectivenet_results.npz")

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

    sat_optimizer = torch.optim.Adam(sat_model.parameters(), lr=1e-3, weight_decay=1e-5)
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
            epoch=epoch
        )

        sat_val_metrics = sat_method.one_run(
            model=sat_model,
            dataloader=sat_val_dl,
            hf_model=hf_model,
            tau=0.5,
            train_body=train_body,
            epoch=epoch
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

    sat_usage_vals: List[float] = []
    sat_acc_vals: List[float] = []

    for tau in threshold_grid:
        sat_test_metrics = sat_method.one_run(
            model=sat_model,
            dataloader=sat_test_dl,
            hf_model=hf_model,
            tau=float(tau),
            train_body=train_body,
            epoch=num_baseline_epochs
        )

        sat_usage_vals.append(100 * sat_test_metrics["usage"])
        sat_acc_vals.append(100 * sat_test_metrics["accuracy"])

    sat_data_path: str = os.path.join(file_folder, "sat_results.npz")
    np.savez(
        sat_data_path,
        usage=np.array(sat_usage_vals),
        acc=np.array(sat_acc_vals),
        thresholds=np.array(threshold_grid)
    )
    run_summary["result_files"].append("sat_results.npz")

    run_summary["timing"]["total_experiment_sec"] = time.perf_counter() - experiment_start_time
    logger.info(f"Total experiment time: {format_duration(run_summary['timing']['total_experiment_sec'])}")
    with open(os.path.join(folder_name, "summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    logger.info("EXPERIMENT FINISHED")

if __name__ == "__main__":
    main()
