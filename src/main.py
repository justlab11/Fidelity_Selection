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

from datasets import *
from models import build_unet, build_resnet, CustomResNet18, LatentCNNHead, CustomMLP
from helpers import *
from comparisons import *
from losses import MetaLossFunction

from custom_types import ConfigOptions

@click.command()
@click.option("--config_file", default="../config.yml")
def main(config_file):
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
    shutil.copyfile(config_file, file_folder)

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

    logger.info(f"\nSETTING SEED")
    set_all_seeds(seed=seed)

    logger.info(f"\nBUILDING DATASET & DATALOADERS")
    train_ds, test_ds, val_ds = build_dataset(
        dataset_name=dataset_name,
        seed=seed,
        folder="../data"
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

    logger.info(f"\nBUILDING MODELS")
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
        input_size=hf_input_size,
        output_size=output_size,
        latent_size=latent_size
    )

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

        train_ds: BodyDataset = BodyDataset(
            folder_path=train_body_folder
        )

        test_ds: BodyDataset = BodyDataset(
            folder_path=test_body_folder
        )

        val_ds: BodyDataset = BodyDataset(
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

    logger.info(f"\nTRAINING HF AND LF MODELS")
    lf_model: nn.Module = lf_model.to(DEVICE)
    hf_model: nn.Module = hf_model.to(DEVICE)

    lf_optimizer = torch.optim.Adam(lf_model.parameters(), lr=1e-3, weight_decay=1e-5)
    hf_optimizer = torch.optim.Adam(hf_model.parameters(), lr=1e-3, weight_decay=1e-5)

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
        trained_lf_path = os.abspath(trained_lf_path)
        try:
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
        trained_hf_path = os.abspath(trained_hf_path)
        try:
            hf_model.load_state_dict(
                torch.load(
                    trained_hf_path, 
                    weights_only=True
            ))
            train_hf_model = False
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

    for epoch in range(num_classifier_epochs):
        logger.info(f"Epoch {epoch+1} Summary:")
        if train_lf_model:
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

            if lf_val_acc > lf_best_acc:
                lf_best_acc = lf_val_acc
                lf_state_dict = lf_model.state_dict()
                torch.save(lf_state_dict, lf_model_file)

            logger.info(f"\tLF Train  Loss: {lf_train_loss:.4f} | Accuracy: {100 * lf_train_acc:.2f}%")
            logger.info(f"\tLF Val    Loss: {lf_val_loss:.4f}   | Accuracy: {100 * lf_val_acc:.2f}%")

        if train_hf_model:
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

            if hf_val_acc > hf_best_acc:
                hf_best_acc = hf_val_acc
                hf_state_dict = hf_model.state_dict()
                torch.save(hf_state_dict, hf_model_file)

            logger.info(f"\tHF Train  Loss: {hf_train_loss:.4f} | Accuracy: {100 * hf_train_acc:.2f}%")
            logger.info(f"\tHF Val    Loss: {hf_val_loss:.4f}   | Accuracy: {100 * hf_val_acc:.2f}%")

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

    logger.info("\nSAVING LATENT REPRESENTATIONS")
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

    logger.info("\nRUNNING DEFAULT FE")
    adaptive_search: AdaptiveGridSearch = AdaptiveGridSearch(
        fe_model=fe_model,
        device=DEVICE,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        model_folder=model_folder
    )

    usage_values: List[float] = [i/10 for i in range(1, 11)]

    for usage in usage_values:
        fe_usage_vals: float = []
        fe_acc_vals: float = []

        _, test_acc, test_use = adaptive_search.find_r_for_target(
            usage=usage
        )

        fe_usage_vals.append(test_use)
        fe_acc_vals.append(test_acc)

    fe_usage_vals = np.array(fe_usage_vals)
    fe_acc_vals = np.array(fe_acc_vals)

    fe_data_path: str = os.path.join(file_folder, "fe_results.npz") 
    np.savez(
        fe_data_path,
        usage=fe_usage_vals,
        acc=fe_acc_vals
    )

    # ends the script if you don't want comparisons
    if not run_comparisons:
        logger.info("\nEXPERIMENT FINISHED")
        return
    
    logger.info("\nRUNNING DEFAULT SOFTMAX RESPONSE")

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

    logger.info("\nRUNNING DEFAULT SELECTIVENET")
    selnet_model: nn.Module = build_model(
        model_name=lf_model_name,
        input_size=lf_input_size,
        output_size=output_size,
        latent_size=latent_size
    )    

    
