import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from typing import *
from sklearn.svm import SVC
import json
import click
import os.path as path
from torchvision.models.feature_extraction import create_feature_extractor

import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from dataset import HypercubeDataset
from utils import load_yaml_options, build_mlp, classifier_one_run, config_early_stop

from custom_types import ClusterClassificationConfig, Augmentation

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option("--config_file", default="cluster_classification/config.yml")
def main(config_file):
    config: ClusterClassificationConfig = load_yaml_options(config_file, dataset="cluster")

    dataset: HypercubeDataset = HypercubeDataset(
        config=config,
    )

    train_set: Dataset = dataset.train()
    test_set: Dataset = dataset.test()
    val_set: Dataset = dataset.val()

    batch_size: int = config.stage1.batch_size
    train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader: DataLoader = DataLoader(test_set, batch_size=batch_size)
    val_loader: DataLoader = DataLoader(val_set, batch_size=batch_size)

    input_size: int = config.dataset.settings.num_dims
    output_size: int = config.dataset.settings.num_classes
    latent_size: int = config.parameters.latent_representation_size

    hf_dataset_augmentation: Augmentation = config.dataset.augmentations["high_fidelity"]
    lf_dataset_augmentation: Augmentation = config.dataset.augmentations["low_fidelity"]

    lf_model = build_mlp(
        input_size=input_size,
        num_layers=2,
        output_size=output_size,
        hidden_size=latent_size
    )

    hf_model = build_mlp(
        input_size=input_size,
        num_layers=2,
        output_size=output_size,
        hidden_size=latent_size
    )

    # start training classifiers
    num_epochs: int = config.stage1.epochs
    lf_save_location: str = config.stage1.lf_save_location
    hf_save_location: str = config.stage1.hf_save_location
    results_save_location: str = config.stage1.results_save_location

    lf_model_save_name: str = f"cluster_{input_size}_{output_size}_{latent_size}-{lf_dataset_augmentation.augmentation}_{lf_dataset_augmentation.strength}_{lf_dataset_augmentation.steps}"
    lf_model_save_name = path.join(lf_save_location, lf_model_save_name)

    hf_model_save_name: str = f"cluster_{input_size}_{output_size}_{latent_size}-{hf_dataset_augmentation.augmentation}_{hf_dataset_augmentation.strength}_{hf_dataset_augmentation.steps}"
    hf_model_save_name = path.join(hf_save_location, hf_model_save_name)

    print("Training LF Model...")

    if not path.exists(lf_model_save_name+".pt"):
        lf_early_stopper = config_early_stop(config)
        lf_optimizer = torch.optim.Adam(
            lf_model.parameters(),
            lr=3e-3,
            weight_decay=1e-5
        )

        if config.stage1.loss_fun == "CE":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()

        # lf_metadata = build_metadata(config)

        for epoch in range(num_epochs):
            train_loss, train_acc = classifier_one_run(lf_model, train_loader, criterion, fidelity="lf", optimizer=lf_optimizer)
            test_loss, test_acc = classifier_one_run(lf_model, test_loader, criterion, fidelity="lf")
            val_loss, val_acc = classifier_one_run(lf_model, val_loader, criterion, fidelity="lf")

            # hf_metadata["loss"]["train"].append(train_loss)
            # hf_metadata["loss"]["test"].append(test_loss)
            # hf_metadata["loss"]["val"].append(val_loss)

            # hf_metadata["acc"]["train"].append(train_acc)
            # hf_metadata["acc"]["test"].append(test_acc)
            # hf_metadata["acc"]["val"].append(val_acc)

            print(train_acc, val_acc)
            if lf_early_stopper.early_stop(val_loss):
                break
        
        final_acc = round(val_acc*100, 2)

        torch.save(lf_model.state_dict(), lf_model_save_name+".pt")
        # with open(lf_model_save_name+"_meta.json", "w") as json_file:
        #     json.dump(lf_metadata, json_file, indent=4)
    
    else:
        lf_model.load_state_dict(
            torch.load(lf_model_save_name+".pt", weights_only=True)
        )

    print("Training HF Model...")

    if not path.exists(hf_model_save_name+".pt"):
        hf_early_stopper = config_early_stop(config)
        hf_optimizer = torch.optim.Adam(
            hf_model.parameters(),
            lr=3e-3,
            weight_decay=1e-5
        )

        if config.stage1.loss_fun == "CE":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()

        # lf_metadata = build_metadata(config)

        for epoch in range(num_epochs):
            train_loss, train_acc = classifier_one_run(hf_model, train_loader, criterion, fidelity="hf", optimizer=hf_optimizer)
            test_loss, test_acc = classifier_one_run(hf_model, test_loader, criterion, fidelity="hf")
            val_loss, val_acc = classifier_one_run(hf_model, val_loader, criterion, fidelity="hf")

            # hf_metadata["loss"]["train"].append(train_loss)
            # hf_metadata["loss"]["test"].append(test_loss)
            # hf_metadata["loss"]["val"].append(val_loss)

            # hf_metadata["acc"]["train"].append(train_acc)
            # hf_metadata["acc"]["test"].append(test_acc)
            # hf_metadata["acc"]["val"].append(val_acc)

            print(train_acc, val_acc)
            if hf_early_stopper.early_stop(val_loss):
                break
        
        final_acc = round(val_acc*100, 2)

        torch.save(hf_model.state_dict(), hf_model_save_name+".pt")
        # with open(lf_model_save_name+"_meta.json", "w") as json_file:
        #     json.dump(lf_metadata, json_file, indent=4)

    else:
        hf_model.load_state_dict(
            torch.load(hf_model_save_name+".pt", weights_only=True)
        )

    # fe model training

    num_epochs: int = config.stage2.epochs
    fe_save_location: str = config.stage2.fe_save_location
    results_save_location: str = config.stage2.results_save_location
    batch_size: int = config.stage2.batch_size

    fe_model = build_mlp(
        input_size=latent_size,
        num_layers=4,
        output_size=2,
        hidden_size=latent_size
    )

    second_to_last_layer_index = 2 * 4 - 2  # Multiply by 2 because of ReLU layers
    second_to_last_layer_name = str(second_to_last_layer_index)

    return_nodes = {second_to_last_layer_name: 'output'}
    lf_latent_space_model = create_feature_extractor(
        lf_model,
        return_nodes=return_nodes
    )

    for epoch in range(num_epochs):
        

if __name__ == "__main__":
    main()