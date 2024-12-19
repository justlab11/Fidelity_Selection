import sys
import os.path as path
import os
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from typing import *
import json
import click
from torchvision.models.feature_extraction import create_feature_extractor

from dataset import HypercubeDataset, FidelityDataset
from loss import FidelityEvaluationLoss
from utils import load_yaml_options, build_mlp, classifier_one_run, config_early_stop, generate_r_range

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

    if not path.exists(lf_save_location):
        os.makedirs(lf_save_location)

    if not path.exists(hf_save_location):
        os.makedirs(hf_save_location)

    results_save_location: str = config.stage1.results_save_location

    lf_model_save_name: str = f"cluster_{input_size}_{output_size}_{latent_size}-{lf_dataset_augmentation.augmentation}_{lf_dataset_augmentation.strength}_{lf_dataset_augmentation.steps}"
    lf_model_save_name = path.join(lf_save_location, lf_model_save_name)

    hf_model_save_name: str = f"cluster_{input_size}_{output_size}_{latent_size}-{hf_dataset_augmentation.augmentation}_{hf_dataset_augmentation.strength}_{hf_dataset_augmentation.steps}"
    hf_model_save_name = path.join(hf_save_location, hf_model_save_name)

    print("Training LF Model...")

    if not path.exists(lf_model_save_name+".pt"):
        lf_early_stopper = config_early_stop(config.stage1.early_stop)
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
        best_val_loss = 1000

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
            if (val_loss < best_val_loss):
                best_val_loss = val_loss
                torch.save(lf_model.state_dict(), lf_model_save_name+".pt")

            if lf_early_stopper.early_stop(val_loss):
                break

        # with open(lf_model_save_name+"_meta.json", "w") as json_file:
        #     json.dump(lf_metadata, json_file, indent=4)
    
    else:
        lf_model.load_state_dict(
            torch.load(lf_model_save_name+".pt", weights_only=True)
        )

    _, lf_acc = classifier_one_run(lf_model, val_loader, torch.nn.CrossEntropyLoss(), fidelity="lf")
    print(f"Final LF Accuracy: {lf_acc*100}%")

    print("Training HF Model...")

    if not path.exists(hf_model_save_name+".pt"):
        hf_early_stopper = config_early_stop(config.stage1.early_stop)
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
        best_val_loss = 1000

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
            if (val_loss < best_val_loss):
                best_val_loss = val_loss
                torch.save(hf_model.state_dict(), hf_model_save_name+".pt")

            if hf_early_stopper.early_stop(val_loss):
                break

        # with open(lf_model_save_name+"_meta.json", "w") as json_file:
        #     json.dump(lf_metadata, json_file, indent=4)

    else:
        hf_model.load_state_dict(
            torch.load(hf_model_save_name+".pt", weights_only=True)
        )

    _, hf_acc = classifier_one_run(hf_model, val_loader, torch.nn.CrossEntropyLoss(), fidelity="hf")
    print(f"Final HF Accuracy: {hf_acc*100}%")

    # fe model training

    num_epochs: int = config.stage2.epochs
    fe_save_location: str = config.stage2.fe_save_location
    results_save_location: str = config.stage2.results_save_location
    batch_size: int = config.stage2.batch_size
    num_reruns: int = config.parameters.num_reruns

    return_nodes = {"3": 'output'}
    lf_latent_space_model = create_feature_extractor(
        lf_model,
        return_nodes=return_nodes
    )

    fe_train_set = FidelityDataset(
        lf_latent_model=lf_latent_space_model,
        lf_model = lf_model,
        hf_model = hf_model,
        dataset = train_set
    )
    fe_test_set = FidelityDataset(
        lf_latent_model=lf_latent_space_model,
        lf_model = lf_model,
        hf_model = hf_model,
        dataset = test_set
    )
    fe_val_set = FidelityDataset(
        lf_latent_model=lf_latent_space_model,
        lf_model = lf_model,
        hf_model = hf_model,
        dataset = val_set
    )

    fe_train_loader: DataLoader = DataLoader(fe_train_set, batch_size=batch_size, shuffle=True)
    fe_test_loader: DataLoader = DataLoader(fe_test_set, batch_size=batch_size)
    fe_val_loader: DataLoader = DataLoader(fe_val_set, batch_size=batch_size)

    print(f"Build r in range: {config.stage2.r_range.start}-{config.stage2.r_range.stop}")
    r_values = generate_r_range(
        start = config.stage2.r_range.start,
        stop = config.stage2.r_range.stop,
        num_steps = config.stage2.r_range.num_steps,
        scale = config.stage2.r_range.scale,
        direction = config.stage2.r_range.direction
    )

    r_value_ranges = {}

    for n in range(num_reruns):
        for r_val in r_values:
            r_val_str = str(r_val)

            print(f"{n} - R Value: {round(r_val, 5)}")

            if n == 0:
                r_value_ranges[r_val_str] = [[0, 0, 0] for _ in range(num_reruns)]

            fe_model_save_name = f"fe_model_{round(r_val, 5)}"
            fe_model_save_name = path.join(fe_save_location, fe_model_save_name)

            fe_model = build_mlp(
                input_size=latent_size,
                num_layers=4,
                output_size=2,
                hidden_size=latent_size
            )

            fe_optimizer = torch.optim.Adam(
                fe_model.parameters(),
                lr=config.stage2.fe_model.optimizer.lr,
                weight_decay=config.stage2.fe_model.optimizer.weight_decay
            )

            criterion = FidelityEvaluationLoss(
                num_classes=output_size,
                r=r_val
            )

            fe_early_stop = config_early_stop(config.stage2.fe_model.early_stop)
            best_val_loss = 1000

            for epoch in range(num_epochs):
                train_loss, train_acc, train_use = classifier_one_run(fe_model, fe_train_loader, criterion, fidelity="gate", optimizer=fe_optimizer)
                test_loss, test_acc, test_use = classifier_one_run(fe_model, fe_test_loader, criterion, fidelity="gate")
                val_loss, val_acc, val_use = classifier_one_run(fe_model, fe_val_loader, criterion, fidelity="gate")

                # hf_metadata["loss"]["train"].append(train_loss)
                # hf_metadata["loss"]["test"].append(test_loss)
                # hf_metadata["loss"]["val"].append(val_loss)

                # hf_metadata["acc"]["train"].append(train_acc)
                # hf_metadata["acc"]["test"].append(test_acc)
                # hf_metadata["acc"]["val"].append(val_acc)

                if (val_loss < best_val_loss):
                    best_val_loss = val_loss
                    # print(val_loss, round(train_acc, 4)*100, round(val_acc, 4)*100, round(val_use, 4)*100)
                    r_value_ranges[r_val_str][n] = [val_loss, val_acc, val_use]
                    # torch.save(fe_model.state_dict(), fe_model_save_name+".pt")

                if fe_early_stop.early_stop(val_loss):
                    break
        
            print(r_value_ranges[r_val_str][n])

        with open(path.join(results_save_location, "fe_metadata_toy4.json"), "w") as json_file:
            json.dump(r_value_ranges, json_file, indent=4)

    
if __name__ == "__main__":
    main()