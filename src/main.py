import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import JaccardIndex
from typing import *
import json
import click
import os.path as path
import logging
import os

from datasets import *
from models import build_unet, build_resnet, build_mlp, LatentCNNHead, CustomMLP
from helpers import seed_all, load_yaml_options, build_dataset, classifier_one_run, create_fe_dataset, reset_all_weights
from losses import MetaLossFunction

from custom_types import ConfigOptions

@click.command()
@click.option("--config_file", default="../config.yml")
def main(config_file):
    config: ConfigOptions = load_yaml_options(config_file)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed: int = config.random_seed
    latent_size: int = config.latent_size
    dataset: str = config.dataset.name

    hf_aug: str = config.dataset.high_fidelity.aug_name
    hf_aug_str: float = config.dataset.high_fidelity.strength

    lf_aug: str = config.dataset.low_fidelity.aug_name
    lf_aug_str: float = config.dataset.low_fidelity.strength

    folder_name: str = os.path.join("results", f"{dataset}_hf={hf_aug}-lf={lf_aug}-{seed}-{latent_size}")

    model_folder: str = os.path.join(folder_name, "models")
    file_folder: str = os.path.join(folder_name, "files")
    image_folder: str = os.path.join(folder_name, "images")

    for folder in [model_folder, file_folder, image_folder]:
        os.makedirs(folder, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(folder_name, "log.log"),
        filemode='w'
    )

    logger = logging.getLogger(__name__)
    logger.info("SETTINGS")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Folder: {folder}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"\tHF: Augmentation [{hf_aug}] at strength [{hf_aug_str}] (if applicable)")
    logger.info(f"\tLF: Augmentation [{lf_aug}] at strength [{lf_aug_str}] (if applicable)")
    logger.info(f"Latent Size: {latent_size}")
    logger.info(f"Seed: {seed}")

    logger.info(f"SETTING SEED")
    seed_all(seed=seed)

    logger.info(f"BUILDING DATASET & DATALOADERS")
    train_ds, test_ds, val_ds = build_dataset(
        dataset_settings=config.dataset,
        seed=seed
    )

    classifier_batch_size = config.classifier_training.batch_size
    cls_train_loader = DataLoader(train_ds, batch_size=classifier_batch_size, shuffle=True)
    cls_test_loader = DataLoader(test_ds, batch_size=classifier_batch_size, shuffle=False)
    cls_val_loader = DataLoader(val_ds, batch_size=classifier_batch_size, shuffle=False)

    logger.info("BUILDING MODELS")

    if dataset == "toy":
        input_size: int = train_ds.get_input_size()
        num_classes: int = train_ds.get_num_classes()

        lf_model = CustomMLP(
            input_size=input_size,
            num_layers=2,
            output_size=num_classes,
            hidden_size=latent_size,
        )

        hf_model = CustomMLP(
            input_size=input_size*2,
            num_layers=2,
            output_size=num_classes,
            hidden_size=latent_size,
        )

        fe_model = CustomMLP(
            input_size=latent_size,
            num_layers=2,
            output_size=2,
            hidden_size=latent_size,
        )
    
    elif dataset == "mnist":
        num_classes: int = 10

        lf_model = build_resnet(
            latent_size=latent_size,
            output_size=num_classes,
            three_channel=True
        )

        hf_model = build_resnet(
            latent_size=latent_size,
            output_size=num_classes,
            three_channel=False
        )

        fe_model = build_mlp(
            input=latent_size,
            num_layers=5,
            output_size=2,
            hidden_size=latent_size
        )

    elif dataset == "crop":
        num_classes: int = 14

        lf_model = build_unet(
            num_channels=3,
            num_classes=num_classes
        )

        hf_model = build_unet(
            num_channels=6,
            num_classes=num_classes
        )

        fe_model = LatentCNNHead(
            in_channels=1024,
            num_classes=num_classes
        )

    logger.info("STARTING CLASSIFIER TRAINING")
    classifier_loss = config.classifier_training.loss_fun
    classifier_epochs = config.classifier_training.epochs

    logger.info("TRAINING LF MODEL")
    lf_model.to(DEVICE)
    lf_optimizer = torch.optim.Adam(
        lf_model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    lf_scheduler = CosineAnnealingLR(lf_optimizer, T_max=classifier_epochs//2, eta_min=1e-6)

    if classifier_loss == "CE":
        criterion = torch.nn.CrossEntropyLoss()
    elif classifier_loss == "MSE":
        criterion = torch.nn.MSELoss()
    else:
        criterion = JaccardIndex(task='multiclass', num_classes=num_classes)

    lf_fname: str = os.path.join(
        model_folder,
        "lf_model.pt"
    )
    best_val_acc = 0

    for epoch in range(classifier_epochs):
        train_loss, train_acc = classifier_one_run(lf_model, cls_train_loader, criterion, fidelity="lf", optimizer=lf_optimizer, scheduler=lf_scheduler)
        val_loss, val_acc = classifier_one_run(lf_model, cls_val_loader, criterion, fidelity="lf")

        log_msg = f"Epoch {epoch+1}: {train_loss:.4f}, {train_acc*100:.2f}% | {val_loss:.4f}, {val_acc*100:.2f}%"
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            log_msg += " <- BEST"
            torch.save(lf_model.state_dict(), lf_fname)

        logger.info(log_msg)

    lf_model.load_state_dict(torch.load(lf_fname, weights_only=True))

    logger.info("TRAINING HF MODEL")
    hf_model.to(DEVICE)
    hf_optimizer = torch.optim.Adam(
        hf_model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    hf_scheduler = CosineAnnealingLR(hf_optimizer, T_max=classifier_epochs//2, eta_min=1e-6)

    hf_fname: str = os.path.join(
        model_folder,
        "hf_model.pt"
    )

    best_val_acc = 0

    for epoch in range(classifier_epochs):
        train_loss, train_acc = classifier_one_run(hf_model, cls_train_loader, criterion, fidelity="hf", optimizer=hf_optimizer, scheduler=hf_scheduler)
        val_loss, val_acc = classifier_one_run(hf_model, cls_val_loader, criterion, fidelity="hf")

        log_msg = f"Epoch {epoch+1}: {train_loss:.4f}, {train_acc*100:.2f}% | {val_loss:.4f}, {val_acc*100:.2f}%"
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            log_msg += " <- BEST"
            torch.save(hf_model.state_dict(), hf_fname)

        logger.info(log_msg)

    hf_model.load_state_dict(torch.load(hf_fname, weights_only=True))

    logger.info("BUILDING FE DATALOADERS")
    num_reruns = config.num_reruns
    fe_settings = config.fe_training.r_range

    if fe_settings.direction == "normal":
        start = fe_settings.start
        stop = fe_settings.stop
    else:
        start = fe_settings.stop
        stop = fe_settings.start

    if fe_settings.scale == "linear":
        r_range = torch.linspace(
            start=start,
            end=stop,
            steps=fe_settings.num_steps
        )
    else:
        r_range = torch.logspace(
            start=start,
            end=stop,
            steps=fe_settings.num_steps
        )

    fe_train_ds = create_fe_dataset(
        dataloader=cls_train_loader,
        lf_model=lf_model,
        hf_model=hf_model,
        device=DEVICE
    )

    fe_test_ds = create_fe_dataset(
        dataloader=cls_test_loader,
        lf_model=lf_model,
        hf_model=hf_model,
        device=DEVICE
    )

    fe_val_ds = create_fe_dataset(
        dataloader=cls_val_loader,
        lf_model=lf_model,
        hf_model=hf_model,
        device=DEVICE
    )

    fe_batch_size = config.fe_training.batch_size
    fe_train_loader = DataLoader(fe_train_ds, batch_size=fe_batch_size, shuffle=True)
    fe_test_loader = DataLoader(fe_test_ds, batch_size=fe_batch_size, shuffle=False)
    fe_val_loader = DataLoader(fe_val_ds, batch_size=fe_batch_size, shuffle=False)

    fe_epochs = config.fe_training.epochs
    fe_model.to(DEVICE)

    logger.info("STARTING FE TRAINING")

    best_val_acc = torch.zeros((num_reruns, len(r_range)))
    best_val_loss = torch.inf * torch.ones((num_reruns, len(r_range)))

    for rerun in range(num_reruns):
        for i, r_val in enumerate(r_range):
            reset_all_weights(fe_model)
            fe_optimizer = torch.optim.Adam(
                fe_model.parameters(),
                lr=1e-3,
                weight_decay=1e-5
            )

            fe_scheduler = CosineAnnealingLR(fe_optimizer, T_max=fe_epochs//2, eta_min=1e-6)


            criterion = MetaLossFunction(ch=[r_val], cw=1, device=DEVICE)

            logger.info(f"R Value: {r_val}")

            fe_fname: str = os.path.join(
                model_folder,
                f"fe_model-{r_val:.4f}.pt"
            )

            for epoch in range(fe_epochs):
                train_loss, train_acc, train_hc = classifier_one_run(fe_model, fe_train_loader, criterion, fidelity="gate", optimizer=fe_optimizer, scheduler=fe_scheduler)
                val_loss, val_acc, val_hc = classifier_one_run(fe_model, fe_val_loader, criterion, fidelity="gate")

                log_msg = f"Epoch {epoch+1}: {train_loss:.4f}, {train_acc*100:.2f}%, {train_hc*100:.2f}% | {val_loss:.4f}, {val_acc*100:.2f}%, {val_hc*100:.2f}%"

                if val_loss < best_val_loss[rerun, i]:
                    best_val_loss[rerun, i] = val_loss
                    best_val_acc[rerun, i] = val_acc
                    torch.save(fe_model.state_dict(), fe_fname)

                    log_msg += "<- BEST"

                logger.info(log_msg)

    
    # ##### HIGH FIDELITY TRAINING
    # hf_load_location = str(config.stage1.hf_model.load_file)

    # if not path.exists(hf_load_location):
    #     hf_early_stopper = config_early_stop(config)
    #     hf_optimizer = torch.optim.Adam(
    #         hf_model.parameters(),
    #         lr=config.stage1.hf_model.optimizer.lr,
    #         weight_decay=config.stage1.hf_model.optimizer.weight_decay
    #     )
    #     hf_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #         hf_optimizer, 
    #         gamma=config.stage1.hf_model.scheduler.gamma
    #     )

    #     if config.stage1.loss_fun == "CE":
    #         criterion = torch.nn.CrossEntropyLoss()
    #     else:
    #         criterion = torch.nn.MSELoss()
            
    #     classifier_epochs = config.stage1.epochs

    #     hf_metadata = build_metadata(config)

    #     for epoch in range(classifier_epochs):
    #         train_loss, train_acc = classifier_one_run(hf_model, train_loader, criterion, fidelity="hf", optimizer=hf_optimizer, scheduler=hf_scheduler)
    #         test_loss, test_acc = classifier_one_run(hf_model, test_loader, criterion, fidelity="hf", optimizer=hf_optimizer, scheduler=hf_scheduler)
    #         val_loss, val_acc = classifier_one_run(hf_model, val_loader, criterion, fidelity="hf", optimizer=hf_optimizer, scheduler=hf_scheduler)

    #         hf_metadata["loss"]["train"].append(train_loss)
    #         hf_metadata["loss"]["test"].append(test_loss)
    #         hf_metadata["loss"]["val"].append(val_loss)

    #         hf_metadata["acc"]["train"].append(train_acc)
    #         hf_metadata["acc"]["test"].append(test_acc)
    #         hf_metadata["acc"]["val"].append(val_acc)

    #         print(train_acc, val_acc)
    #         if hf_early_stopper.early_stop(val_loss):
    #             break
        
    #     hf_save_location = config.stage1.hf_model.save_location
    #     final_acc = round(val_acc*100, 2)
    #     # final_acc_str = str(final_acc).replace(".", "_")

    #     hf_model_file = f"hf_model_{final_acc}.pt"
    #     hf_metadata_file = f"hf_model_{final_acc}_meta.json"

    #     torch.save(hf_model.state_dict(), path.join(hf_save_location, hf_model_file))
    #     with open(path.join(hf_save_location, hf_metadata_file), "w") as json_file:
    #         json.dump(hf_metadata, json_file, indent=4)


    # ##### LOW FIDELITY TRAINING
    # lf_load_location = str(config.stage1.lf_model.load_file)

    # if not path.exists(lf_load_location):
    #     lf_early_stopper = config_early_stop(config)
    #     lf_optimizer = torch.optim.Adam(
    #         lf_model.parameters(),
    #         lr=config.stage1.lf_model.optimizer.lr,
    #         weight_decay=config.stage1.lf_model.optimizer.weight_decay
    #     )
    #     lf_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #         lf_optimizer, 
    #         gamma=config.stage1.lf_model.scheduler.gamma
    #     )

    #     lf_metadata = {
    #         "loss": {
    #             "train": [],
    #             "test": [],
    #             "val": [],
    #         },
    #         "acc": {
    #             "train": [],
    #             "test": [],
    #             "val": [],
    #         }
    #     }

    #     for epoch in range(classifier_epochs):
    #         train_loss, train_acc = classifier_one_run(lf_model, train_loader, criterion, fidelity="lf", optimizer=lf_optimizer, scheduler=lf_scheduler)
    #         test_loss, test_acc = classifier_one_run(lf_model, test_loader, criterion, fidelity="lf", optimizer=lf_optimizer, scheduler=lf_scheduler)
    #         val_loss, val_acc = classifier_one_run(lf_model, val_loader, criterion, fidelity="lf", optimizer=lf_optimizer, scheduler=lf_scheduler)

    #         lf_metadata["loss"]["train"].append(train_loss)
    #         lf_metadata["loss"]["test"].append(test_loss)
    #         lf_metadata["loss"]["val"].append(val_loss)

    #         lf_metadata["acc"]["train"].append(train_acc)
    #         lf_metadata["acc"]["test"].append(test_acc)
    #         lf_metadata["acc"]["val"].append(val_acc)

    #         print(train_acc, val_acc)
    #         if lf_early_stopper.early_stop(val_loss):
    #             break

    #     lf_save_location = config.stage1.lf_model.save_location
    #     final_acc = round(val_acc*100, 2)

    #     lf_model_file = f"lf_model_{final_acc}.pt"
    #     lf_metadata_file = f"lf_model_{final_acc}_meta.json"

    #     torch.save(lf_model.state_dict(), path.join(lf_save_location, lf_model_file))
    #     with open(path.join(lf_save_location, lf_metadata_file), "w") as json_file:
    #         json.dump(lf_metadata, json_file, indent=4)

if __name__ == "__main__":
    main()


# qe_train_acc = np.zeros((NUM_RERUNS, len(R_VALS)))
# qe_train_use = np.zeros((NUM_RERUNS, len(R_VALS)))

# qe_test_acc = np.zeros((NUM_RERUNS, len(R_VALS)))
# qe_test_use = np.zeros((NUM_RERUNS, len(R_VALS)))

# qe_val_acc = np.zeros((NUM_RERUNS, len(R_VALS)))
# qe_val_use = np.zeros((NUM_RERUNS, len(R_VALS)))

# full_dataset = BinaryHypercubeDataset(4000, noise_level=1.1)

# for run_num in range(NUM_RERUNS):
#     lf_model = torch.nn.Sequential(
#         torch.nn.Linear(2, 64),
#         torch.nn.ReLU(),
#         torch.nn.Linear(64, 128),
#         torch.nn.ReLU(),
#         torch.nn.Linear(128, 64),
#         torch.nn.ReLU(),
#         torch.nn.Linear(64, 32),
#         torch.nn.ReLU(),
#         torch.nn.Linear(32, 2)
#     ).to(DEVICE)

#     lf_optimizer = torch.optim.Adam(lf_model.parameters(), lr=3e-3, weight_decay=1e-5)
#     criterion = torch.nn.CrossEntropyLoss()
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(lf_optimizer, gamma=0.99)
#     early_stopper = EarlyStopper(patience=5)

#     train_loader, test_loader, val_loader = build_dataloaders(full_dataset)


#     for epoch in range(NUM_FIDELITY_EPOCHS):
#         train_loss, train_acc = classifier_one_run(lf_model, train_loader, criterion, fidelity="lf", optimizer=lf_optimizer, scheduler=scheduler)
#         test_loss, test_acc = classifier_one_run(lf_model, test_loader, criterion, fidelity="lf", optimizer=lf_optimizer, scheduler=scheduler)
#         val_loss, val_acc = classifier_one_run(lf_model, val_loader, criterion, fidelity="lf", optimizer=lf_optimizer, scheduler=scheduler)

#         print(train_acc, val_acc)
#         if early_stopper.early_stop(val_loss):
#             break
#         # scheduler.step(

#     print("\n\n\n")

#     hf_model = torch.nn.Sequential(
#         torch.nn.Linear(2, 64),
#         torch.nn.ReLU(),
#         torch.nn.Linear(64, 128),
#         torch.nn.ReLU(),
#         torch.nn.Linear(128, 64),
#         torch.nn.ReLU(),
#         torch.nn.Linear(64, 32),
#         torch.nn.ReLU(),
#         torch.nn.Linear(32, 2)
#     ).to(DEVICE)

#     hf_optimizer = torch.optim.Adam(hf_model.parameters(), lr=3e-3, weight_decay=1e-5)
#     criterion = torch.nn.CrossEntropyLoss()
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(hf_optimizer, gamma=0.99)
#     early_stopper = EarlyStopper(patience=5)

#     for epoch in range(NUM_FIDELITY_EPOCHS):
#         train_loss, train_acc = classifier_one_run(hf_model, train_loader, criterion, fidelity="hf", optimizer=hf_optimizer, scheduler=scheduler)
#         test_loss, test_acc = classifier_one_run(hf_model, test_loader, criterion, fidelity="hf", optimizer=hf_optimizer, scheduler=scheduler)
#         val_loss, val_acc = classifier_one_run(hf_model, val_loader, criterion, fidelity="hf", optimizer=hf_optimizer, scheduler=scheduler)

#         print(train_acc, val_acc)
#         if early_stopper.early_stop(val_loss):
#             break

#     svm_model = SVC(kernel="rbf", class_weight="balanced")
#     fe_svm_one_run(svm_model, hf_model, lf_model, train_loader)

    # qe_train_loader, qe_test_loader, qe_val_loader = build_qe_dataloaders(lf_model, hf_model, train_loader, test_loader, val_loader)

    # dataloaders = {
    #     "train": qe_train_loader,
    #     "test": qe_test_loader,
    #     "val": qe_val_loader
    # }

    # test_positions = qe_test_loader.inputs

    # svm_model = None
    # for r, r_val in enumerate(R_VALS):
    #     print(f"r: {r_val} ({round((r+1)/len(R_VALS)*100, 2)}%)")
    #     for loader_name in dataloaders.keys():
    #         loader = dataloaders[loader_name]

    #         lf_embeds = loader.dataset.lf_embeddings
    #         lf_preds = loader.dataset.lf_preds
    #         hf_preds = loader.dataset.hf_preds
    #         labels = loader.dataset.labels

    #         lf_correct = np.argmax(lf_preds, axis=1) == labels
    #         hf_correct = np.argmax(hf_preds, axis=1) == labels

    #         best_choices = np.logical_and(~lf_correct, hf_correct).astype(int)
        
    #         if loader_name == "train":
    #             weights = np.where(best_choices==1, r_val, 1)
    #             svm_model = SVC(kernel="rbf", class_weight="balanced")
    #             svm_model.fit(lf_embeds, best_choices, sample_weight=weights)

    #         outputs = svm_model.predict(lf_embeds)
    #         acc = np.sum(lf_correct[outputs==0]) + np.sum(hf_correct[outputs==1]) / len(outputs)
    #         usage = np.sum(outputs) / len(outputs)

    #         if loader_name == "train":
    #             qe_train_acc[run_num, r] = acc
    #             qe_train_use[run_num, r] = usage

    #         elif loader_name == "test":
    #             qe_test_acc[run_num, r] = acc
    #             qe_test_use[run_num, r] = usage

    #         else:
    #             qe_val_acc[run_num, r] = acc
    #             qe_val_use[run_num, r] = usage

    #         print(f"\tAcc: {round(acc, 2) * 100}%", end="\t\t")
    #         print(f"\tUse: {round(usage, 2) * 100}%")
