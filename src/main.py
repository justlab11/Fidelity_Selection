import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from typing import *
from sklearn.svm import SVC

from datasets import BinaryHypercubeDataset
from helpers import *


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

R_VALS = np.linspace(0.01, 3, 200)
NUM_RERUNS = 1
NUM_FIDELITY_EPOCHS = 50
NUM_QE_EPOCHS = 200

qe_train_acc = np.zeros((NUM_RERUNS, len(R_VALS)))
qe_train_use = np.zeros((NUM_RERUNS, len(R_VALS)))

qe_test_acc = np.zeros((NUM_RERUNS, len(R_VALS)))
qe_test_use = np.zeros((NUM_RERUNS, len(R_VALS)))

qe_val_acc = np.zeros((NUM_RERUNS, len(R_VALS)))
qe_val_use = np.zeros((NUM_RERUNS, len(R_VALS)))

full_dataset = BinaryHypercubeDataset(4000, noise_level=1.1)

for run_num in range(NUM_RERUNS):
    lf_model = torch.nn.Sequential(
        torch.nn.Linear(2, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2)
    ).to(DEVICE)

    lf_optimizer = torch.optim.Adam(lf_model.parameters(), lr=3e-3, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(lf_optimizer, gamma=0.99)
    early_stopper = EarlyStopper(patience=5)

    train_loader, test_loader, val_loader = build_dataloaders(full_dataset)


    for epoch in range(NUM_FIDELITY_EPOCHS):
        train_loss, train_acc = classifier_one_run(lf_model, train_loader, criterion, fidelity="lf", optimizer=lf_optimizer, scheduler=scheduler)
        test_loss, test_acc = classifier_one_run(lf_model, test_loader, criterion, fidelity="lf", optimizer=lf_optimizer, scheduler=scheduler)
        val_loss, val_acc = classifier_one_run(lf_model, val_loader, criterion, fidelity="lf", optimizer=lf_optimizer, scheduler=scheduler)

        print(train_acc, val_acc)
        if early_stopper.early_stop(val_loss):
            break
        # scheduler.step(

    print("\n\n\n")

    hf_model = torch.nn.Sequential(
        torch.nn.Linear(2, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2)
    ).to(DEVICE)

    hf_optimizer = torch.optim.Adam(hf_model.parameters(), lr=3e-3, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(hf_optimizer, gamma=0.99)
    early_stopper = EarlyStopper(patience=5)

    for epoch in range(NUM_FIDELITY_EPOCHS):
        train_loss, train_acc = classifier_one_run(hf_model, train_loader, criterion, fidelity="hf", optimizer=hf_optimizer, scheduler=scheduler)
        test_loss, test_acc = classifier_one_run(hf_model, test_loader, criterion, fidelity="hf", optimizer=hf_optimizer, scheduler=scheduler)
        val_loss, val_acc = classifier_one_run(hf_model, val_loader, criterion, fidelity="hf", optimizer=hf_optimizer, scheduler=scheduler)

        print(train_acc, val_acc)
        if early_stopper.early_stop(val_loss):
            break

    svm_model = SVC(kernel="rbf", class_weight="balanced")
    fe_svm_one_run(svm_model, hf_model, lf_model, train_loader)

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
