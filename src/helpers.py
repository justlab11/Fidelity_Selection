import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset


def one_run(model, dataloader, criterion, fidelity, optimizer=None, scheduler=None):
    """
    Perform one run through the DataLoader.

    Args:
    - model (torch.nn.Module): The neural network model.
    - dataloader (DataLoader): The DataLoader for iterating over the dataset.
    - criterion (callable): The loss function.
    - optimizer (torch.optim.Optimizer, optional): The optimizer for updating model weights.

    Returns:
    - float: Average loss over the run.
    - int: Correct predictions for calculating accuracy if needed.
    """
    if fidelity not in ["lf", "hf", "gate"]:
        return

    model.train(mode=bool(optimizer))
    device = next(model.parameters()).device
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_high = 0
    if fidelity == "lf":
        for data, _, target in dataloader:
            target = target.type(torch.LongTensor)
            data, target = data.to(device, torch.float), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            predicted = torch.argmax(output, dim=1)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)
        if scheduler:
            scheduler.step()

    elif fidelity == "hf":
        for _, data, target in dataloader:
            target = target.type(torch.LongTensor)
            data, target = data.to(device, torch.float), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            predicted = torch.argmax(output, dim=1)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)
        if scheduler:
            scheduler.step()

    elif fidelity == "gate":
        for lf_embeddings, lf_preds, hf_preds, target in dataloader:
            lf_embeddings = lf_embeddings.to(device, torch.float)
            lf_preds = lf_preds.to(device, torch.float)
            hf_preds = hf_preds.to(device, torch.float)

            target = target.type(torch.LongTensor)
            target = target.to(device)

            outputs = model(lf_embeddings)
            preds = torch.stack([
                lf_preds,
                hf_preds
            ])

            loss = criterion(target, preds, outputs)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            choices = torch.argmax(outputs, dim=1)
            lf_acc = torch.argmax(lf_preds, dim=1) == target
            hf_acc = torch.argmax(hf_preds, dim=1) == target

            total_loss += loss.item() * lf_preds.size(0)
            total_correct += torch.sum(hf_acc[choices==1]) + torch.sum(lf_acc[choices==0]).item()
            total_samples += lf_preds.size(0)
            total_high += torch.sum(choices).item()
        if scheduler:
            scheduler.step()

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    if fidelity=="gate":
        high_count = total_high / total_samples
        return average_loss, accuracy.item(), high_count

    return average_loss, accuracy

def build_qe_model(num_classes: int=2):
    qe_model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
        nn.Softmax(dim=1)
    )

    return qe_model


def build_dataloaders(full_dataset, noise=1):
    # train_dataset = BinaryHypercubeDataset(1275, noise_level=noise)
    # test_dataset = BinaryHypercubeDataset(150, noise_level=noise)
    # val_dataset = BinaryHypercubeDataset(75, noise_level=noise)

    rand_idxs = np.random.choice(3, p=[.85, .1, .05], size=len(full_dataset))
    idxs = np.arange(len(full_dataset))

    train_dataset = Subset(full_dataset, idxs[rand_idxs==0])

    test_dataset = Subset(full_dataset, idxs[rand_idxs==1])
    val_dataset = Subset(full_dataset, idxs[rand_idxs==2])

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    return train_loader, test_loader, val_loader

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False