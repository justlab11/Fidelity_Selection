import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.models.feature_extraction import create_feature_extractor
from models import build_mlp, build_resnet
from custom_types import Options
import yaml
from os import path


def classifier_one_run(model, dataloader, criterion, fidelity, optimizer=None, scheduler=None):
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

class Builder:
    def __init__(self, config: Options, device: torch.device):
        self.config = config
        self.device = device

        self.input_sizes = {
            "toy": self.config.dataset.toy_dataset_parameters.num_dims,
            "mnist": None,
            "cifar10": None,
            "cifar100": None,
        }

        self.output_sizes = {
            "toy": self.config.dataset.toy_dataset_parameters.num_classes,
            "mnist": 10,
            "cifar10": 10,
            "cifar100": 100,
        }

    def __build_model__(self, model_type, dataset_name, pretrained=None):
        input_size = self.input_sizes[dataset_name]
        output_size = self.output_sizes[dataset_name]
        latent_size = self.config.parameters.latent_representation_size

        if "resnet" in model_type:
            model = build_resnet(
                resnet_size=model_type,
                latent_size=latent_size,
                output_size=output_size,
                pretrained=pretrained,
                device=self.device
            )
        
        else:
            num_layers = self.config.stage1.hf_model.classifier.num_layers

            model = build_mlp(
                input_size=input_size,
                num_layers=num_layers,
                output_size=output_size
            )

        return model

    def build_classifiers(self):
        dataset_name = self.config.dataset.name

        hf_model_type = self.config.stage1.hf_model.classifier.type
        lf_model_type = self.config.stage1.lf_model.classifier.type

        if dataset_name == "toy" and hf_model_type != "mlp":
            raise ValueError("stage1.hf_model.classifier.type must be 'mlp' with the 'toy' dataset")
        
        if dataset_name == "toy" and lf_model_type != "mlp":
            raise ValueError("stage1.lf_model.classifier.type must be 'mlp' with the 'toy' dataset")
        
        hf_pretrained = self.config.stage1.hf_model.classifier.pretrained
        lf_pretrained = self.config.stage1.lf_model.classifier.pretrained

        hf_model = self.__build_model__(
            model_type=hf_model_type,
            dataset_name=dataset_name,
            pretrained=hf_pretrained
        )

        lf_model = self.__build_model__(
            model_type=lf_model_type,
            dataset_name=dataset_name,
            pretrained=lf_pretrained
        )

        hf_model_path = str(self.config.stage1.hf_model.load_file)
        if path.exists(hf_model_path):
            torch.load(hf_model_path)

        lf_model_path = str(self.config.stage1.lf_model.load_file)
        if path.exists(lf_model_path):
            torch.load(lf_model_path)




def load_yaml_options(config_file: str) -> Options:
    with open(config_file, 'r') as file:
        yaml_data = yaml.safe_load(file)

    return Options.model_validate(yaml_data)
    



def fe_nn_one_run(fe_model, hf_model, lf_model, dataloader, criterion, optimizer=None, scheduler=None):
    # device = next(hf_model.parameters()).device

    # lf_body = create_feature_extractor(
    #     lf_model, {"7": "body"}
    # ).to(device)

    # for hf_data, lf_data, target in dataloader:
    #     target = target.type(torch.LongTensor)
    #     hf_data = hf_data.to(device)
    #     lf_data = lf_data.to(device)
    #     target = target.to(device)

    #     lf_embeddings = lf_body(lf_data)
    #     lf_output = lf_model(lf_data)
    #     hf_output = hf_model(hf_data)

    #     fe_output = fe_model(lf_embeddings)
        
    #     loss = criterion(target, preds, outputs)
    #     if optimizer:
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    pass

def fe_svm_one_run(fe_model, hf_model, lf_model, dataloader, hf_weight, mode="train"):
    device = next(hf_model.parameters()).device

    lf_body = create_feature_extractor(
        lf_model, {"7": "body"}
    ).to(device)

    num_samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    lf_embeddings = np.zeros((num_samples, 32))
    lf_preds = np.zeros((num_samples, 2))
    hf_preds = np.zeros((num_samples, 2))
    labels = np.zeros(num_samples)

    for i, (hf_data, lf_data, target) in enumerate(dataloader):
        target = target.type(torch.LongTensor) 
        hf_data = hf_data.to(device, torch.float)
        lf_data = lf_data.to(device, torch.float)
        target = target.to(device)

        lf_embs_tmp = lf_body(lf_data)["body"].detach().cpu().numpy()
        lf_output_tmp = lf_model(lf_data).detach().cpu().numpy()
        hf_output_tmp = hf_model(hf_data).detach().cpu().numpy()

        offset = len(lf_embs_tmp)

        lf_embeddings[i*batch_size:(i*batch_size+offset)] = lf_embs_tmp
        lf_preds[i*batch_size:(i*batch_size+offset)] = lf_output_tmp
        hf_preds[i*batch_size:(i*batch_size+offset)] = hf_output_tmp
        labels[i*batch_size:(i*batch_size+offset)] = target.cpu().numpy()

    lf_correct = np.argmax(lf_preds, axis=1) == labels
    hf_correct = np.argmax(hf_preds, axis=1) == labels

    best_choices = np.logical_and(~lf_correct, hf_correct).astype(int)

    if mode=="train":
        weights = np.where(best_choices==1, hf_weight, 1)
        fe_model.fit(lf_embeddings, labels, sample_weights=weights)
        


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