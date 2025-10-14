import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms as transforms
import yaml
from os import path
import logging

from datasets import HypercubeDataset, MNISTDataset, CropDataset, CUBDataset, FE_Dataset
from custom_types import ConfigOptions, DatasetSettings
from models import CustomMLP, CustomResNet18, CustomViT, build_unet, LatentCNNHead

logger = logging.getLogger(__name__)

def load_yaml_options(config_file: str) -> ConfigOptions:
    with open(config_file, 'r') as file:
        yaml_data = yaml.safe_load(file)

    return ConfigOptions(**yaml_data)

def set_all_seeds(seed=42):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch current GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch all GPUs (if using multi-GPU)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables benchmark for reproducibility

    return True

class AddGaussianNoise(object):
    def __init__(self, std):
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std
    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"

def build_mnist_transform(aug_name, aug_strength):
    aug_transforms = [transforms.ToTensor()]  # Always start with ToTensor

    if aug_name == "rotation":
        aug_transforms.append(transforms.RandomRotation(degrees=aug_strength))
    elif aug_name == "noise":
        aug_transforms.append(AddGaussianNoise(std=aug_strength))
    # else: no additional augmentation

    # Ensure image is 3-channel for ResNet
    def to_rgb(x):
        return x.expand(3, -1, -1) if x.shape[0] == 1 else x
    aug_transforms.append(transforms.Lambda(to_rgb))

    # Resize and crop to match ResNet-18 input
    aug_transforms += [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    return transforms.Compose(aug_transforms)

def build_dataset(dataset_name: str, seed: int, folder="../data"):
    match dataset_name:
        # toy example where 2d points are split into 4 clusters
        # lf = noisy; hf = clean
        case "toy_2d":
            num_dims = 2
            num_clusters = 2 ** num_dims
            hf_std = np.random.uniform(0.25, 0.4, size=num_clusters)
            lf_std = np.random.uniform(0.4, 0.6, size=num_clusters)

            total_num_samples = 200
            train_samples = int(total_num_samples*.8)
            test_samples = int(total_num_samples*.1)
            val_samples = int(total_num_samples*.1)

            train_samples_per_cluster = np.full(num_clusters, train_samples // num_clusters)
            test_samples_per_cluster = np.full(num_clusters, test_samples // num_clusters)
            val_samples_per_cluster = np.full(num_clusters, val_samples // num_clusters)

            train_ds = HypercubeDataset(
                num_dims=num_dims,
                num_samples=train_samples_per_cluster,
                hf_std=hf_std,
                lf_std=lf_std,
            )

            test_ds = HypercubeDataset(
                num_dims=num_dims,
                num_samples=test_samples_per_cluster,
                hf_std=hf_std,
                lf_std=lf_std,
            )

            val_ds = HypercubeDataset(
                num_dims=num_dims,
                num_samples=val_samples_per_cluster,
                hf_std=hf_std,
                lf_std=lf_std,
            )
        
        # toy example where 5d points are split into 10 clusters
        # lf = noisy; hf = clean
        case "toy_5d":
            num_dims = 5
            num_clusters = 2 ** num_dims
            hf_std = np.random.uniform(0.25, 0.4, size=num_clusters)
            lf_std = np.random.uniform(0.4, 0.6, size=num_clusters)

            total_num_samples = 500
            train_samples = int(total_num_samples*.8)
            test_samples = int(total_num_samples*.1)
            val_samples = int(total_num_samples*.1)

            train_samples_per_cluster = np.full(num_clusters, train_samples // num_clusters)
            test_samples_per_cluster = np.full(num_clusters, test_samples // num_clusters)
            val_samples_per_cluster = np.full(num_clusters, val_samples // num_clusters)

            train_ds = HypercubeDataset(
                num_dims=num_dims,
                num_samples=train_samples_per_cluster,
                hf_std=hf_std,
                lf_std=lf_std,
            )

            test_ds = HypercubeDataset(
                num_dims=num_dims,
                num_samples=test_samples_per_cluster,
                hf_std=hf_std,
                lf_std=lf_std,
            )

            val_ds = HypercubeDataset(
                num_dims=num_dims,
                num_samples=val_samples_per_cluster,
                hf_std=hf_std,
                lf_std=lf_std,
            )

        # mnist example 
        # lf = noisy; hf = clean
        case "mnist_noise":
            hf_augment: str = "noise"
            hf_aug_str: float = 0.0
            
            lf_augment: str = "noise"
            lf_aug_str: float = 0.7

            hf_transform = build_mnist_transform(
                aug_name=hf_augment,
                aug_strength=hf_aug_str
            )

            lf_transform = build_mnist_transform(
                aug_name=lf_augment,
                aug_strength=lf_aug_str
            )

            train_ds = MNISTDataset(
                split="train",
                root=folder,
                seed=seed,
                hf_transform=hf_transform,
                lf_transform=lf_transform
            )

            test_ds = MNISTDataset(
                split="test",
                root=folder,
                seed=seed,
                hf_transform=hf_transform,
                lf_transform=lf_transform
            )

            val_ds = MNISTDataset(
                split="val",
                root=folder,
                seed=seed,
                hf_transform=hf_transform,
                lf_transform=lf_transform
            )
        
        # mnist example
        # lf = rotated; hf = clean
        case "mnist_rotation":
            hf_augment: str = "rotation"
            hf_aug_str: float = 0.0
            
            lf_augment: str = "rotation"
            lf_aug_str: float = 90

            hf_transform = build_mnist_transform(
                aug_name=hf_augment,
                aug_strength=hf_aug_str
            )

            lf_transform = build_mnist_transform(
                aug_name=lf_augment,
                aug_strength=lf_aug_str
            )

            train_ds = MNISTDataset(
                split="train",
                root=folder,
                seed=seed,
                hf_transform=hf_transform,
                lf_transform=lf_transform
            )

            test_ds = MNISTDataset(
                split="test",
                root=folder,
                seed=seed,
                hf_transform=hf_transform,
                lf_transform=lf_transform
            )

            val_ds = MNISTDataset(
                split="val",
                root=folder,
                seed=seed,
                hf_transform=hf_transform,
                lf_transform=lf_transform
            )

        # cub-200 example (200 bird classes)
        # lf = grayscale; hf = rgb
        case "bird_grayscale":
            train_ds = CUBDataset(
                root=folder,
                split="train",
                seed=seed,
                grayscale=True
            )

            test_ds = CUBDataset(
                root=folder,
                split="test",
                seed=seed,
                grayscale=True
            )

            val_ds = CUBDataset(
                root=folder,
                split="val",
                seed=seed,
                grayscale=True
            )      

        # cub-200 example (200 bird classes)
        # lf = rgb; hf = rgb (used if lf and hf are different models)
        case "bird_color":
            train_ds = CUBDataset(
                root=folder,
                split="train",
                seed=seed,
                grayscale=False
            )

            test_ds = CUBDataset(
                root=folder,
                split="test",
                seed=seed,
                grayscale=False
            )

            val_ds = CUBDataset(
                root=folder,
                split="val",
                seed=seed,
                grayscale=False
            )    

        # https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification
        # multi-temporal crop classification dataset
        # lf = rgb; hf = rgb + 3 IR channels
        case "crop":
            train_ds = CropDataset(
                root=folder,
                split="train",
                seed=seed,
            )

            test_ds = CropDataset(
                root=folder,
                split="test",
                seed=seed,
            )

            val_ds = CropDataset(
                root=folder,
                split="val",
                seed=seed,
            )

        case _:
            logger.error("Dataset Name is invalid")
            raise ValueError("Dataset Name is invalid")
        
    return train_ds, test_ds, val_ds

def build_model(model_name, latent_size, output_size, input_size=None):
    match model_name:
        # multilayer perceptron used for toy dataset case
        case "mlp":
            if input_size is None:
                logger.error("Parameter input_size must be set for this model")
                raise ValueError("Parameter input_size must be set for this model")

            model = CustomMLP(
                input_size=input_size,
                num_layers=2,
                output_size=output_size,
                hidden_size=32
            )

        # ResNet18 used for image classification tasks (MNIST + CUB-200)
        case "resnet":
            if input_size is None:
                logger.error("Parameter input_size must be set for this model")
                raise ValueError("Parameter input_size must be set for this model")

            model = CustomResNet18(
                latent_size=latent_size,
                num_channels=input_size,
                output_size=output_size
            )

        # VIT used for image classification tasks (MNIST + CUB-200)
        case "vit":
            model = CustomViT(
                latent_size=latent_size,
                output_size=output_size
            )

        # UNET used for image segmentation tasks (Crop)
        case "unet":
            if input_size is None:
                logger.error("Parameter input_size must be set for this model")
                raise ValueError("Parameter input_size must be set for this model")

            model = build_unet(
                num_channels=input_size,
                num_classes=output_size
            )

        # CNN head specifically for the FE model if the UNET is used for the LF model
        case "cnn_head":
            model = LatentCNNHead(
                in_channels=latent_size,
                num_classes=output_size
            )

    return model

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
            output = model(data)[-1]

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
        for lf_data, hf_data, target in dataloader:
            target = target.type(torch.LongTensor)
            data = torch.cat([lf_data, hf_data], dim=1)

            data, target = data.to(device, torch.float), target.to(device)
            output = model(data)[-1]
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

            outputs = model(lf_embeddings)[-1]
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

def create_fe_dataset(dataloader, lf_model, hf_model, device):
    lf_embeddings = []
    lf_preds = []
    hf_preds = []
    labels = []

    # Set models to eval mode
    lf_model.eval()
    hf_model.eval()

    lf_model.to(device)
    hf_model.to(device)

    with torch.no_grad():
        for lf_sample, hf_sample, label in dataloader:
            if device is not None:
                lf_sample = lf_sample.to(device)
                hf_sample = hf_sample.to(device)
                label = label.to(device)

            # LF model
            lf_outs = lf_model(lf_sample)
            lf_embed = lf_outs[0]
            lf_pred = lf_outs[-1]

            # HF model (concatenate lf_sample and hf_sample along last dim)
            hf_input = torch.cat([lf_sample, hf_sample], dim=1)
            hf_outs = hf_model(hf_input)
            hf_pred = hf_outs[-1]

            lf_embeddings.append(lf_embed.cpu())
            lf_preds.append(lf_pred.cpu())
            hf_preds.append(hf_pred.cpu())
            labels.append(label.cpu())

    # Concatenate all batches
    lf_embeddings = torch.cat(lf_embeddings, dim=0)
    lf_preds = torch.cat(lf_preds, dim=0)
    hf_preds = torch.cat(hf_preds, dim=0)
    labels = torch.cat(labels, dim=0)

    return FE_Dataset(lf_embeddings, lf_preds, hf_preds, labels)

def reset_all_weights(model):
    def weight_reset(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    model.apply(weight_reset)


# class ModelBuilder:
#     def __init__(self, config: Options, device: torch.device):
#         self.config = config
#         self.device = device

#         self.input_sizes = {
#             "toy": self.config.dataset.toy_dataset_parameters.num_dims,
#             "mnist": None,
#             "cifar10": None,
#             "cifar100": None,
#         }

#         self.output_sizes = {
#             "toy": self.config.dataset.toy_dataset_parameters.num_classes,
#             "mnist": 10,
#             "cifar10": 10,
#             "cifar100": 100,
#         }

#     def __build_classifier__(self, model_type, dataset_name, pretrained=None):
#         input_size = self.input_sizes[dataset_name]
#         output_size = self.output_sizes[dataset_name]
#         latent_size = self.config.parameters.latent_representation_size

#         if "resnet" in model_type:
#             num_layers = int(model_type[6:])

#             model = build_resnet(
#                 resnet_size=num_layers,
#                 latent_size=latent_size,
#                 output_size=output_size,
#                 pretrained=pretrained,
#                 device=self.device
#             )
        
#         else:
#             num_layers = self.config.stage1.hf_model.classifier.num_layers

#             model = build_mlp(
#                 input_size=input_size,
#                 num_layers=num_layers,
#                 output_size=output_size
#             )

#         return model

#     def build_classifiers(self):
#         dataset_name = self.config.dataset.name

#         hf_model_type = self.config.stage1.hf_model.classifier.type
#         lf_model_type = self.config.stage1.lf_model.classifier.type

#         if dataset_name == "toy" and hf_model_type != "mlp":
#             raise ValueError("stage1.hf_model.classifier.type must be 'mlp' with the 'toy' dataset")
        
#         if dataset_name == "toy" and lf_model_type != "mlp":
#             raise ValueError("stage1.lf_model.classifier.type must be 'mlp' with the 'toy' dataset")
        
#         hf_pretrained = self.config.stage1.hf_model.classifier.pretrained
#         lf_pretrained = self.config.stage1.lf_model.classifier.pretrained

#         hf_model = self.__build_classifier__(
#             model_type=hf_model_type,
#             dataset_name=dataset_name,
#             pretrained=hf_pretrained
#         )

#         lf_model = self.__build_classifier__(
#             model_type=lf_model_type,
#             dataset_name=dataset_name,
#             pretrained=lf_pretrained
#         )

#         hf_model_path = str(self.config.stage1.hf_model.load_file)
#         if path.exists(hf_model_path):
#             hf_state_dict = torch.load(hf_model_path)
#             hf_model.load_state_dict(hf_state_dict)

#         lf_model_path = str(self.config.stage1.lf_model.load_file)
#         if path.exists(lf_model_path):
#             lf_state_dict = torch.load(lf_model_path)
#             lf_model.load_state_dict(lf_state_dict)

#         return hf_model, lf_model


# def fe_nn_one_run(fe_model, hf_model, lf_model, dataloader, criterion, optimizer=None, scheduler=None):
#     # device = next(hf_model.parameters()).device

#     # lf_body = create_feature_extractor(
#     #     lf_model, {"7": "body"}
#     # ).to(device)

#     # for hf_data, lf_data, target in dataloader:
#     #     target = target.type(torch.LongTensor)
#     #     hf_data = hf_data.to(device)
#     #     lf_data = lf_data.to(device)
#     #     target = target.to(device)

#     #     lf_embeddings = lf_body(lf_data)
#     #     lf_output = lf_model(lf_data)
#     #     hf_output = hf_model(hf_data)

#     #     fe_output = fe_model(lf_embeddings)
        
#     #     loss = criterion(target, preds, outputs)
#     #     if optimizer:
#     #         optimizer.zero_grad()
#     #         loss.backward()
#     #         optimizer.step()
#     pass

# def fe_svm_one_run(fe_model, hf_model, lf_model, dataloader, hf_weight, mode="train"):
#     device = next(hf_model.parameters()).device

#     lf_body = create_feature_extractor(
#         lf_model, {"7": "body"}
#     ).to(device)

#     num_samples = len(dataloader.dataset)
#     batch_size = dataloader.batch_size

#     lf_embeddings = np.zeros((num_samples, 32))
#     lf_preds = np.zeros((num_samples, 2))
#     hf_preds = np.zeros((num_samples, 2))
#     labels = np.zeros(num_samples)

#     for i, (hf_data, lf_data, target) in enumerate(dataloader):
#         target = target.type(torch.LongTensor) 
#         hf_data = hf_data.to(device, torch.float)
#         lf_data = lf_data.to(device, torch.float)
#         target = target.to(device)

#         lf_embs_tmp = lf_body(lf_data)["body"].detach().cpu().numpy()
#         lf_output_tmp = lf_model(lf_data).detach().cpu().numpy()
#         hf_output_tmp = hf_model(hf_data).detach().cpu().numpy()

#         offset = len(lf_embs_tmp)

#         lf_embeddings[i*batch_size:(i*batch_size+offset)] = lf_embs_tmp
#         lf_preds[i*batch_size:(i*batch_size+offset)] = lf_output_tmp
#         hf_preds[i*batch_size:(i*batch_size+offset)] = hf_output_tmp
#         labels[i*batch_size:(i*batch_size+offset)] = target.cpu().numpy()

#     lf_correct = np.argmax(lf_preds, axis=1) == labels
#     hf_correct = np.argmax(hf_preds, axis=1) == labels

#     best_choices = np.logical_and(~lf_correct, hf_correct).astype(int)

#     if mode=="train":
#         weights = np.where(best_choices==1, hf_weight, 1)
#         fe_model.fit(lf_embeddings, labels, sample_weights=weights)
        


# def build_qe_model(num_classes: int=2):
#     qe_model = nn.Sequential(
#         nn.Linear(32, 64),
#         nn.ReLU(),
#         nn.Linear(64, 128),
#         nn.ReLU(),
#         nn.Linear(128, 64),
#         nn.ReLU(),
#         nn.Linear(64, num_classes),
#         nn.Softmax(dim=1)
#     )

#     return qe_model


# def build_dataloaders(full_dataset, noise=1):
#     # train_dataset = BinaryHypercubeDataset(1275, noise_level=noise)
#     # test_dataset = BinaryHypercubeDataset(150, noise_level=noise)
#     # val_dataset = BinaryHypercubeDataset(75, noise_level=noise)

#     rand_idxs = np.random.choice(3, p=[.85, .1, .05], size=len(full_dataset))
#     idxs = np.arange(len(full_dataset))

#     train_dataset = Subset(full_dataset, idxs[rand_idxs==0])

#     test_dataset = Subset(full_dataset, idxs[rand_idxs==1])
#     val_dataset = Subset(full_dataset, idxs[rand_idxs==2])

#     train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

#     return train_loader, test_loader, val_loader

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
    

# def build_metadata(config: Options):
#     augmentation = config.dataset.augmentation
#     dataset = config.dataset.name
#     aug_deg = config.dataset.augmentation_level

#     metadata = MetaData(
#         loss = PerformanceData(
#             train = [],
#             test = [],
#             val = []
#         ),
#         acc = PerformanceData(
#             train = [],
#             test = [],
#             val = []
#         ),
#         augmentation = augmentation,
#         augmentation_degree = aug_deg,
#         dataset = dataset
#     )

#     return metadata