import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms as transforms
import yaml
import os
import logging
from torch.utils.data import DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

from datasets import HypercubeDataset, MNISTDataset, CropDataset, CUBDataset
from custom_types import ConfigOptions, FEResult
from models import CustomMLP, CustomResNet18, CustomViT, build_unet, LatentCNNHead
from losses import MetaLossFunction
from comparisons import SelectiveNetMethod

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
                hidden_size=latent_size
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
                in_channels=input_size,
                num_classes=output_size
            )

    return model

def classifier_one_run(model, dataloader, criterion, fidelity, train_body=True, optimizer=None, scheduler=None):
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
    total_correct = 0.0
    total_samples = 0.0
    total_high = 0

    if fidelity == "lf":
        for data, _, target in dataloader:
            target = target.type(torch.LongTensor)
            data, target = data.to(device, torch.float), target.to(device)
            
            if train_body:
                output = model(data)["output"]

            else:
                output = model.head(data)["output"]

            loss = criterion(output, target)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            predicted = torch.argmax(output, dim=1)

            is_segmentation = (output.ndim == 4 and target.ndim >= 3)
            if is_segmentation:
                total_correct += (predicted == target).float().sum().item()
                total_samples += predicted.numel()  # B*C*H*W but C=1 after argmax
            else:
                total_correct += (predicted == target).sum().item()
                total_samples += data.size(0)

        if scheduler:
            scheduler.step()

    elif fidelity == "hf":
        for lf_data, hf_data, target in dataloader:
            target = target.type(torch.LongTensor)
            data = torch.cat([lf_data, hf_data], dim=1)
            data, target = data.to(device, torch.float), target.to(device)

            if train_body:
                output = model(data)["output"]

            else:
                output = model.head(data)["output"]

            loss = criterion(output, target)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            predicted = torch.argmax(output, dim=1)

            is_segmentation = (output.ndim == 4 and target.ndim >= 3)
            if is_segmentation:
                total_correct += (predicted == target).float().sum().item()
                total_samples += predicted.numel()  # B*C*H*W but C=1 after argmax
            else:
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

def save_body(
        lf_model: nn.Module,
        hf_model: nn.Module,  
        dataloader: DataLoader,
        save_folder: str, 
        device):
    
    lf_model = lf_model.to(device)
    hf_model = hf_model.to(device)
    os.mkdir(save_folder)

    lf_model.eval()
    hf_model.eval()

    with torch.no_grad():
        for batch_idx, (lf_sample, hf_sample, label) in enumerate(dataloader):
            lf_sample = lf_sample.to(device)
            hf_sample = hf_sample.to(device)
            labels = labels.to(device)

            lf_out_batch = lf_model(lf_sample)["body_output"].cpu()
            hf_out_batch = hf_model(hf_sample)["body_output"].cpu()
            labels_batch = labels.cpu()

            batch_size = lf_out_batch.size(0)
            for i in range(batch_size):
                lf_out = lf_out_batch[i]
                hf_out = hf_out_batch[i]
                label = labels_batch[i]

                torch.save({
                    "lf_body_output": lf_out,
                    "hf_body_output": hf_out,
                    "label": label
                }, os.path.join(save_folder, f"sample_{batch_idx}_{i}.pt"))

    logger.info(f"Finished saving split to {save_folder}")

def save_latent(
        lf_model: nn.Module,
        hf_model: nn.Module,  
        dataloader: DataLoader,
        save_folder: str, 
        train_body: bool,
        device):

    lf_model = lf_model.to(device)
    hf_model = hf_model.to(device)
    os.mkdir(save_folder)

    lf_model.eval()
    hf_model.eval()

    with torch.no_grad():
        for batch_idx, (lf, hf, labels) in enumerate(dataloader):
            lf = lf.to(device)
            hf = hf.to(device)
            labels = labels.to(device)

            if train_body:
                lf_head = lf_model(lf)
                hf_head = hf_model(hf)
            else:
                lf_head = lf_model.head(lf)
                hf_head = hf_model.head(hf)
            

            lf_latent = lf_head["latent"]
            lf_output = lf_head["output"]

            # Pass hf through hf_model head (latent + output)
            hf_output = hf_head["output"]

            lf_latent = lf_latent.cpu()
            lf_output = lf_output.cpu()
            hf_output = hf_output.cpu()
            labels = labels.cpu()

            batch_size = lf_latent.size(0)
            for i in range(batch_size):
                torch.save({
                    "lf_latent": lf_latent[i],
                    "lf_output": lf_output[i],
                    "hf_output": hf_output[i],
                    "label": labels[i]
                }, os.path.join(save_folder, f"sample_{batch_idx}_{i}.pt"))

    logger.info(f"Finished saving split to {save_folder}")

def get_folder_size(folder):
    total_size = 0
    try:
        for entry in os.scandir(folder):
            if entry.is_file():
                total_size += entry.stat().st_size
            elif entry.is_dir():
                total_size += get_folder_size(entry.path)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return 0
    return total_size

def reset_all_weights(model):
    def weight_reset(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    model.apply(weight_reset)

class GaussianProcessSearch:
    def __init__(self, fe_model, device, train_dl, val_dl, test_dl, model_folder):
        self.evaluated_points = []
        self.fe_model = fe_model
        self.device = device
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.model_folder = model_folder

    def expected_improvement(X, X_sample, gp, xi=0.01):
        mu, sigma = gp.predict(X, return_std=True)
        mu_sample = gp.predict(X_sample)

        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            from scipy.stats import norm
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def propose_location(acquisition, X_sample, gp, bounds, n_restarts=25):
        dim = X_sample.shape[1]
        min_val = 1e20
        min_x = None
        
        def min_obj(X):
            return -acquisition(X.reshape(-1, dim), X_sample, gp)
        
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x
                
        return min_x.reshape(-1, dim)
    
    def train_fe_model(self, r_val):
        criterion = MetaLossFunction(
            ch=[r_val],
            cw=1,
            device=self.device
        )

        self.fe_model = reset_all_weights(self.fe_model)
        self.fe_model = self.fe_model.to(self.device)
        fe_optimizer = torch.optim.Adam(self.fe_model.parameters(), lr=3e-4, weight_decay=1e-5)

        best_val_loss = float('inf')
        best_val_acc = float('inf')
        best_val_use = float('inf')
        fe_model_file = os.path.join(self.model_folder, f"fe_model-{r_val}.pt")

        for _ in range(30):
            _, _, _ = classifier_one_run(
                model=self.fe_model,
                dataloader=self.train_dl,
                criterion=criterion,
                fidelity="gate",
                optimizer=fe_optimizer
            )

            val_loss, val_acc, val_use = classifier_one_run(
                model=self.fe_model,
                dataloader=self.val_dl,
                criterion=criterion,
                fidelity="gate",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_use = val_use
                torch.save(self.fe_model.state_dict(), fe_model_file)

        return best_val_loss, best_val_acc, best_val_use

    def evaluate_fe_model(self, r_val):
        criterion = MetaLossFunction(
            ch=[r_val],
            cw=1,
            device=self.device
        )

        fe_model_file = os.path.join(self.model_folder, f"fe_model-{r_val}.pt")
        self.fe_model.load_state_dict(torch.load(fe_model_file, weights_only=True))
        self.fe_model = self.fe_model.to(self.device)

        _, test_acc, test_use = classifier_one_run(
            model=self.fe_model,
            dataloader=self.test_dl,
            criterion=criterion,
            fidelity="gate",
        )

        return test_acc, test_use

    def find_r_for_target(self, usage, tolerance=1e-3):
        kernel = ConstantKernel(1.0, (0.1, 10)) * RBF(length_scale=0.1, length_scale_bounds=(0.01, 0.5))

        closest_usage = 10000
        reruns = 0
        while abs(usage - closest_usage) > tolerance:
            reruns += 1
            if self.evaluated_points == []:
                loss, acc, use = self.train_fe_model(0)
                self.evaluated_points.append(
                    FEResult(
                        r = 0, loss = loss,
                        accuracy = acc, usage = use 
                    )
                )
                closest_usage = use
                continue

            r_vals = np.array([point.r for point in self.evaluated_points])
            usages = np.array([point.usage for point in self.evaluated_points])

            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False)
            gp.fit(r_vals, usages)
            bounds = np.array([[0, 1]])  # Adjust to your parameter space
            next_point = self.propose_location(self.expected_improvement, r_vals, gp, bounds)

            loss, acc, use = self.train_fe_model(next_point)
            self.evaluated_points.append(
                FEResult(
                    r = next_point, loss = loss,
                    accuracy = acc, usage = use 
                )
            )

            usages = np.append(usages, use)
            idx = np.argmin(np.abs(usages - usage))
            closest_usage = usages[idx]

        idx = np.argmin(np.abs([point.usage for point in self.evaluated_points] - usage))
        best_point = self.evaluated_points[idx]

        test_acc, test_use = self.evaluate_fe_model(best_point.r)
        test_acc *= 100
        test_use *= 100

        logger.info(f"\nTook {reruns} passes to find best r value")
        logger.info(f"For usage {usage}:")
        logger.info(f"\tBest r: {best_point.r}")
        logger.info(f"\tClosest val usage: {best_point.use:.4f}")
        logger.info(f"\tTest usage: {test_use:.2f} / Test acc: {test_acc:.2f}")

        return best_point

class AdaptiveGridSearch:
    def __init__(self, fe_model, device, train_dl, val_dl, test_dl, model_folder):
        self.evaluated_points = []
        self.fe_model = fe_model
        self.device = device
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.model_folder = model_folder

    def find_bracket(self, usage):
        # Sort points by r to ensure order
        self.evaluated_points.sort(key=lambda x: x[0])

        for i in range(len(self.evaluated_points) - 1):
            r_a, use_a = self.evaluated_points[i]
            r_b, use_b = self.evaluated_points[i+1]
            if use_a <= usage < use_b:
                return r_a, r_b
            
        # If no exact bracket, use full range as fallback
        return 0.001, 1

    def train_fe_model(self, r_val):
        criterion = MetaLossFunction(
            ch=[r_val],
            cw=1,
            device=self.device
        )

        self.fe_model = reset_all_weights(self.fe_model)
        self.fe_model = self.fe_model.to(self.device)
        fe_optimizer = torch.optim.Adam(self.fe_model.parameters(), lr=3e-4, weight_decay=1e-5)

        best_val_loss = float('inf')
        fe_model_file = os.path.join(self.model_folder, f"fe_model-{r_val}.pt")

        for epoch in range(30):
            _, _, _ = classifier_one_run(
                model=self.fe_model,
                dataloader=self.train_dl,
                criterion=criterion,
                fidelity="gate",
                optimizer=fe_optimizer
            )

            val_loss, _, val_use = classifier_one_run(
                model=self.fe_model,
                dataloader=self.val_dl,
                criterion=criterion,
                fidelity="gate",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.fe_model.state_dict(), fe_model_file)

        return val_use

    def evaluate_fe_model(self, r_val):
        criterion = MetaLossFunction(
            ch=[r_val],
            cw=1,
            device=self.device
        )

        fe_model_file = os.path.join(self.model_folder, f"fe_model-{r_val}.pt")
        self.fe_model.load_state_dict(torch.load(fe_model_file, weights_only=True))
        self.fe_model = self.fe_model.to(self.device)

        _, test_acc, test_use = classifier_one_run(
            model=self.fe_model,
            dataloader=self.test_dl,
            criterion=criterion,
            fidelity="gate",
        )

        return test_acc, test_use

    def find_r_for_target(self, usage, tolerance=1e-3):
        r_low, r_high = self.find_bracket(usage)
        reruns = 0
        while r_high - r_low > tolerance:
            reruns += 1
            r_mid = (r_low + r_high) / 2
            use_mid = self.train_fe_model(r_mid)
            self.evaluated_points.append((r_mid, use_mid))
            if use_mid > usage:
                r_low = r_mid
            else:
                r_high = r_mid

        test_acc, test_use = self.evaluate_fe_model(r_high)
        test_acc *= 100
        test_use *= 100

        logger.info(f"\nTook {reruns} passes to find best r value")
        logger.info(f"For usage {usage}:")
        logger.info(f"\tBest r: {r_high}")
        logger.info(f"\tClosest val usage: {self.evaluated_points[-1][1]:.4f}")
        logger.info(f"\tTest usage: {test_use:.2f} / Test acc: {test_acc:.2f}")
        
        return r_high, test_acc, test_use
    
# def fe_svm_one_run(fe_m odel, hf_model, lf_model, dataloader, hf_weight, mode="train"):
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

def selnet_one_run(model, dataloader, train_body=True, optimizer=None):
    model.train(mode=bool(optimizer))
    device = next(model.parameters()).device
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_high = 0


    for data, _, target in dataloader:
        target = target.type(torch.LongTensor)
        data, target = data.to(device, torch.float), target.to(device)
        
        if train_body:
            output = model(data)["output"]

        else:
            output = model.head(data)["output"]

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
    