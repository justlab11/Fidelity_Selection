import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torchvision.transforms as transforms
import yaml
import os
import logging
import time
import math
import csv
from torch.utils.data import DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

from datasets import HypercubeDataset, MNISTDataset, CropDataset, CUBDataset, LLVIPDataset, BigEarthNetDataset
from custom_types import ConfigOptions, FEResult
from models import CustomMLP, CustomResNet18, CustomViT, build_unet, LatentCNNHead, build_yolov5, YOLOv5FidelityModel
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
        # toy example where 2d points are split into 4 clusters at the corners
        # of the unit square, grouped into 2 classes by diagonal (XOR-style:
        # (0,0)/(1,1) are one class, (0,1)/(1,0) are the other) - so the
        # cluster nearest any given point is only half the story; the other
        # coordinate matters too. This is deliberate: with hf_std tight enough
        # to keep each cluster essentially pure and lf_std wide enough for
        # neighboring opposite-class clusters' tails to overlap, the region
        # where LF is genuinely ambiguous forms a "+"-shaped band straddling
        # x=0.5 and y=0.5 (equidistant between opposite-class neighbors),
        # while LF stays reliable out in each cluster's own corner - exactly
        # the "confident on the outskirts, need HF in the +" pattern the FE
        # gate is meant to learn. (Verified via an SVC ceiling: hf_std~0.28/
        # lf_std~0.51 gives ~90-92% for HF vs. ~72-73% for LF - deliberately
        # not near-perfect for HF, so there's still real HF error for the gate
        # to weigh against LF's - LF = noisy; hf = clean.)
        case "toy_2d":
            num_dims = 2
            num_clusters = 2 ** num_dims
            hf_std = np.random.uniform(0.26, 0.30, size=num_clusters)
            lf_std = np.random.uniform(0.48, 0.54, size=num_clusters)

            total_num_samples = 3000
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
            mean = train_ds.mean
            std = train_ds.std

            test_ds = CropDataset(
                root=folder,
                split="test",
                seed=seed,
            )
            test_ds.mean = mean
            test_ds.std = std

            val_ds = CropDataset(
                root=folder,
                split="val",
                seed=seed,
            )
            val_ds.mean = mean
            val_ds.std = std

        # https://bupt-ai-cz.github.io/LLVIP/
        # paired visible/infrared pedestrian-detection dataset
        # lf = visible; hf = infrared
        case "llvip":
            train_ds = LLVIPDataset(
                root=folder,
                split="train",
                seed=seed,
            )

            test_ds = LLVIPDataset(
                root=folder,
                split="test",
                seed=seed,
            )

            val_ds = LLVIPDataset(
                root=folder,
                split="val",
                seed=seed,
            )

        # https://bigearth.net/
        # paired Sentinel-1 (SAR) / Sentinel-2 (multispectral optical) dataset
        # lf = s1; hf = s2
        case "bigearthnet":
            train_ds = BigEarthNetDataset(
                root=folder,
                split="train",
                seed=seed,
            )

            test_ds = BigEarthNetDataset(
                root=folder,
                split="test",
                seed=seed,
            )

            val_ds = BigEarthNetDataset(
                root=folder,
                split="val",
                seed=seed,
            )

        case _:
            logger.error("Dataset Name is invalid")
            raise ValueError("Dataset Name is invalid")

    return train_ds, test_ds, val_ds

def build_model(model_name, latent_size, output_size, input_size=None, weights_path=None, device="cpu"):
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

        # YOLOv5 detection model used for the LLVIP pedestrian-detection dataset.
        # Unlike the other cases, this doesn't build a fresh model to be trained -
        # it loads a pretrained checkpoint directly, so weights_path is required
        # (this is why ClassifierSettings enforces trained_lf_model/trained_hf_model
        # be set in the config whenever lf_model/hf_model is "yolo").
        case "yolo":
            if weights_path is None:
                logger.error("Parameter weights_path must be set for this model")
                raise ValueError("Parameter weights_path must be set for this model")

            model = build_yolov5(
                weights_path=weights_path,
                device=device
            )

    return model

def assemble_hf_input(lf_data, hf_data, hf_input_mode: str, hf_model: nn.Module = None):
    """Build the tensor fed into the HF model, per the configured input mode."""
    if isinstance(hf_model, YOLOv5FidelityModel):
        # YOLOv5's backbone expects a plain 3-channel image - hf_input_mode's
        # "concat" (LF+HF channels stacked) doesn't apply to a pretrained
        # detection checkpoint, so always hand it the raw HF-modality image.
        return hf_data

    if hf_input_mode == "concat":
        return torch.cat([lf_data, hf_data], dim=1)
    elif hf_input_mode == "hf_only":
        return hf_data
    else:
        raise ValueError(f"Invalid hf_input_mode: {hf_input_mode}")

def compute_yolo_detection_correctness(
        preds: torch.Tensor,
        targets: torch.Tensor,
        img_size,
        conf_thres: float = 0.25,
        nms_iou_thres: float = 0.45,
        match_iou_thres: float = 0.5,
        required_recall: float = 0.5):
    """Turns raw YOLO detection output into a per-image correctness signal, so it
    can stand in for classification's "argmax(output) == target" everywhere else
    in the pipeline (gate routing, FE/gate loss, reported accuracy).

    Args:
    - preds: (B, num_anchors, 5+num_classes) raw model output, pre-NMS.
    - targets: (B, max_boxes, 5) padded [cls, xc, yc, w, h] ground truth, with
      xc/yc/w/h normalized to [0, 1] (LLVIPDataset's convention) and unused rows
      padded with a class of -1.
    - img_size: (H, W) of the letterboxed input preds/targets were computed
      against - used to un-normalize target boxes into the pixel space
      non_max_suppression's output is already in.

    An image counts as "correct" if at least `required_recall` of its ground-
    truth boxes are matched (IoU >= match_iou_thres, greedy by detection
    confidence) by a surviving detection. This is a deliberately simple stand-in
    for real mAP-based detection scoring, not a publication-grade metric - tune
    required_recall/match_iou_thres/conf_thres to taste.

    Returns a (B,) float tensor of 1.0/0.0 per image.
    """
    from yolov5.utils.general import non_max_suppression, xywh2xyxy, box_iou

    h, w = img_size
    scale = torch.tensor([w, h, w, h], device=targets.device, dtype=targets.dtype)

    detections = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=nms_iou_thres)

    correct = torch.zeros(preds.size(0), dtype=torch.float32)
    for i, det in enumerate(detections):
        valid = targets[i, :, 0] >= 0
        gt_boxes = xywh2xyxy(targets[i, valid, 1:5] * scale)
        num_gt = gt_boxes.size(0)

        if num_gt == 0:
            correct[i] = 1.0 if det.size(0) == 0 else 0.0
            continue

        if det.size(0) == 0:
            correct[i] = 0.0
            continue

        ious = box_iou(det[:, :4], gt_boxes)  # (num_det, num_gt)
        order = torch.argsort(det[:, 4], descending=True)
        matched_gt = torch.zeros(num_gt, dtype=torch.bool, device=ious.device)
        matched_count = 0

        for d in order:
            if matched_gt.all():
                break
            row = ious[d].clone()
            row[matched_gt] = -1
            best_gt = torch.argmax(row)
            if row[best_gt] >= match_iou_thres:
                matched_gt[best_gt] = True
                matched_count += 1

        recall = matched_count / num_gt
        correct[i] = 1.0 if recall >= required_recall else 0.0

    return correct

def compute_gate_routing_details(model, dataloader, device):
    """One no-grad pass over a "gate"-fidelity dataloader (lf_embeddings, lf_preds,
    hf_preds, target), returning per-sample arrays: the gate's routing decision and
    both models' own correctness — everything needed to build the ground-truth
    "HF needed" / "LF fine" labels and the routing confusion counts, without
    duplicating classifier_one_run's loss/aggregate-accuracy bookkeeping.

    lf_embeddings may be flat (B, latent_dim) for an "mlp" gate or spatial
    (B, C, H, W) for a "cnn_head" gate (segmentation, or YOLO's unpooled
    backbone feature) - model(lf_embeddings) doesn't care either way. The
    returned "lf_latent" is pooled to (N, C) (mean over any spatial dims)
    before ever being concatenated across the full dataset: keeping the raw
    per-sample spatial tensor for the whole test set is what blows up memory
    for a large spatial gate (e.g. LLVIP/YOLO's (N, 1024, 20, 20) feature map
    is several GB at N in the thousands), and nothing downstream
    (AdaptiveGridSearch._project_latent's PCA/logistic-regression basis,
    _compute_decision_boundary's spatially-uniform reconstruction) ever needs
    more than this pooled summary plus the per-sample shape captured in
    "latent_sample_shape".
    """
    model.eval()

    lf_latents, lf_corrects, hf_corrects, choices, labels = [], [], [], [], []
    latent_sample_shape = None

    with torch.no_grad():
        for lf_embeddings, lf_preds, hf_preds, target in dataloader:
            lf_embeddings = lf_embeddings.to(device, torch.float)
            lf_preds = lf_preds.to(device, torch.float)
            hf_preds = hf_preds.to(device, torch.float)
            target = target.long().to(device)

            choice = torch.argmax(model(lf_embeddings)["output"], dim=1)  # 0=LF, 1=HF

            if latent_sample_shape is None:
                latent_sample_shape = tuple(lf_embeddings.shape[1:])

            lf_pooled = (
                lf_embeddings.mean(dim=tuple(range(2, lf_embeddings.ndim)))
                if lf_embeddings.ndim > 2 else lf_embeddings
            )

            lf_latents.append(lf_pooled.cpu().numpy())
            lf_corrects.append((torch.argmax(lf_preds, dim=1) == target).cpu().numpy())
            hf_corrects.append((torch.argmax(hf_preds, dim=1) == target).cpu().numpy())
            choices.append(choice.cpu().numpy())
            labels.append(target.cpu().numpy())

    return {
        "lf_latent": np.concatenate(lf_latents, axis=0),
        "lf_correct": np.concatenate(lf_corrects, axis=0),
        "hf_correct": np.concatenate(hf_corrects, axis=0),
        "choice": np.concatenate(choices, axis=0),
        "label": np.concatenate(labels, axis=0),
        "latent_sample_shape": latent_sample_shape,
    }

def classifier_one_run(model, dataloader, criterion, fidelity, train_body=True, optimizer=None, scheduler=None, hf_input_mode: str = "concat", grad_clip_norm: float = None):
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
    # Segmentation losses (crop/unet) are already averaged per-pixel by
    # criterion, so they need to be weighted/divided by batch count, not by
    # total_samples - that's pixel count (B*H*W) for segmentation, since it
    # also doubles as the accuracy denominator. Reusing total_samples for both
    # would divide a loss already keyed to `loss.item() * batch_size` by a
    # pixel count thousands of times larger, silently flooring the printed
    # loss to 0.0000 while accuracy (whose numerator/denominator scale
    # together) stays correct.
    total_loss_samples = 0.0
    total_high = 0

    if fidelity == "lf":
        for data, _, target in dataloader:
            data = data.to(device, torch.float)

            if isinstance(model, YOLOv5FidelityModel):
                target = target.to(device)
                output = model(data)["output"]
                correct = compute_yolo_detection_correctness(output, target, img_size=data.shape[-2:])

                total_loss += (1 - correct).sum().item()
                total_correct += correct.sum().item()
                total_samples += data.size(0)
                total_loss_samples += data.size(0)
                continue

            target = target.type(torch.LongTensor).to(device)

            if train_body:
                output = model(data)["output"]

            else:
                output = model.head(data)["output"]

            loss = criterion(output, target)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_loss_samples += data.size(0)
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
            data = assemble_hf_input(lf_data, hf_data, hf_input_mode, hf_model=model) if train_body else hf_data
            data = data.to(device, torch.float)

            if isinstance(model, YOLOv5FidelityModel):
                target = target.to(device)
                output = model(data)["output"]
                correct = compute_yolo_detection_correctness(output, target, img_size=data.shape[-2:])

                total_loss += (1 - correct).sum().item()
                total_correct += correct.sum().item()
                total_samples += data.size(0)
                total_loss_samples += data.size(0)
                continue

            target = target.type(torch.LongTensor).to(device)

            if train_body:
                output = model(data)["output"]

            else:
                output = model.head(data)["output"]

            loss = criterion(output, target)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_loss_samples += data.size(0)
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

            outputs = model(lf_embeddings)["output"]
            preds = torch.stack([
                lf_preds,
                hf_preds
            ])

            loss = criterion(target, preds, outputs)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            choices = torch.argmax(outputs, dim=1)
            lf_acc = torch.argmax(lf_preds, dim=1) == target
            hf_acc = torch.argmax(hf_preds, dim=1) == target

            total_loss += loss.item() * lf_preds.size(0)
            total_correct += torch.sum(hf_acc[choices==1]) + torch.sum(lf_acc[choices==0]).item()
            total_samples += lf_preds.size(0)
            total_loss_samples += lf_preds.size(0)
            total_high += torch.sum(choices).item()
        if scheduler:
            scheduler.step()

    average_loss = total_loss / total_loss_samples
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
        device,
        hf_input_mode: str = "concat"):

    lf_model = lf_model.to(device)
    hf_model = hf_model.to(device)
    os.mkdir(save_folder)

    lf_model.eval()
    hf_model.eval()

    with torch.no_grad():
        for batch_idx, (lf_sample, hf_sample, label) in enumerate(dataloader):
            hf_sample = assemble_hf_input(lf_sample, hf_sample, hf_input_mode, hf_model=hf_model)

            lf_sample = lf_sample.to(device)
            hf_sample = hf_sample.to(device)
            label = label.to(device)

            lf_out_batch = lf_model(lf_sample)["body_output"].cpu()
            hf_out_batch = hf_model(hf_sample)["body_output"].cpu()
            labels_batch = label.cpu()

            batch_size = lf_out_batch.size(0)
            for i in range(batch_size):
                # .clone() detaches each sample into its own storage - without
                # it, lf_out_batch[i] is a view still sharing the *whole batch's*
                # underlying storage, so torch.save would silently serialize
                # every sibling sample into each per-sample file too (an easy-to-
                # miss ~batch_size x disk/IO blowup).
                lf_out = lf_out_batch[i].clone()
                hf_out = hf_out_batch[i].clone()
                label = labels_batch[i].clone()

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
        device="cpu",
        hf_input_mode: str = "concat",
        latent_key: str = "latent"):
    """
    latent_key selects which of the LF model's forward()-dict entries gets
    saved as "lf_latent" per sample. Every model besides UNet only has one
    sensible choice ("latent" - already a small projected vector), but UNet
    also exposes "body" (its actual encoder bottleneck - e.g. 1024x14x14 -
    vs. "latent"'s decoder output at the full 224x224 input resolution, kept
    spatial for a future per-pixel/region-level gate). main.py currently
    passes "body" for crop, since saving "latent" for every sample of that
    ~90k-sample dataset needs ~1.6TB of disk; pass "latent" there to bring
    the full-resolution map back (see UNet.forward's comment for the rest of
    what that would take).
    """

    lf_model = lf_model.to(device)
    hf_model = hf_model.to(device)
    os.mkdir(save_folder)

    lf_model.eval()
    hf_model.eval()

    lf_is_yolo = isinstance(lf_model, YOLOv5FidelityModel)
    hf_is_yolo = isinstance(hf_model, YOLOv5FidelityModel)
    if lf_is_yolo != hf_is_yolo:
        raise ValueError("Mixing a YOLO model with a non-YOLO model across lf/hf is not supported")

    with torch.no_grad():
        for batch_idx, (lf, hf, labels) in enumerate(dataloader):
            hf = assemble_hf_input(lf, hf, hf_input_mode, hf_model=hf_model)

            lf = lf.to(device)
            hf = hf.to(device)
            labels = labels.to(device)

            if train_body:
                lf_head = lf_model(lf)
                hf_head = hf_model(hf)
            else:
                lf_head = lf_model.head(lf)
                hf_head = hf_model.head(hf)


            lf_latent = lf_head[latent_key]

            if lf_is_yolo:
                # Redefine what "lf_output"/"hf_output"/"label" mean for a YOLO
                # cascade: instead of raw class logits, store a per-image
                # [incorrect, correct] pair (see compute_yolo_detection_correctness)
                # and a constant target of 1 ("correct" is always the true class).
                # argmax([1-correct, correct]) == 1 then reproduces `correct`
                # exactly, so every downstream consumer (compute_gate_routing_details,
                # the gate's CE loss, the routing plots) keeps working unmodified.
                lf_correct = compute_yolo_detection_correctness(lf_head["output"], labels, img_size=lf.shape[-2:])
                hf_correct = compute_yolo_detection_correctness(hf_head["output"], labels, img_size=hf.shape[-2:])

                lf_output = torch.stack([1 - lf_correct, lf_correct], dim=1)
                hf_output = torch.stack([1 - hf_correct, hf_correct], dim=1)
                labels = torch.ones(lf.size(0), dtype=torch.long)
            else:
                lf_output = lf_head["output"]

                # Pass hf through hf_model head (latent + output)
                hf_output = hf_head["output"]
                labels = labels.cpu()

            lf_latent = lf_latent.cpu()
            lf_output = lf_output.cpu()
            hf_output = hf_output.cpu()

            batch_size = lf_latent.size(0)
            for i in range(batch_size):
                # .clone() detaches each sample into its own storage - without
                # it, lf_latent[i] is a view still sharing the *whole batch's*
                # underlying storage, so torch.save would silently serialize
                # every sibling sample into each per-sample file too (an easy-to-
                # miss ~batch_size x disk/IO blowup - severe once lf_latent is a
                # large spatial (C, H, W) map rather than a small flat vector).
                torch.save({
                    "lf_latent": lf_latent[i].clone(),
                    "lf_output": lf_output[i].clone(),
                    "hf_output": hf_output[i].clone(),
                    "label": labels[i].clone()
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

def gate_logit_stats(model, dataloader, device):
    """Min/max/mean-abs of the gate model's raw routing logits over one pass.
    Diagnostic for softmax saturation during gate training - a routing MLP whose
    logits blow up to e.g. [+50, -50] underflows softmax to exactly (1.0, 0.0) in
    float32, killing the gradient for good. Watching these values climb toward
    the tens/hundreds while val_usage/val_loss go flat is the signature of that
    collapse (see AdaptiveGridSearch.train_fe_model's epoch_log).
    """
    model.eval()
    logit_min, logit_max, abs_sum, count = float("inf"), float("-inf"), 0.0, 0

    with torch.no_grad():
        for lf_embeddings, _, _, _ in dataloader:
            lf_embeddings = lf_embeddings.to(device, torch.float)
            outputs = model(lf_embeddings)["output"]

            logit_min = min(logit_min, outputs.min().item())
            logit_max = max(logit_max, outputs.max().item())
            abs_sum += outputs.abs().sum().item()
            count += outputs.numel()

    return logit_min, logit_max, abs_sum / count

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
    def __init__(
            self, fe_model, device, train_dl, val_dl, test_dl, model_folder,
            class_weighted_loss=False, gate_epochs=30, gate_lr=3e-4, gate_grad_clip_norm=None, seed=42):
        # evaluated_points is reset at the start of every rerun (see run_reruns) so
        # each rerun's bracket search is a genuinely independent sweep, not warm-
        # started off a previous rerun's history. all_evaluated_points pools every
        # rerun's points instead — more raw (r, usage, acc) samples only helps
        # plot_gate_sensitivity.py's derivative/zoom analysis, so nothing is reset there.
        self.evaluated_points = []
        self.all_evaluated_points = []
        self.epoch_log = []
        self.search_diagnostics = []
        # One entry per find_r_for_target call: the final gate model's per-sample
        # routing decision + both models' correctness, tagged by rerun/target_usage.
        # Feeds the routing projection plot and the routing confusion-table plot.
        self.routing_snapshots = []
        self.rerun_idx = 0
        self.fe_model = fe_model
        self.device = device
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.model_folder = model_folder
        self.class_weighted_loss = class_weighted_loss
        self.gate_epochs = gate_epochs
        self.gate_lr = gate_lr
        self.gate_grad_clip_norm = gate_grad_clip_norm
        self.seed = seed

        # Re-snapshotted at the start of every rerun (see run_reruns) - every
        # r_val *within* a rerun trains from this same fixed init/seed, so
        # differences in usage(r) reflect r itself rather than which random
        # basin that r_val's own from-scratch init happened to land near
        # (usage(r) was observed to swing from 0.99 to 0.26 to 1.0 across r
        # values barely 0.01 apart on LLVIP/YOLO, which is what motivated
        # this). Left un-fixed *across* reruns so run_reruns still measures
        # genuine run-to-run variance instead of repeating the same search
        # n_reruns times.
        self.fe_init_state = None
        self.rerun_seed = seed

        # lf_latent is the same fixed test-set data on every call (it's precomputed
        # LF-model output, independent of the gate model), so the projection basis
        # is fit once and reused everywhere — every snapshot's 2D coordinates line
        # up in the same space.
        self.proj_mean = None
        self.proj_basis = None  # (latent_dim, 2), orthonormal columns
        self.boundary_xx = None
        self.boundary_yy = None
        # Per-sample shape of the real (possibly spatial) latent, e.g. (1024,)
        # for a flat mlp-gate latent or (1024, 20, 20) for a cnn_head-gate's
        # spatial one — needed to reshape the flattened projection basis's
        # reconstructions back into what self.fe_model actually expects.
        self.latent_sample_shape = None

    def _project_latent(self, lf_latent, latent_sample_shape, hf_needed=None, grid_size=120, pad_frac=0.05):
        """Fits (once) a 2D linear basis on lf_latent and a matching decision-
        boundary meshgrid in that plane. Returns coords_2d.

        Unlike plain PCA (which picks the directions of highest variance in
        lf_latent, with no reason to align with where the gate actually
        disagrees with itself), this basis is supervised: axis 1 is the
        logistic-regression direction that best separates hf_needed from
        lf_fine, axis 2 is the top PCA direction of what's left after removing
        axis 1 (so it's orthogonal to axis 1 and still soaks up leftover
        structure). The basis stays linear and exactly invertible — same as
        PCA — so _compute_decision_boundary's inverse-transform trick (running
        the real fe_model over reconstructed latents) still holds.

        lf_latent is always the already-pooled (N, C) summary
        compute_gate_routing_details produces (mean over spatial dims for a
        cnn_head gate's (B, C, H, W) YOLO/UNet feature map) - flattening every
        spatial position instead (e.g. 409,600 dims for a 20x20x1024 YOLO
        feature) would blow up memory catastrophically (a single (N, D)
        float32 array several GB at N in the thousands) and put D far above
        N, where the supervised logistic-regression fit degenerates
        (near-perfect separation, unstable coefficients). latent_sample_shape
        (the true per-sample (C,) or (C, H, W) shape, from
        compute_gate_routing_details) is kept separately so
        _compute_decision_boundary still knows how to broadcast reconstructed
        grid points back into what self.fe_model actually expects - the real
        routing decisions elsewhere always use the true, unpooled spatial latent.
        """
        self.latent_sample_shape = latent_sample_shape
        lf_latent_flat = lf_latent

        if self.proj_basis is None:
            if hf_needed is None:
                raise ValueError("hf_needed labels are required to fit the projection basis")

            self.proj_mean = lf_latent_flat.mean(axis=0)
            centered = lf_latent_flat - self.proj_mean

            if len(np.unique(hf_needed)) < 2:
                # hf_needed (hf_correct & ~lf_correct) is constant across this
                # eval set - e.g. an easy task where lf_model is already ~100%
                # accurate, so hf is essentially never "needed". No separating
                # direction to fit, so fall back to plain (unsupervised) PCA for
                # both axes instead of raising - this basis is only for the
                # routing-projection plot, not the actual routing decision.
                logger.warning(
                    "_project_latent: hf_needed has a single class in this eval set; "
                    "falling back to unsupervised PCA for the projection basis"
                )
                pca = PCA(n_components=2)
                pca.fit(centered)
                axis1, axis2 = pca.components_[0], pca.components_[1]
                axis1 /= np.linalg.norm(axis1)
                axis2 /= np.linalg.norm(axis2)
            else:
                # Fit the separating direction inside the top-variance PCA
                # subspace of the latent, not the raw D dims directly. A ReLU
                # latent typically has several near-constant/sparse dimensions
                # (zero for almost every sample, nonzero for a handful) - dividing
                # by std below blows those up into huge standardized values for
                # exactly those few points, and an L2-penalized logistic fit can
                # exploit that "for free" to separate a handful of leverage
                # points instead of the real hf_needed/lf_fine boundary (observed
                # directly on toy_2d: axis1 ended up spanning ~1000x less range
                # than axis2, with the extreme axis1 outliers turning out to be
                # ordinary lf_fine points, not the hard/ambiguous ones).
                # Restricting the fit to the top principal directions first
                # denoises those degenerate dims away before the supervised step
                # ever sees them.
                n_components = min(15, centered.shape[0] - 1, centered.shape[1])
                pca_denoise = PCA(n_components=n_components)
                reduced = pca_denoise.fit_transform(centered)  # (N, k)

                reduced_std = reduced.std(axis=0)
                reduced_std[reduced_std == 0] = 1
                clf = LogisticRegression(max_iter=1000)
                clf.fit(reduced / reduced_std, hf_needed.astype(int))

                # Undo the subspace standardization, then re-expand from the
                # k-dim PCA subspace back into full latent space.
                axis1 = pca_denoise.components_.T @ (clf.coef_[0] / reduced_std)
                axis1 /= np.linalg.norm(axis1)

                # Remove axis1's component, then take the top PCA direction of the
                # residual — orthogonal to axis1 by construction.
                residual = centered - np.outer(centered @ axis1, axis1)
                pca_resid = PCA(n_components=1)
                pca_resid.fit(residual)
                axis2 = pca_resid.components_[0]
                axis2 /= np.linalg.norm(axis2)

            self.proj_basis = np.stack([axis1, axis2], axis=1)
            coords_2d = centered @ self.proj_basis

            x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
            y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
            x_pad = (x_max - x_min) * pad_frac
            y_pad = (y_max - y_min) * pad_frac
            self.boundary_xx, self.boundary_yy = np.meshgrid(
                np.linspace(x_min - x_pad, x_max + x_pad, grid_size),
                np.linspace(y_min - y_pad, y_max + y_pad, grid_size)
            )
        else:
            coords_2d = (lf_latent_flat - self.proj_mean) @ self.proj_basis

        return coords_2d

    def _compute_decision_boundary(self, batch_size=128):
        """Runs the *actual* trained self.fe_model (not a proxy classifier) over
        grid points in the cached projection plane, reconstructed back to the
        real latent dimensionality via the basis's transpose (exact since the
        basis columns are orthonormal). This is a genuine slice of the real
        decision surface through that 2D plane — necessarily an approximation
        (the plane can't capture the other latent_dim-2 axes), but it reflects
        the model's own nonlinearity rather than a re-fit linear stand-in.

        For a spatial gate (cnn_head), proj_mean/proj_basis operate on the
        pooled (C,) summary _project_latent fits on, so each reconstructed grid
        point is broadcast out to the model's true (C, H, W) input shape -
        treating it as a spatially-uniform feature map, a further approximation
        on top of the 2D-plane one above. Processed in batches so this never
        materializes more than batch_size full spatial tensors at once (all
        grid_size**2 of them at full (C, H, W) size at once is the same
        multi-GB blowup _project_latent's pooling was written to avoid).
        """
        grid_2d = np.column_stack([self.boundary_xx.ravel(), self.boundary_yy.ravel()])
        grid_latent = (self.proj_mean + grid_2d @ self.proj_basis.T).astype(np.float32)

        is_spatial = len(self.latent_sample_shape) > 1
        pooled_channels = self.latent_sample_shape[0] if is_spatial else None

        self.fe_model.eval()
        hf_probs = []
        with torch.no_grad():
            for start in range(0, grid_latent.shape[0], batch_size):
                chunk = torch.from_numpy(grid_latent[start:start + batch_size]).to(self.device)

                if is_spatial:
                    spatial_dims = self.latent_sample_shape[1:]
                    chunk = chunk.view(chunk.shape[0], pooled_channels, *([1] * len(spatial_dims)))
                    chunk = chunk.expand(chunk.shape[0], *self.latent_sample_shape).contiguous()
                else:
                    chunk = chunk.reshape((-1,) + self.latent_sample_shape)

                logits = self.fe_model(chunk)["output"]
                hf_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

        hf_prob = np.concatenate(hf_probs, axis=0)
        return hf_prob.reshape(self.boundary_xx.shape)

    def find_bracket(self, usage, min_bracket_width=0.05):
        """Only reuses an existing (r_a, r_b) pair from a prior target's search
        if it's wide enough to represent genuine exploration room. Bisection
        naturally converges its *own* final bracket down to ~tolerance width by
        design (see find_r_for_target) - blindly reusing that razor-thin final
        bracket as if it were informative for a completely different target
        usage is where this used to go wrong: usage(r) has repeatedly turned
        out to behave more like a step function than the smooth monotonic
        curve bisection assumes (see the gate's per-epoch routing logs), so a
        narrow leftover bracket can spuriously "contain" almost any target
        usage - silently skipping bisection (0 passes) and just replaying
        whichever checkpoint happened to converge for the earlier target,
        instead of doing real work for this one. Requiring a minimum width
        forces a fresh full-range search whenever the only "bracket" on hand is
        really just a stale, over-converged sliver.
        """
        # Sort points by r to ensure order
        self.evaluated_points.sort(key=lambda x: x[0])

        for i in range(len(self.evaluated_points) - 1):
            r_a, use_a, *_ = self.evaluated_points[i]
            r_b, use_b, *_ = self.evaluated_points[i+1]
            # usage decreases as r increases (see the bisection direction in
            # find_r_for_target), so for r_a < r_b we expect use_a >= use_b
            if use_b <= usage <= use_a and (r_b - r_a) >= min_bracket_width:
                return r_a, r_b, False

        # If no sufficiently wide bracket, use full range as fallback — this is
        # also the concrete signal that a target usage may end up unreachable
        # within tolerance.
        logger.warning(f"No sufficiently wide bracket found for usage={usage}; falling back to full range [0.001, 1]")
        return 0.001, 1, True

    def train_fe_model(self, r_val):
        criterion = MetaLossFunction(
            ch=[r_val],
            cw=1,
            device=self.device,
            class_weighted=self.class_weighted_loss
        )

        set_all_seeds(self.rerun_seed)
        self.fe_model.load_state_dict(self.fe_init_state)
        self.fe_model = self.fe_model.to(self.device)
        fe_optimizer = torch.optim.Adam(self.fe_model.parameters(), lr=self.gate_lr, weight_decay=1e-5)

        best_val_loss = float('inf')
        best_val_acc = None
        best_val_use = None
        fe_model_file = os.path.join(self.model_folder, f"fe_model-{r_val}.pt")

        for epoch in range(self.gate_epochs):
            epoch_start = time.perf_counter()

            train_loss, _, _ = classifier_one_run(
                model=self.fe_model,
                dataloader=self.train_dl,
                criterion=criterion,
                fidelity="gate",
                optimizer=fe_optimizer,
                grad_clip_norm=self.gate_grad_clip_norm
            )

            val_loss, val_acc, val_use = classifier_one_run(
                model=self.fe_model,
                dataloader=self.val_dl,
                criterion=criterion,
                fidelity="gate",
            )

            # Diagnostic: min/max/mean-abs of the gate's raw routing logits. Climbing
            # toward the tens/hundreds while val_usage/val_loss go flat is the
            # signature of softmax saturation killing the gradient - see gate_logit_stats.
            logit_min, logit_max, logit_absmean = gate_logit_stats(self.fe_model, self.val_dl, self.device)

            epoch_time_sec = time.perf_counter() - epoch_start

            if not math.isfinite(val_loss):
                logger.warning(f"Non-finite val_loss ({val_loss}) training gate model at r_val={r_val}, epoch={epoch}")

            # Live progress - previously the only record of gate training was
            # self.epoch_log, written to gate_epoch_metrics.csv once at the very
            # end of the whole run, so a long search gave zero visibility into
            # whether it was progressing or stuck.
            logger.info(
                f"[gate rerun={self.rerun_idx} r={r_val:.6f}] epoch {epoch + 1}/{self.gate_epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"val_usage={val_use:.4f} logit_absmean={logit_absmean:.2f} ({epoch_time_sec:.2f}s/epoch)"
            )

            self.epoch_log.append({
                "rerun": self.rerun_idx, "r_val": r_val, "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "val_acc": val_acc, "val_usage": val_use,
                "logit_min": logit_min, "logit_max": logit_max, "logit_absmean": logit_absmean,
                "epoch_time_sec": epoch_time_sec
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_use = val_use
                torch.save(self.fe_model.state_dict(), fe_model_file)

        return best_val_use, best_val_acc

    def save_epoch_log(self, file_path):
        if not self.epoch_log:
            logger.warning("AdaptiveGridSearch.epoch_log is empty; nothing to save")
            return

        fieldnames = list(self.epoch_log[0].keys())
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.epoch_log)
        logger.info(f"Saved {len(self.epoch_log)} gate-model epoch rows to {file_path}")

    def save_search_diagnostics(self, file_path):
        if not self.search_diagnostics:
            logger.warning("AdaptiveGridSearch.search_diagnostics is empty; nothing to save")
            return

        fieldnames = list(self.search_diagnostics[0].keys())
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.search_diagnostics)
        logger.info(f"Saved {len(self.search_diagnostics)} search diagnostic rows to {file_path}")

    def save_evaluated_points(self, file_path):
        if not self.all_evaluated_points:
            logger.warning("AdaptiveGridSearch.all_evaluated_points is empty; nothing to save")
            return

        points = np.array(self.all_evaluated_points, dtype=float)
        np.savez(
            file_path,
            r_values=points[:, 0],
            usage_values=points[:, 1],
            acc_values=points[:, 2],
            rerun_values=points[:, 3]
        )
        logger.info(f"Saved {len(self.all_evaluated_points)} (r, usage, acc, rerun) points to {file_path}")

    def save_routing_snapshots(self, file_path):
        if not self.routing_snapshots:
            logger.warning("AdaptiveGridSearch.routing_snapshots is empty; nothing to save")
            return

        # lf_latent/latent_2d/boundary_xx/boundary_yy are identical for every
        # snapshot (same fixed test-set lf_latent - already pooled to (N, C) by
        # compute_gate_routing_details regardless of gate type, same cached
        # projection basis/grid), so they're saved once rather than duplicated
        # per snapshot; only boundary_zz (the actual per-gate-model decision
        # field) legitimately varies.
        lf_latent_flat = self.routing_snapshots[0]["lf_latent"]
        latent_2d = (lf_latent_flat - self.proj_mean) @ self.proj_basis

        np.savez(
            file_path,
            rerun=np.array([s["rerun"] for s in self.routing_snapshots]),
            target_usage=np.array([s["target_usage"] for s in self.routing_snapshots]),
            lf_latent=self.routing_snapshots[0]["lf_latent"],
            lf_correct=np.stack([s["lf_correct"] for s in self.routing_snapshots], axis=0),
            hf_correct=np.stack([s["hf_correct"] for s in self.routing_snapshots], axis=0),
            choice=np.stack([s["choice"] for s in self.routing_snapshots], axis=0),
            label=np.stack([s["label"] for s in self.routing_snapshots], axis=0),
            latent_2d=latent_2d,
            boundary_xx=self.boundary_xx,
            boundary_yy=self.boundary_yy,
            boundary_zz=np.stack([s["boundary_zz"] for s in self.routing_snapshots], axis=0)
        )
        logger.info(f"Saved {len(self.routing_snapshots)} routing snapshots to {file_path}")

    def evaluate_fe_model(self, r_val):
        fe_model_file = os.path.join(self.model_folder, f"fe_model-{r_val}.pt")
        self.fe_model.load_state_dict(torch.load(fe_model_file, weights_only=True))
        self.fe_model = self.fe_model.to(self.device)

        # Single forward pass over the test set gets us test_acc/test_use *and*
        # everything the routing/projection plots need, in one shot — done here (right
        # after loading this r_val's just-trained weights) rather than reloading
        # the checkpoint later, since fe_model-{r_val}.pt gets overwritten by the
        # next rerun that happens to land on the same r_val.
        details = compute_gate_routing_details(self.fe_model, self.test_dl, self.device)

        correct = np.where(details["choice"] == 1, details["hf_correct"], details["lf_correct"])
        test_acc = float(correct.mean())
        test_use = float(details["choice"].mean())

        # Decision-boundary field for the routing plot: fits/caches the
        # projection plane on first call (using this call's ground truth to
        # pick a separating basis), then runs *this* r_val's actual fe_model
        # (still loaded above) over the plane's grid, reconstructed back to
        # latent space — so the contour reflects the real model, not a re-fit
        # proxy.
        hf_needed = details["hf_correct"] & ~details["lf_correct"]
        self._project_latent(details["lf_latent"], details["latent_sample_shape"], hf_needed)
        details["boundary_zz"] = self._compute_decision_boundary()

        return test_acc, test_use, details

    def _cached_eval(self, r_val, tol=1e-6):
        """Looks up an r_val already trained earlier in this rerun (e.g. every
        full-range fallback in find_bracket starts bisection from the same
        deterministic first midpoint, 0.5005 - without this, every target that
        falls back would silently retrain that same r_val from scratch)."""
        for r, use, acc in self.evaluated_points:
            if abs(r - r_val) <= tol:
                return use, acc
        return None

    def find_r_for_target(self, usage, tolerance=1e-3):
        r_low, r_high, bracket_fallback_used = self.find_bracket(usage)
        bisection_passes = 0
        while r_high - r_low > tolerance:
            bisection_passes += 1
            r_mid = (r_low + r_high) / 2

            cached = self._cached_eval(r_mid)
            if cached is not None:
                use_mid, acc_mid = cached
            else:
                use_mid, acc_mid = self.train_fe_model(r_mid)
                self.evaluated_points.append((r_mid, use_mid, acc_mid))
                self.all_evaluated_points.append((r_mid, use_mid, acc_mid, self.rerun_idx))

            if use_mid > usage:
                r_low = r_mid
            else:
                r_high = r_mid

        test_acc, test_use, gate_details = self.evaluate_fe_model(r_high)
        test_acc *= 100
        test_use *= 100

        self.routing_snapshots.append({
            "rerun": self.rerun_idx,
            "target_usage": usage,
            **gate_details
        })

        self.search_diagnostics.append({
            "rerun": self.rerun_idx, "target_usage": usage, "bisection_passes": bisection_passes,
            "bracket_fallback_used": bracket_fallback_used,
            "final_r": r_high, "final_test_usage": test_use, "final_test_acc": test_acc
        })

        logger.info(f"\nTook {bisection_passes} passes to find best r value")
        logger.info(f"For usage {usage}:")
        logger.info(f"\tBest r: {r_high}")
        logger.info(f"\tClosest val usage: {self.evaluated_points[-1][1]:.4f}")
        logger.info(f"\tTest usage: {test_use:.2f} / Test acc: {test_acc:.2f}")

        return r_high, test_acc, test_use

    def run_reruns(self, usage_values, n_reruns):
        """Runs the full usage_values sweep n_reruns times, resetting the live
        bracket-search history (evaluated_points) at the start of each rerun so
        every rerun is a genuinely independent search — not warm-started off a
        previous rerun's bisection results, which would understate the true
        run-to-run variance. epoch_log/search_diagnostics/all_evaluated_points
        keep accumulating across every rerun (each row tagged via self.rerun_idx)
        for full traceability.

        Returns (usage_runs, acc_runs), each shape (n_reruns, len(usage_values)),
        already on the 0-100 scale find_r_for_target reports.
        """
        usage_runs = np.zeros((n_reruns, len(usage_values)))
        acc_runs = np.zeros((n_reruns, len(usage_values)))

        for rerun_idx in range(n_reruns):
            self.rerun_idx = rerun_idx
            self.evaluated_points = []

            # Fresh random init + seed per rerun (so reruns still capture
            # genuine run-to-run variance), but every r_val's search within
            # this rerun trains from this same snapshot/seed (see __init__).
            self.rerun_seed = self.seed + rerun_idx
            set_all_seeds(self.rerun_seed)
            reset_all_weights(self.fe_model)
            self.fe_init_state = {k: v.clone() for k, v in self.fe_model.state_dict().items()}

            for target_idx, target_usage in enumerate(usage_values):
                _, test_acc, test_use = self.find_r_for_target(usage=target_usage)
                usage_runs[rerun_idx, target_idx] = test_use
                acc_runs[rerun_idx, target_idx] = test_acc

        return usage_runs, acc_runs

    def _gate_scores(self, dataloader):
        """One pass over a gate-fidelity dataloader, returning per-sample
        arrays: self.fe_model's own softmax confidence that HF is needed
        (p_hf), and each fidelity's own correctness (lf_correct/hf_correct).
        Doesn't apply a hard routing decision itself - run_fe_sr_grid sweeps
        that post-hoc over a threshold grid, which is what makes a fine
        threshold grid cheap: one forward pass per r checkpoint here, not one
        per (r, threshold) cell.
        """
        self.fe_model.eval()
        p_hf_list, lf_correct_list, hf_correct_list = [], [], []

        with torch.no_grad():
            for lf_embeddings, lf_preds, hf_preds, target in dataloader:
                lf_embeddings = lf_embeddings.to(self.device, torch.float)
                lf_preds = lf_preds.to(self.device, torch.float)
                hf_preds = hf_preds.to(self.device, torch.float)
                target = target.long().to(self.device)

                outputs = self.fe_model(lf_embeddings)["output"]
                p_hf = torch.softmax(outputs, dim=1)[:, 1]

                p_hf_list.append(p_hf.cpu().numpy())
                lf_correct_list.append((torch.argmax(lf_preds, dim=1) == target).cpu().numpy())
                hf_correct_list.append((torch.argmax(hf_preds, dim=1) == target).cpu().numpy())

        return (
            np.concatenate(p_hf_list, axis=0),
            np.concatenate(lf_correct_list, axis=0),
            np.concatenate(hf_correct_list, axis=0),
        )

    def run_fe_sr_grid(self, usage_values, threshold_grid=None):
        """"FE model + softmax response": builds a (r, threshold) usage/accuracy
        grid by reusing the already-trained fe_model-{r}.pt checkpoints this
        search saved along the way (one per r value it explored - see
        train_fe_model), the same way SelectiveNet/SAT sweep a (c, threshold)
        grid in main.py. "Just the FE model" (fe_results.npz, from run_reruns)
        uses each r's gate with its own hard argmax routing decision; this
        reuses those SAME checkpoints but replaces the hard argmax with the
        gate's own continuous confidence that HF is needed (self._gate_scores'
        p_hf), swept over a threshold grid. Sweeping is cheap - one forward
        pass per r (not per (r, threshold) cell), since usage/accuracy at every
        threshold can be computed in one vectorized pass over already-collected
        per-sample scores - so this can afford a much finer grid than r alone.

        For each target usage, picks the (r, threshold) cell whose val usage is
        <= target with the *highest val accuracy* (not necessarily the highest
        usage under budget, unlike the SelectiveNet/SAT selection in main.py -
        usage(r) has repeatedly turned out non-monotonic enough here that
        "more usage" isn't a safe proxy for "more accurate").

        Returns a dict with the selected per-target-usage results plus the
        full grids, for main.py to save to fe_sr_results.npz/fe_sr_grid.npz.
        """
        if threshold_grid is None:
            threshold_grid = np.linspace(0.0, 1.0, 21)
        threshold_grid = np.asarray(threshold_grid)

        r_files = sorted(
            f for f in os.listdir(self.model_folder)
            if f.startswith("fe_model-") and f.endswith(".pt")
        )
        r_values = [float(f[len("fe_model-"):-len(".pt")]) for f in r_files]

        n_r, n_t = len(r_values), len(threshold_grid)
        val_usage_grid = np.zeros((n_r, n_t))
        val_acc_grid = np.zeros((n_r, n_t))
        test_usage_grid = np.zeros((n_r, n_t))
        test_acc_grid = np.zeros((n_r, n_t))

        for r_idx, fname in enumerate(r_files):
            self.fe_model.load_state_dict(
                torch.load(os.path.join(self.model_folder, fname), weights_only=True)
            )
            self.fe_model = self.fe_model.to(self.device)

            val_p_hf, val_lf_correct, val_hf_correct = self._gate_scores(self.val_dl)
            test_p_hf, test_lf_correct, test_hf_correct = self._gate_scores(self.test_dl)

            # decision[i, j] = route sample i to HF at threshold_grid[j]
            val_decision = val_p_hf[:, None] >= threshold_grid[None, :]
            test_decision = test_p_hf[:, None] >= threshold_grid[None, :]

            val_usage_grid[r_idx] = val_decision.mean(axis=0)
            val_acc_grid[r_idx] = np.where(
                val_decision, val_hf_correct[:, None], val_lf_correct[:, None]
            ).mean(axis=0)
            test_usage_grid[r_idx] = test_decision.mean(axis=0)
            test_acc_grid[r_idx] = np.where(
                test_decision, test_hf_correct[:, None], test_lf_correct[:, None]
            ).mean(axis=0)

        fe_sr_usage_vals, fe_sr_acc_vals, chosen_r, chosen_threshold = [], [], [], []
        flat_val_usage = val_usage_grid.ravel()
        flat_val_acc = val_acc_grid.ravel()

        for target_usage in usage_values:
            under_budget = np.where(flat_val_usage <= target_usage)[0]

            if under_budget.size > 0:
                best_flat_idx = under_budget[np.argmax(flat_val_acc[under_budget])]
            else:
                logger.warning(
                    f"No (r, threshold) combo has val usage <= {target_usage}; "
                    f"falling back to the closest val usage overall"
                )
                best_flat_idx = np.argmin(np.abs(flat_val_usage - target_usage))

            r_idx, t_idx = np.unravel_index(best_flat_idx, val_usage_grid.shape)

            fe_sr_usage_vals.append(100 * test_usage_grid[r_idx, t_idx])
            fe_sr_acc_vals.append(100 * test_acc_grid[r_idx, t_idx])
            chosen_r.append(r_values[r_idx])
            chosen_threshold.append(threshold_grid[t_idx])

        return {
            "usage": np.array(fe_sr_usage_vals),
            "acc": np.array(fe_sr_acc_vals),
            "target_usage": np.array(usage_values),
            "chosen_r": np.array(chosen_r),
            "chosen_threshold": np.array(chosen_threshold),
            "r_grid": np.array(r_values),
            "threshold_grid": threshold_grid,
            "val_usage_grid": val_usage_grid,
            "val_acc_grid": val_acc_grid,
            "test_usage_grid": test_usage_grid,
            "test_acc_grid": test_acc_grid,
        }

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
    