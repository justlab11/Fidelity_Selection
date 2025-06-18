import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, random_split
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights
from custom_types import Options
import glob
import os.path as path
import tifffile as tif
import cv2

import numpy as np
import torch
from datasets import load_dataset

class HypercubeDataset(Dataset):
    def __init__(
        self,
        num_dims: int,
        num_samples: np.ndarray,  # shape: (num_clusters,)
        hf_std: np.ndarray,       # shape: (num_clusters,)
        lf_std: np.ndarray,       # shape: (num_clusters,)
        group_classes: bool=True
    ):
        self.num_dims = num_dims
        self.num_clusters = 2 ** num_dims
        self.num_classes = self.num_clusters // 2 if group_classes else self.num_clusters
        self.group_classes = group_classes

        # Insert zero at the start for your indexing scheme
        num_samples = np.insert(num_samples, 0, 0)

        self.hf_points, self.labels = self._generate_points_and_labels(
            num_dims, self.num_classes, 
            num_samples, hf_std, group_classes
        )

        self.lf_points, _ = self._generate_points_and_labels(
            num_dims, self.num_classes,
            num_samples, lf_std, group_classes
        )

        self.hf_points = torch.from_numpy(self.hf_points.astype(np.float32))
        self.lf_points = torch.from_numpy(self.lf_points.astype(np.float32))
        self.labels = torch.from_numpy(self.labels.astype(np.int64))

    def __len__(self):
        return len(self.lf_points)

    def __getitem__(self, idx):
        return self.lf_points[idx], self.hf_points[idx], self.labels[idx]

    def _hypercube_corners(self, num_dims):
        corners = np.zeros((2 ** num_dims, num_dims))
        for i in range(2 ** num_dims):
            binary = np.binary_repr(i, width=num_dims)
            for j in range(num_dims):
                corners[i, j] = int(binary[j])
        return corners

    def _get_label_set(self, num_classes, group_classes):
        label_set = [i for i in range(num_classes)]
        if group_classes:
            tmp = list(reversed(label_set))
            label_set += tmp
        return label_set

    def get_num_classes(self):
        return self.num_classes

    def _generate_points_and_labels(
        self, num_dims, num_classes,
        num_samples_per_cluster, stds, group_classes
    ):
        means = self._hypercube_corners(num_dims)
        label_set = self._get_label_set(num_classes, group_classes)

        total_samples = num_samples_per_cluster.sum().item()
        points = np.zeros((total_samples, num_dims))
        labels = np.zeros(total_samples)

        for i in range(len(num_samples_per_cluster) - 1):
            start = num_samples_per_cluster[:i + 1].sum()
            end = num_samples_per_cluster[:i + 2].sum()

            points[start:end] = np.random.normal(
                means[i], stds[i],
                size=(num_samples_per_cluster[i + 1], num_dims)
            )
            labels[start:end] = label_set[i]

        return points, labels
    
class MNISTDataset(Dataset):
    def __init__(
        self,
        split: str,
        root: str = "../data",
        hf_transform=None,
        lf_transform=None,
        val_ratio: float = 0.5,
        seed: int = 42
    ):
        assert split in ["train", "test", "val"], "split must be 'train', 'test', or 'val'"

        if split == "test":
            base_dataset = datasets.MNIST(
                root=root, train=False, transform=None, download=True
            )
        else:
            base_dataset = datasets.MNIST(
                root=root, train=True, transform=None, download=True
            )

        if split in ["train", "val"]:
            # Split the test set deterministically
            n = len(base_dataset)
            val_size = int(n * val_ratio)
            train_size = n - val_size
            generator = torch.Generator().manual_seed(seed)
            train_set, val_set = random_split(base_dataset, [train_size, val_size], generator=generator)
            if split == "train":
                base_dataset = train_set
            else:
                base_dataset = val_set

        # Now, create two datasets with different transforms but the same indices
        self.hf_transform = hf_transform
        self.lf_transform = lf_transform
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]

        hf_img = self.hf_transform(img) if self.hf_transform else img
        lf_img = self.lf_transform(img) if self.lf_transform else img

        return lf_img, hf_img, target
    
class CropDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        seed: int = 42,
        test_ratio: float = 0.2
    ):
        assert split in ["train", "test", "val"], "split must be 'train', 'test', or 'val'"

        self.fileset = []
        self.generator = torch.Generator().manual_seed(seed)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts (H, W, C) numpy to (C, H, W) tensor
            transforms.Resize((224, 224)),  # Resize to model input size
            transforms.Normalize(
                mean=[496.38, 816.70, 927.55, 2961.28, 2638.13, 1742.81],
                std=[286.37, 359.40, 577.06, 897.17, 954.61, 922.32]
            ),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts (H, W) to (1, H, W)
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Lambda(lambda x: x.squeeze(0).long())  # Remove channel dim and ensure integer type
        ])

        if split == "val":
            val_folder = path.join(root, "multi-temporal-crop-classification", "validation_data.txt")
            with open(val_folder, "r") as f:
                val_chips = f.readlines()

            val_filenames = [
                path.join(
                    root, 
                    "multi-temporal-crop-classification", 
                    "validation_chips", line.strip()
                ) for line in val_chips
            ]

            for i in indices:
                for t in range(3): # timestep 
                    for q in range(4): # quandrant of the image
                        self.fileset.append((val_filenames[i], t, q))

        else:
            train_folder = path.join(root, "multi-temporal-crop-classification", "training_data.txt")
            with open(train_folder, "r") as f:
                train_chips = f.readlines()

            train_test_filenames = [
                path.join(
                    root, 
                    "multi-temporal-crop-classification", 
                    "training_chips", line.strip()
                ) for line in train_chips
            ]

            n = len(train_test_filenames)
            test_size = int(n * test_ratio)
            train_size = n - test_size

            train_set, test_set = random_split(train_test_filenames, [train_size, test_size], generator=self.generator)
            if split == "train":
                indices = train_set.indices
            else:
                indices = test_set.indices

            for i in indices:
                for t in range(3): # timestep 
                    for q in range(4): # quandrant of the image
                        self.fileset.append((train_test_filenames[i], t, q))

    def __len__(self):
        return len(self.fileset)
    
    def __getitem__(self, idx:int):
        base_fname, timestep = self.fileset[idx]
        mask_fname = base_fname + ".mask.tif"
        image_fname = base_fname + "_merged.tif"

        mask = tif.imread(mask_fname)
        image_set = tif.imread(image_fname)

        x_start = torch.randint(0, 224 - 56 + 1, (1,), generator=self.generator).item()
        x_end = x_start + 56

        y_start = torch.randint(0, 224 - 56 + 1, (1,), generator=self.generator).item()
        y_end = y_start + 56

        hf_image = image_set[x_start:x_end, y_start:y_end, 6*timestep:6*(timestep+1)]
        hf_image = self.img_transform(hf_image)

        lf_image = hf_image[:, :, :3]

        mask = mask[x_start:x_end, y_start:y_end]
        mask = self.mask_transform(mask)

        return hf_image, lf_image, mask
    
        
class QE_Dataset(Dataset):
    def __init__(self, lf_embeddings, lf_preds, hf_preds, labels):
        super(QE_Dataset, self).__init__()

        self.lf_embeddings = lf_embeddings
        self.lf_preds = lf_preds
        self.hf_preds = hf_preds
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx:int):
        return self.lf_embeddings[idx], self.lf_preds[idx], self.hf_preds[idx], self.labels[idx]