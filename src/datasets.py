import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, random_split
from PIL import Image
import os.path as path
import tifffile as tif
import os
import warnings

import numpy as np
import torch

from torch.utils.data import Dataset
import numpy as np
import torch

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
        
        # After generating hf_points and labels
        lf_points = np.zeros_like(self.hf_points)

        for i in range(self.hf_points.shape[0]):
            cluster = int(self.labels[i])

            hf_std_val = hf_std[cluster]
            lf_std_val = lf_std[cluster]

            # Compute the extra std needed
            extra_std = np.sqrt(lf_std_val**2 - hf_std_val**2)

            # Add extra noise to hf_point
            lf_points[i] = self.hf_points[i] + np.random.normal(0, extra_std, size=self.num_dims)

        self.lf_points = torch.from_numpy(lf_points.astype(np.float32)) 

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
    
    def get_lf_input_size(self):
        return self.num_dims
    
    def get_hf_input_size(self):
        return self.num_dims
    
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

    def get_num_classes(self):
        return 10
    
    def get_lf_input_size(self):
        return 3
    
    def get_hf_input_size(self):
        return 3
    
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
        test_ratio: float = 0.2,
    ):
        assert split in ["train", "test", "val"], "split must be 'train', 'test', or 'val'"

        self.fileset = []
        self.generator = torch.Generator().manual_seed(seed)
        self.split = split

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts (H, W, C) numpy to (C, H, W) tensor
            transforms.Resize((224, 224)),  # Resize to model input size
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x).long().unsqueeze(0)),  # uint8→long DIRECTLY
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Lambda(lambda x: x.squeeze(0))
        ])

        self.mean = 0
        self.std = 0

        if split == "val":
            val_folder = path.join(root, "multi-temporal-crop-classification", "validation_data.txt")
            with open(val_folder, "r") as f:
                val_chips = f.readlines()

            val_filenames = [
                path.join(
                    root, 
                    "multi-temporal-crop-classification", 
                    "validation_chips", "validation_chips", line.strip()
                ) for line in val_chips
            ]

            for i in range(len(val_filenames)):
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
                    "training_chips", "training_chips", line.strip()
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
            
            if split == "train":
                self._compute_dataset_stats()

    def get_num_classes(self):
        return 14

    def get_lf_input_size(self):
        return 3
    
    def get_hf_input_size(self):
        return 6

    def __len__(self):
        return len(self.fileset)
    
    def _compute_dataset_stats(self):
        """Compute global mean/std from unique chips in this split only"""
        unique_chips = set(f[0] for f in self.fileset)  # Remove timestep/quadrant duplicates
        
        all_data = []
        for chip_path in unique_chips:  # Just one sample per unique chip
            image_fname = chip_path + "_merged.tif"
            image_set = tif.imread(image_fname)
            all_data.append(image_set)
        
        all_data = np.stack(all_data, axis=0)  # [N_chips, H, W, 18]
        self.mean = all_data.mean(axis=(0,1,2))    # [18]
        self.std = all_data.std(axis=(0,1,2)) + 1e-8  # [18]
        
    def __getitem__(self, idx:int):
        base_fname, timestep, _ = self.fileset[idx]
        mask_fname = base_fname + ".mask.tif"
        image_fname = base_fname + "_merged.tif"

        mask = tif.imread(mask_fname)
        image_set = tif.imread(image_fname)

        if self.mean is not None:
            image_set = (image_set - self.mean[None, None, :]) / self.std[None, None, :]
        else:
            warnings.warn("Please set self.mean and self.std to the values from the train set")

        if self.split == "test":
            hf_image = image_set[:56, :56, 6*timestep:6*(timestep+1)]
        else:
            x_start = torch.randint(0, 224 - 56 + 1, (1,), generator=self.generator).item()
            x_end = x_start + 56

            y_start = torch.randint(0, 224 - 56 + 1, (1,), generator=self.generator).item()
            y_end = y_start + 56

            hf_image = image_set[x_start:x_end, y_start:y_end, 6*timestep:6*(timestep+1)]

        hf_image[:,:,:3] = hf_image[:, :, [2, 1, 0]]
        hf_image = self.img_transform(hf_image)

        lf_image = hf_image[:3]
        hf_image = hf_image[3:] # we concatenate later in the code so we split here

        mask = mask[x_start:x_end, y_start:y_end]
        mask = self.mask_transform(mask)

        return lf_image, hf_image, mask

class CropDatasetOld(Dataset):
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
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x).long().unsqueeze(0)),  # uint8→long DIRECTLY
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Lambda(lambda x: x.squeeze(0))
        ])

        if split == "val":
            val_folder = path.join(root, "multi-temporal-crop-classification", "validation_data.txt")
            with open(val_folder, "r") as f:
                val_chips = f.readlines()

            val_filenames = [
                path.join(
                    root, 
                    "multi-temporal-crop-classification", 
                    "validation_chips", "validation_chips", line.strip()
                ) for line in val_chips
            ]

            for i in range(len(val_filenames)):
                for t in range(3): # timestep 
                    for q in range(4): # quandrant of the image
                        self.fileset.append((val_filenames[i], t, q))
            
            self._compute_dataset_stats()

        else:
            train_folder = path.join(root, "multi-temporal-crop-classification", "training_data.txt")
            with open(train_folder, "r") as f:
                train_chips = f.readlines()

            train_test_filenames = [
                path.join(
                    root, 
                    "multi-temporal-crop-classification", 
                    "training_chips", "training_chips", line.strip()
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
            
            self._compute_dataset_stats()

    def get_num_classes(self):
        return 14

    def get_lf_input_size(self):
        return 3
    
    def get_hf_input_size(self):
        return 6

    def __len__(self):
        return len(self.fileset)
    
    def _compute_dataset_stats(self):
        """Compute global mean/std from unique chips in this split only"""
        unique_chips = set(f[0] for f in self.fileset)  # Remove timestep/quadrant duplicates
        
        all_data = []
        for chip_path in unique_chips:  # Just one sample per unique chip
            image_fname = chip_path + "_merged.tif"
            image_set = tif.imread(image_fname)
            all_data.append(image_set)
        
        all_data = np.stack(all_data, axis=0)  # [N_chips, H, W, 18]
        self.mean = all_data.mean(axis=(0,1,2))    # [18]
        self.std = all_data.std(axis=(0,1,2)) + 1e-8  # [18]
        
    def __getitem__(self, idx:int):
        base_fname, timestep, quadrant = self.fileset[idx]
        mask_fname = base_fname + ".mask.tif"
        image_fname = base_fname + "_merged.tif"

        mask = tif.imread(mask_fname)
        image_set = tif.imread(image_fname)

        if self.mean is not None:
            image_set = (image_set - self.mean[None, None, :]) / self.std[None, None, :]

        q_x, q_y = (quadrant // 2) * 112, (quadrant % 2) * 112
        hf_image = image_set[q_x:q_x+112, q_y:q_y+112, 6*timestep:6*(timestep+1)]
        hf_image[:,:,:3] = hf_image[:, :, [2, 1, 0]]
        hf_image = self.img_transform(hf_image)

        lf_image = hf_image[:3]
        hf_image = hf_image[3:] # we concatenate later in the code so we split here

        mask = mask[q_x:q_x+112, q_y:q_y+112]
        mask = self.mask_transform(mask)

        return lf_image, hf_image, mask
    
class CUBDataset(Dataset):
    def __init__(
            self, 
            root: str, 
            split: str, 
            seed:int=42,
            grayscale=True
    ):
        assert split in ["train", "test", "val"], "split must be 'train', 'test', or 'val'"

        self.root = root
        self.split = split
        self.generator = torch.Generator().manual_seed(seed)
        self.grayscale = grayscale
        
        if self.grayscale:
            self.lf_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert image to grayscale with 3 channels
                transforms.ToTensor(),  # Converts (H, W, C) numpy to (C, H, W) tensor
                transforms.Resize((224, 224)),  # Resize to model input size
            ])

        self.hf_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts (H, W, C) numpy to (C, H, W) tensor
            transforms.Resize((224, 224)),  # Resize to model input size
        ])

        image_split = self.get_split()

        if split != "test":
            num_images = len(image_split)
            val_ratio = .1
            val_size = int(val_ratio * num_images)
            train_size = num_images - val_size

            train_set, val_set = random_split(image_split, [train_size, val_size], generator=self.generator)
            indices = train_set.indices if split == "train" else val_set.indices
            self.data = [image_split[i] for i in indices]

        else:
            self.data = image_split

    def get_split(self):
        splits_file = os.path.join(self.root, "train_test_split.txt")
        image_id_file = os.path.join(self.root, "images.txt")

        splits = np.loadtxt(splits_file, dtype=int)
        image_ids = np.loadtxt(image_id_file, dtype=str)

        image_ids_dict = {
            int(image_ids[i, 0]): str(image_ids[i, 1]) 
            for i in range(len(image_ids))
        }

        image_split_dict = {
            image_ids_dict[splits[i, 0]]: int(splits[i, 1]) 
            for i in range(len(splits))
        }

        # for this dataset 1 = train and 0 = test
        # split_idx is 0 if "test", but 1 if "train" or "val" since we build val from train
        split_idx = 0 if self.split=="test" else 1

        images_for_split = [key for key in image_split_dict.keys() if image_split_dict[key]==split_idx]

        return images_for_split

    def get_num_classes(self):
        return 200

    def get_lf_input_size(self):
        return 3
    
    def get_lf_input_size(self):
        return 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        local_fname = self.data[idx]
        # example: 045.Northern_Fulmar/Northern_Fulmar_0010_44112.jpg

        metadata = local_fname.split(".")
        # example: [045, Northern_Fulmar/Northern_Fulmar_0010_44112, jpg]

        label = int(metadata[0])-1 # 045 -> labeled as class 44 because 1 indexing
        fname = os.path.join(self.root, "images", local_fname)

        image = Image.open(fname).convert("RGB")

        hf_img = self.hf_transform(image)
        if self.grayscale:
            lf_img = self.lf_transform(image)
        else:
            lf_img = hf_img.clone()

        return lf_img, hf_img, label
        
class BodyDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = sorted(os.listdir(folder_path))  # List all .pt files sorted

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        data = torch.load(file_path)

        lf_body_output = data['lf_body_output']
        hf_body_output = data['hf_body_output']
        label = data['label']

        return lf_body_output, hf_body_output, label

class FE_Dataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = sorted(os.listdir(folder_path))  # Sorted list of .pt file names

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        data = torch.load(file_path)

        lf_latent = data['lf_latent']
        lf_output = data['lf_output']
        hf_output = data['hf_output']
        label = data['label']

        return lf_latent, lf_output, hf_output, label
    
