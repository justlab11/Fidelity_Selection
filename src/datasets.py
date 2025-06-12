import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, Subset
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
from torch.utils.data import Dataset

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

class DualFidelityDataset:
    def __init__(self, config: Options, data_folder="./data", 
                 val_split=0.5, random_seed=42):
        
        self.config = config
        self.dataset_name = config.dataset.name.lower()
        self.augmentation = config.dataset.augmentation.lower()
        self.augmentation_level = config.dataset.augmentation_level

        self.folder = config.dataset.folder

        self.val_split = val_split
        self.random_seed = random_seed

        valid_dataset_names = ["mnist", "cifar10", "cifar100"]
        valid_aug_names = ["noise", "blur", "random_rotation"]

        if self.dataset_name not in valid_dataset_names:
            valid_dataset_str = ", ".join(valid_dataset_names) 
            raise ValueError(f"Unsupported dataset. Choose {valid_dataset_str}.")
        
        if self.augmentation not in valid_aug_names:
            valid_aug_str = ", ".join(valid_aug_names) 
            raise ValueError(f"Unsupported augmentation. Choose {valid_aug_str}.")
        
        # Set random seeds
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Load the train and test datasets
        if self.dataset_name == 'mnist':
            self.train_dataset = datasets.MNIST(root=data_folder, train=True, download=True)
            self.test_dataset = datasets.MNIST(root=data_folder, train=False, download=True)

        elif self.dataset_name == 'cifar10':
            self.train_dataset = datasets.CIFAR10(root=data_folder, train=True, download=True)
            self.test_dataset = datasets.CIFAR10(root=data_folder, train=False, download=True)

        elif self.dataset_name == 'cifar100':
            self.train_dataset = datasets.CIFAR100(root=data_folder, train=True, download=True)
            self.test_dataset = datasets.CIFAR100(root=data_folder, train=False, download=True)

        elif self.dataset_name == 'crop':
            all_train_files = glob.glob(
                path.join(self.folder, "training_chips/*")
            )
            train_files = [file for file in all_train_files if not path.basename(file).startswith(".")]

            all_val_files = glob.glob(
                path.join(self.folder, "validation_chips/*")
            )
            val_files = [file for file in all_val_files if not path.basename(file).startswith(".")]

            self.train_dataset = CropClassificationDataset(train_files)
            self.test_dataset = CropClassificationDataset(val_files)            
            
                
        # Split test set into test and validation sets
        test_indices = list(range(len(self.test_dataset)))
        val_indices, test_indices = train_test_split(
            test_indices,
            test_size=self.val_split,
            random_state=self.random_seed,
            stratify=self.test_dataset.targets
        )
        
        # Create datasets
        self.train_data = self.train_dataset
        self.val_data = torch.utils.data.Subset(self.test_dataset, val_indices)
        self.test_data = torch.utils.data.Subset(self.test_dataset, test_indices)

    def train(self):
        return AugmentedDataset(self.train_data, self.config)

    def val(self):
        return AugmentedDataset(self.val_data, self.config)

    def test(self):
        return AugmentedDataset(self.test_data, self.config)


class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, config: Options):
        self.base_dataset = base_dataset

        self.hf_aug_info = config.dataset.augmentations.high_fidelity
        self.lf_aug_info = config.dataset.augmentations.low_fidelity

        # ResNet18 preprocessing
        weights = ResNet18_Weights.DEFAULT
        if config.dataset.name in ["mnist", "cifar10", "cifar100"]: 
            self.preprocess = weights.transforms()
        else:
            self.preprocess = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # Convert to PIL Image if necessary
        if not isinstance(image, Image.Image):
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            else:
                image = Image.fromarray(np.uint8(image))
        
        # Convert grayscale to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply augmentation
        if self.augmentation == 'noise':
            augmented_image = self.add_noise(image)
        elif self.augmentation == 'blur':
            augmented_image = self.apply_blur(image)
        elif self.augmentation == 'random_rotation':
            augmented_image = self.random_rotate(image)
        
        # Apply ResNet18 preprocessing
        original_image = self.preprocess(image)
        augmented_image = self.preprocess(augmented_image)
        
        return augmented_image, original_image, label

    def add_noise(self, image):
        np_image = np.array(image)
        noise = np.random.normal(0, self.degree * 255, np_image.shape)
        noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def apply_blur(self, image):
        return image.filter(ImageFilter.GaussianBlur(radius=self.degree))

    def random_rotate(self, image):
        angle = np.random.uniform(-self.degree, self.degree)
        return image.rotate(angle)


class CropClassificationDataset(Dataset):
    def __init__(self, files, num_subsamples=8):
        self.num_subsamples = num_subsamples
        base_names = set()

        # First pass: Collect all base names
        for filename in files:
            if filename.endswith('_merged.tif'):
                base_name = filename[:-11]  # Remove '_merged.tif'
                base_names.add(base_name)
            elif filename.endswith('.mask.tif'):
                base_name = filename[:-9]  # Remove '.mask.tif'
                base_names.add(base_name)

        self.samples = []

        for base_name in base_names:
            img_name = base_name + "_merged.tif"
            mask_name = base_name + ".mask.tif"
            if not path.exists(img_name):
                raise FileNotFoundError(f"No file named {img_name}")
            
            if not path.exists(mask_name):
                raise FileNotFoundError(f"No file named {mask_name}")
            
            # Pre-define subsections and layouts for each sample
            for _ in range(self.num_subsamples * 3):
                layout = np.random.randint(0, 3)
                subsection = np.random.randint(0, 224 - 56 + 1, 2)
                self.samples.append({
                    'img_file': img_name,
                    'mask_file': mask_name,
                    'layout': layout,
                    'subsection': subsection
                })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_file = sample['img_file']
        mask_file = sample['mask_file']
        layout = sample['layout']
        subsection = sample['subsection']

        img = tif.imread(img_file).reshape(224, 224, 6, 3)
        mask = tif.imread(mask_file)

        img = img[subsection[0]:subsection[0]+56, subsection[1]:subsection[1]+56, :, layout]
        mask = mask[subsection[0]:subsection[0]+56, subsection[1]:subsection[1]+56]

        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        img /= np.max(img)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        return img, mask