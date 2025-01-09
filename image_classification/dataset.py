import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from custom_types import ImageClassificationConfig
from torchvision import datasets, transforms
import glob
import os.path as path
from PIL import Image, ImageFilter
from torchvision.models import ResNet18_Weights

class DualFidelityDataset:
    def __init__(self, config: ImageClassificationConfig, 
                 val_split=0.5, random_seed=42):
        
        self.config = config
        self.dataset_name = config.dataset.name.lower()
        data_folder = config.dataset.folder

        self.folder = config.dataset.folder

        self.val_split = val_split
        self.random_seed = random_seed

        valid_dataset_names = ["mnist", "cifar10", "cifar100"]

        if self.dataset_name not in valid_dataset_names:
            valid_dataset_str = ", ".join(valid_dataset_names) 
            raise ValueError(f"Unsupported dataset. Choose {valid_dataset_str}.")
        
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
    def __init__(self, base_dataset, config: ImageClassificationConfig):
        self.base_dataset = base_dataset

        self.hf_aug_info = config.dataset.augmentations["high_fidelity"]
        self.lf_aug_info = config.dataset.augmentations["low_fidelity"]

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
        
        if self.hf_aug_info.augmentation == 'noise':
            original_image = self.add_noise(image, self.hf_aug_info.strength)
        elif self.hf_aug_info.augmentation == 'blur':
            original_image = self.apply_blur(image, self.hf_aug_info.strength)
        elif self.hf_aug_info.augmentation == 'random_rotation':
            original_image = self.random_rotate(image, self.hf_aug_info.strength)
        else:
            original_image = image

        if self.lf_aug_info.augmentation == 'noise':
            augmented_image = self.add_noise(image, self.lf_aug_info.strength)
        elif self.lf_aug_info.augmentation == 'blur':
            augmented_image = self.apply_blur(image, self.lf_aug_info.strength)
        elif self.lf_aug_info.augmentation == 'random_rotation':
            augmented_image = self.random_rotate(image, self.lf_aug_info.strength)
        else:
            augmented_image = image
        
        # Apply ResNet18 preprocessing
        original_image = self.preprocess(original_image)
        augmented_image = self.preprocess(augmented_image)
        
        return augmented_image, original_image, label

    def add_noise(self, image, degree):
        np_image = np.array(image)
        noise = np.random.normal(0, degree * 255, np_image.shape)
        noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def apply_blur(self, image, degree):
        return image.filter(ImageFilter.GaussianBlur(radius=degree))

    def random_rotate(self, image, degree):
        angle = np.random.uniform(-degree, degree)
        return image.rotate(angle)

# class FidelityDataset(Dataset):
#     def __init__(self, lf_latent_model, lf_model, hf_model, dataset):
#         self.lf_latent_model = lf_latent_model
#         self.lf_model = lf_model
#         self.hf_model = hf_model
#         self.dataset = dataset
#         self.device = next(lf_latent_model.parameters()).device  

#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         noisy_sample, clean_sample, label = self.dataset[idx]
#         noisy_sample = noisy_sample.to(self.device)
#         clean_sample = clean_sample.to(self.device)
#         label = label.to(self.device)

#         with torch.no_grad():
#             latent_rep = self.lf_latent_model(noisy_sample)['output']
#             lf_output = self.lf_model(noisy_sample)
#             hf_output = self.hf_model(clean_sample)

#         return latent_rep, lf_output, hf_output, label
    

class FidelityDataset(Dataset):
    def __init__(self, lf_latent_model, lf_model, hf_model, dataset):
        self.device = next(lf_latent_model.parameters()).device
        self.preloaded_data = []

        # Preload all data
        for noisy_sample, clean_sample, label in dataset:
            noisy_sample = noisy_sample.to(self.device)
            clean_sample = clean_sample.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                latent_rep = lf_latent_model(noisy_sample)['output']
                lf_output = lf_model(noisy_sample)
                hf_output = hf_model(clean_sample)

            self.preloaded_data.append((latent_rep, lf_output, hf_output, label))

    def __len__(self):
        return len(self.preloaded_data)
    
    def __getitem__(self, idx):
        return self.preloaded_data[idx]