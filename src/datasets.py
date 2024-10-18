import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, Subset
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights
from custom_types import Options

class HypercubeDataset:
    def __init__(self, config: Options, val_split=0.2, test_split=0.2, random_seed=42):
        
        self.config: Options = config

        toy_dataset_parameters = self.config.dataset.toy_dataset_parameters

        self.N_dim = toy_dataset_parameters.num_dims
        self.N_samples = toy_dataset_parameters.num_samples
        self.N_classes = toy_dataset_parameters.num_classes
        self.radius = toy_dataset_parameters.radius
        self.noise_level = self.config.dataset.augmentation_level
        self.wrong_class_prob = toy_dataset_parameters.wrong_class_prob

        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        self.generate_data()

    def generate_data(self):
        # Generate corner coordinates
        corners = np.array(np.meshgrid(*[[0, 1] for _ in range(self.N_dim)])).T.reshape(-1, self.N_dim)
        num_corners = len(corners)
        
        # Assert that the number of classes doesn't exceed the number of corners
        assert self.N_classes <= num_corners, f"Number of classes ({self.N_classes}) cannot exceed number of corners ({num_corners})"
        
        # Randomly select corners for clusters
        cluster_corners = corners[np.random.choice(num_corners, num_corners, replace=False)]
        
        # Scale corners by radius
        cluster_corners = cluster_corners * self.radius
        
        # Randomly assign classes to clusters
        class_assignments = np.random.choice(self.N_classes, num_corners)
        
        # Generate samples for each cluster
        samples_per_cluster = self.N_samples // num_corners
        data = []
        labels = []
        
        for i, corner in enumerate(cluster_corners):
            # Generate samples around the corner
            cluster_samples = np.random.normal(loc=corner, scale=0.1 * self.radius, size=(samples_per_cluster, self.N_dim))
            # Clip samples to ensure they stay within the hypercube
            cluster_samples = np.clip(cluster_samples, 0, self.radius)
            
            # Assign classes to samples, with some probability of wrong class
            cluster_labels = np.full(samples_per_cluster, class_assignments[i])
            wrong_mask = np.random.random(samples_per_cluster) < self.wrong_class_prob
            cluster_labels[wrong_mask] = np.random.choice(self.N_classes, np.sum(wrong_mask))
            
            data.append(cluster_samples)
            labels.append(cluster_labels)
        
        self.data = np.concatenate(data)
        self.labels = np.concatenate(labels)
        
        # Split the data into train, validation, and test sets
        train_val_data, self.test_data, train_val_labels, self.test_labels = train_test_split(
            self.data, self.labels, test_size=self.test_split, stratify=self.labels, random_state=self.random_seed
        )
        
        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(
            train_val_data, train_val_labels, test_size=self.val_split / (1 - self.test_split),
            stratify=train_val_labels, random_state=self.random_seed
        )

    def train(self):
        return HypercubeSubset(self.train_data, self.train_labels, self.noise_level)

    def val(self):
        return HypercubeSubset(self.val_data, self.val_labels, self.noise_level)

    def test(self):
        return HypercubeSubset(self.test_data, self.test_labels, self.noise_level)

class HypercubeSubset(Dataset):
    def __init__(self, data, labels, noise_level):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.noise_level = noise_level

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clean_sample = self.data[idx]
        noisy_sample = clean_sample + torch.randn_like(clean_sample) * self.noise_level
        return self.labels[idx], clean_sample, noisy_sample


class DualFidelityDataset:
    def __init__(self, config, data_folder="./data", 
                 val_split=0.5, random_seed=42):
        

        self.dataset_name = config.dataset.name.lower()
        self.augmentation = config.dataset.augmentation.lower()
        self.augmentation_level = config.dataset.augmentation_level
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
        return AugmentedDataset(self.train_data, self.augmentation, self.augmentation_level)

    def val(self):
        return AugmentedDataset(self.val_data, self.augmentation, self.augmentation_level)

    def test(self):
        return AugmentedDataset(self.test_data, self.augmentation, self.augmentation_level)

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augmentation, degree):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.degree = degree

        # ResNet18 preprocessing
        weights = ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()

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
        
        return label, original_image, augmented_image

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