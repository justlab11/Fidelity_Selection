import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


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
