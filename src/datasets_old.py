import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, Subset
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

class BinaryHypercubeDataset(Dataset):
    def __init__(self, num_samples, N_dims=2, random_segment_ratios=False, spacing=2, noise_level=2):
        super(BinaryHypercubeDataset, self).__init__()

        self.__make_dataset__(num_samples, N_dims, random_segment_ratios, spacing, noise_level)

    def __flip_every_other_set__(self, array):
        # Ensure the array is a NumPy array
        array = np.array(array)

        # Reshape the array to separate pairs for easy manipulation
        pairs = array.reshape(-1, 2)

        # Flip every other pair
        pairs[1::2] = pairs[1::2, ::-1]

        # Reshape back to the original shape
        flipped_array = pairs.reshape(-1)
        return flipped_array

    def __make_dataset__(self, num_samples, N_dims, random_segment_ratios, spacing, noise_level):
        if random_segment_ratios:
            splits = np.random.dirichlet(np.ones(2**N_dims), size=1) * num_samples
            splits = list(splits)
        else:
            splits = [num_samples//(2**N_dims) for _ in range(2**N_dims)]

        splits = [0] + splits
        splits = np.cumsum(splits)
        corners_str = [bin(i)[2:].zfill(N_dims) for i in range(2**N_dims)]
        corners_list = np.array([[int(i) for i in c] for c in corners_str])
        corners_list *= spacing

        label_sets = [i%2 for i in range(2**N_dims)]
        label_sets = np.array(label_sets)
        label_sets = self.__flip_every_other_set__(label_sets)

        self.hf_dataset = np.zeros((num_samples, N_dims))
        self.lf_dataset = np.zeros((num_samples, N_dims))
        self.labels = np.zeros(num_samples)

        for s in range(1, len(splits)):
            # print(splits[s-1],splits[s])
            self.hf_dataset[splits[s-1]:splits[s]] = np.random.multivariate_normal(corners_list[s-1], np.zeros((N_dims,N_dims)), size=(splits[s]-splits[s-1]))
            # print(s-1)
            self.labels[splits[s-1]:splits[s]] = label_sets[s-1]

        self.lf_dataset = self.hf_dataset + np.random.standard_normal(size=(len(self.hf_dataset), N_dims)) * noise_level
        self.hf_dataset += np.random.standard_normal(size=(len(self.hf_dataset), N_dims)) * .4

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx:int):
        return self.lf_dataset[idx], self.hf_dataset[idx], self.labels[idx]

class DualFidelityDataset:
    def __init__(self, dataset_name, augmentation, high_fidelity_ratio, augmentation_degree, 
                 data_folder="./data", val_split=0.5, random_seed=42):
        self.dataset_name = dataset_name.lower()
        self.augmentation = augmentation.lower()
        self.high_fidelity_ratio = high_fidelity_ratio
        self.augmentation_degree = augmentation_degree
        self.val_split = val_split
        self.random_seed = random_seed
        self.mode = 'train'  # Default mode

        valid_dataset_names = ["mnist", "cifar10", "cifar100"]
        valid_aug_names = ["noise", "blur", "random_rotation"]

        if self.dataset_name not in valid_dataset_names:
            valid_dataset_str = ", ".join(valid_dataset_names) 
            raise ValueError(f"Unsupported dataset. Choose {valid_dataset_str}")
        
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
        
        # Create high fidelity datasets
        num_high_fidelity_train = int(len(self.train_dataset) * self.high_fidelity_ratio)
        high_fidelity_train_indices = torch.randperm(len(self.train_dataset))[:num_high_fidelity_train]
        self.high_fidelity_train = Subset(self.train_dataset, high_fidelity_train_indices)
        self.high_fidelity_val = Subset(self.test_dataset, val_indices)
        self.high_fidelity_test = Subset(self.test_dataset, test_indices)
        
        # Create low fidelity datasets
        self.low_fidelity_train = AugmentedDataset(self.train_dataset, self.augmentation, self.augmentation_degree)
        self.low_fidelity_val = Subset(self.test_dataset, val_indices)
        self.low_fidelity_test = Subset(self.test_dataset, test_indices)

    def train(self):
        self.mode = 'train'

    def val(self):
        self.mode = 'val'

    def test(self):
        self.mode = 'test'

    def get_high_fidelity(self):
        if self.mode == 'train':
            return self.high_fidelity_train
        elif self.mode == 'val':
            return self.high_fidelity_val
        elif self.mode == 'test':
            return self.high_fidelity_test
        else:
            raise ValueError("Invalid mode. Use train(), val(), or test() to set the mode.")

    def get_low_fidelity(self):
        if self.mode == 'train':
            return self.low_fidelity_train
        elif self.mode == 'val':
            return self.low_fidelity_val
        elif self.mode == 'test':
            return self.low_fidelity_test
        else:
            raise ValueError("Invalid mode. Use train(), val(), or test() to set the mode.")

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augmentation, degree):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.degree = degree

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # Convert to PIL Image if necessary
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.numpy(), mode='L' if image.shape[0] == 1 else 'RGB')
        
        # Apply augmentation
        if self.augmentation == 'noise':
            image = self.add_noise(image)
        elif self.augmentation == 'blur':
            image = self.apply_blur(image)
        elif self.augmentation == 'random_rotation':
            image = self.random_rotate(image)
        
        # Convert back to tensor
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        
        return image, label

    def add_noise(self, image):
        np_image = np.array(image)
        noise = np.random.normal(0, self.degree, np_image.shape)
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