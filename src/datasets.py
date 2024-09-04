from torch.utils.data import Dataset
import numpy as np


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