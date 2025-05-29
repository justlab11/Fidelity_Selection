import numpy as np
import os
from typing import List

def build_hypercube_dataset(config: HypercubeConfig, splits: ttvSet, seed: int, balanced=True)->List[ttvSplit, ttvSplit]:
    dataset_folder: str = os.path.join(
        "..", "..", "data", "hypercube"
    )

    N_dim: int = config.num_dims
    N_samples: int = config.num_samples
    N_classes: int = config.num_classes
    radius: int = config.radius

    train_split: float = splits.train
    test_split: float = splits.test
    val_split: float = splits.val

    hf_noise_level: float = config["high_fidelity"].noise
    hf_wrong_class_prob: float = config["high_fidelity"].wrong_prob

    lf_noise_level: float = config["low_fidelity"].noise
    lf_wrong_class_prob: float = config["low_fidelity"].wrong_prob

    np.random.seed: int = seed

    corners = np.array(np.meshgrid(*[[0, 1] for _ in range(N_dim)])).T.reshape(-1, N_dim)

    num_corners = len(corners)

    assert N_classes <= num_corners, f"Number of classes ({N_classes}) cannot exceed number of corners ({num_corners})"

    cluster_corners = corners[np.random.choice(num_corners, num_corners, replace=False)]

    cluster_corners *= radius

    class_assignments = np.random.choice(N_classes, num_corners)

    if balanced:
        samples_per_cluster = [N_samples // num_corners] * num_corners

    else:
        samples_per_cluster = np.random.rand(num_corners)
        samples_per_cluster /= samples_per_cluster.sum()
        samples_per_cluster *= N_samples
        samples_per_cluster = samples_per_cluster.astype(int)

    hf_data = []
    lf_data = []
    
    hf_labels = []
    lf_labels = []

    for i, corner in enumerate(cluster_corners):
        hf_cluster_samples: np.ndarray = np.random.normal(
            loc=corner,
            scale=.1 * radius,
            size=(samples_per_cluster[i], N_dim)
        )
        lf_cluster_samples: np.ndarray = np.random.normal(
            loc=corner,
            scale=.1 * radius,
            size=(samples_per_cluster[i], N_dim)
        )

        hf_cluster_samples: np.ndarray = np.clip(hf_cluster_samples, 0, radius)
        lf_cluster_samples = np.clip(lf_cluster_samples, 0, radius)

        lf_cluster_labels: np.ndarray = np.full(
            samples_per_cluster[i],
            class_assignments[i]
        )
        hf_cluster_labels = np.copy(lf_cluster_labels)

        lf_wrong_mask = np.random.random(samples_per_cluster[i]) < lf_wrong_class_prob
        hf_wrong_mask = np.random.random(samples_per_cluster[i]) < hf_wrong_class_prob

        lf_cluster_labels[lf_wrong_mask] = np.random.choice(N_classes, np.sum(lf_wrong_mask))
        hf_cluster_labels[hf_wrong_mask] = np.random.choice(N_classes, np.sum(hf_wrong_mask))

        hf_data.append(hf_cluster_samples)
        lf_data.append(lf_cluster_samples)

        hf_labels.append(hf_cluster_labels)
        lf_labels.append(lf_cluster_labels)

    hf_data = np.concatenate(hf_data)
    lf_data = np.concatenate(lf_data)

    hf_labels = np.concatenate(hf_labels)
    lf_labels = np.concatenate(lf_labels)



