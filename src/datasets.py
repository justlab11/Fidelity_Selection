import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split
from PIL import Image
import os.path as path
import tifffile as tif
import os
import warnings
import logging
import xml.etree.ElementTree as ET

import numpy as np
import torch

from torch.utils.data import Dataset
import numpy as np
import torch
import psutil

logger = logging.getLogger(__name__)

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
    """Reads the pre-chipped, ready-to-use layout: root/{train,val,test}/ each
    containing paired "<basename>_img.tif" (56, 56, 6) / "<basename>_mask.tif"
    (56, 56) files. Each file pair is already the single timestep/quadrant
    crop that the old raw multi-temporal-crop-classification layout (see
    CropDatasetOld) used to carve out on the fly, so there's no
    timestep/quadrant windowing or random-cropping left to do here at
    __getitem__ time - that work already happened once when this simplified
    dataset was generated. The crop still gets resized back up to 224x224
    (matching the old pipeline and build_unet's expected input resolution),
    since it's a region-of-interest crop, not the model's native input size.
    """

    def __init__(
        self,
        root: str,
        split: str,
        seed: int = 42,
    ):
        assert split in ["train", "test", "val"], "split must be 'train', 'test', or 'val'"

        self.split = split
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

        split_folder = path.join(root, split)
        img_suffix = "_img.tif"
        self.fileset = [
            path.join(split_folder, fname[:-len(img_suffix)])
            for fname in sorted(os.listdir(split_folder))
            if fname.endswith(img_suffix)
        ]

        self.mean = None
        self.std = None

        if split == "train":
            self._compute_dataset_stats()

    def get_num_classes(self):
        return 14

    def get_lf_input_size(self):
        return 3

    def get_hf_input_size(self):
        return 3

    def __len__(self):
        return len(self.fileset)

    def _compute_dataset_stats(self, max_samples: int = 1000):
        """Estimate global mean/std from a random subsample of the train split -
        reading all ~148k precropped train chips just for this would be far
        slower than the extra precision is worth."""
        n = len(self.fileset)
        sample_idx = torch.randperm(n, generator=self.generator)[:min(max_samples, n)].tolist()

        all_data = np.stack(
            [tif.imread(self.fileset[i] + "_img.tif") for i in sample_idx], axis=0
        )
        self.mean = all_data.mean(axis=(0, 1, 2))
        self.std = all_data.std(axis=(0, 1, 2)) + 1e-8

    def __getitem__(self, idx: int):
        base_fname = self.fileset[idx]
        image = tif.imread(base_fname + "_img.tif").astype(np.float32)
        mask = tif.imread(base_fname + "_mask.tif")

        if self.mean is not None:
            image = (image - self.mean[None, None, :]) / self.std[None, None, :]
        else:
            warnings.warn("Please set self.mean and self.std to the values from the train set")

        image[:, :, :3] = image[:, :, [2, 1, 0]]
        image = self.img_transform(image)

        lf_image = image[:3]
        hf_image = image[3:] # we concatenate later in the code so we split here

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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.hf_transform = transforms.Compose([
            transforms.ToTensor(),  # Converts (H, W, C) numpy to (C, H, W) tensor
            transforms.Resize((224, 224)),  # Resize to model input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    def get_hf_input_size(self):
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
        
class LLVIPDataset(Dataset):
    """Paired visible/infrared pedestrian-detection dataset (LLVIP). LF is the
    visible-spectrum image by default (ubiquitous, cheap RGB camera hardware)
    and HF is infrared/thermal (specialized sensor, but keeps working at
    night/low-light where visible fails) — pass lf_modality="infrared" to
    flip which checkpoint plays which role.

    Every image has at least one annotated person (this is a pedestrian-
    detection dataset, not a presence/absence one), so `label` here is the
    per-image ground-truth boxes rather than a single class index: a fixed
    (max_boxes, 5) tensor of [cls, xc, yc, w, h] normalized to the letterboxed
    image, YOLOv5's own label convention. Unused rows are padded with -1 so a
    plain DataLoader can batch it without a custom collate_fn — filter real
    rows with `label[:, 0] >= 0`. Ground truth is provided so routing
    "correctness" can be scored against it if needed, even though the
    intended routing decision compares the two models' own detections to each
    other, not to this ground truth directly.

    No official validation split is provided (only train/test, per LLVIP's
    own layout). val can't be carved out of train the way CUBDataset does,
    though, the way it might look like it should be: the YOLO checkpoints this
    project wraps are pretrained on the *entirety* of LLVIP's train folder, so
    any val subset sampled from train would still be fully "seen" by those
    checkpoints no matter how it's drawn - it would measure something close to
    memorization, not generalization. val has to come from test instead.

    LLVIP's filenames encode scene id as their leading two digits (e.g.
    "260532.jpg" is scene "26"), and train/test are already split along whole
    scene boundaries - no scene appears in both. test's frames are sequential
    video within each scene, so a random per-frame val/test split would leak
    near-duplicate adjacent frames across the split (verified: with a plain
    random split, ~99% of a would-be val set had an immediate-neighbor frame
    land in test). Splitting along scene boundaries instead - holding out the
    scene(s) named in val_scene_prefixes as val, keeping every other test
    scene as the final test set - avoids that: every val/test boundary is a
    genuine scene boundary, not a mid-video cut. The unavoidable cost is a
    smaller final test set (the default holds out scene "26", ~533 of test's
    3,463 images).

    lf_img/hf_img are returned as (3, img_size, img_size) float tensors
    scaled to [0, 1] — YOLOv5's own preprocessing convention (no ImageNet
    mean/std normalization, unlike CUBDataset/CropDataset above, which back
    ResNet/ViT/UNet models instead).
    """

    def __init__(
        self,
        root: str,
        split: str,
        seed: int = 42,
        img_size: int = 640,
        max_boxes: int = 20,
        lf_modality: str = "visible",
        val_scene_prefixes: tuple = ("26",),
    ):
        assert split in ["train", "test", "val"], "split must be 'train', 'test', or 'val'"
        assert lf_modality in ["visible", "infrared"], "lf_modality must be 'visible' or 'infrared'"

        # Deferred/local: yolov5's import chain is large (pandas, ultralytics,
        # etc.) and noisy (first-import settings-file creation), so it's only
        # paid by callers that actually construct an LLVIPDataset.
        from yolov5.utils.augmentations import letterbox
        from yolov5.utils.general import xyxy2xywh
        self._letterbox = letterbox
        self._xyxy2xywh = xyxy2xywh

        self.root = root
        self.split = split
        self.img_size = img_size
        self.max_boxes = max_boxes
        self.lf_modality = lf_modality
        self.hf_modality = "infrared" if lf_modality == "visible" else "visible"
        # seed is accepted (unused) for interface parity with build_dataset's
        # uniform per-dataset call signature - this split is scene-based and
        # deterministic, not random, so there's nothing left for it to seed.

        self.file_split = "test" if split in ("test", "val") else "train"
        image_dir = path.join(root, "visible", self.file_split)
        basenames = sorted(
            f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        if split == "val":
            self.basenames = [f for f in basenames if f[:2] in val_scene_prefixes]
        elif split == "test":
            self.basenames = [f for f in basenames if f[:2] not in val_scene_prefixes]
        else:
            self.basenames = basenames

    def get_num_classes(self):
        return 1  # single "person" class; kept for interface parity with the other datasets

    def get_lf_input_size(self):
        return 3

    def get_hf_input_size(self):
        return 3

    def __len__(self):
        return len(self.basenames)

    def _load_image(self, modality, fname):
        fpath = path.join(self.root, modality, self.file_split, fname)
        return np.array(Image.open(fpath).convert("RGB"))  # HWC, RGB, uint8

    def _load_boxes(self, fname):
        xml_path = path.join(self.root, "Annotations", path.splitext(fname)[0] + ".xml")
        annotation = ET.parse(xml_path).getroot()
        boxes = [
            [
                float(obj.find("bndbox/xmin").text), float(obj.find("bndbox/ymin").text),
                float(obj.find("bndbox/xmax").text), float(obj.find("bndbox/ymax").text),
            ]
            for obj in annotation.findall("object")
        ]
        return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

    def _letterbox_image_and_boxes(self, img, boxes):
        padded, ratio, (dw, dh) = self._letterbox(img, self.img_size, auto=False)
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio[0] + dw
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio[1] + dh
        return padded, boxes

    def __getitem__(self, idx):
        fname = self.basenames[idx]

        lf_raw = self._load_image(self.lf_modality, fname)
        hf_raw = self._load_image(self.hf_modality, fname)
        boxes = self._load_boxes(fname)  # xyxy, original pixel space (shared by both modalities)

        # lf/hf share the same original resolution (LLVIP images are pixel-
        # aligned pairs), so letterboxing both at the same img_size produces
        # identical box coordinates either way — computed per-modality anyway
        # so a future per-modality resize policy wouldn't silently desync them.
        lf_img, lf_boxes = self._letterbox_image_and_boxes(lf_raw, boxes)
        hf_img, _ = self._letterbox_image_and_boxes(hf_raw, boxes)

        h, w = lf_img.shape[:2]
        xywh = self._xyxy2xywh(lf_boxes)
        xywh[:, [0, 2]] /= w
        xywh[:, [1, 3]] /= h

        n = min(len(xywh), self.max_boxes)
        label = -np.ones((self.max_boxes, 5), dtype=np.float32)
        label[:n, 0] = 0  # single class: person
        label[:n, 1:] = xywh[:n]

        lf_tensor = torch.from_numpy(lf_img).permute(2, 0, 1).float() / 255.0
        hf_tensor = torch.from_numpy(hf_img).permute(2, 0, 1).float() / 255.0

        return lf_tensor, hf_tensor, torch.from_numpy(label)

class BigEarthNetDataset(Dataset):
    """Paired Sentinel-1 (SAR) / Sentinel-2 (multispectral optical) dataset
    (BigEarthNet v2 / "reben" layout). LF is S1 by default (2-band VV/VH SAR,
    cheap to acquire and weather/cloud-independent but coarse) and HF is S2
    (12-band optical, richer spectral signal but cloud-occluded and pricier to
    task) — pass lf_modality="s2" to flip which checkpoint plays which role.

    root is expected to contain "BigEarthNet-S1/", "BigEarthNet-S2/", and
    "metadata.parquet" (the standard BigEarthNet v2 layout). Patch/scene
    directory names are derived from metadata.parquet's patch_id and s1_name
    columns:
      - S2 dir: root/BigEarthNet-S2/<patch_id minus last 2 "_"-tokens>/<patch_id>/
      - S1 dir: root/BigEarthNet-S1/<s1_name minus last 3 "_"-tokens>/<s1_name>/
    with per-band files named "<dir_name>_<BAND>.tif" inside.

    labels is BigEarthNet's 19-class multi-label nomenclature (a patch can
    show several land-cover types at once), so `label` here is a (19,)
    multi-hot float tensor rather than a single class index.

    S2's bands live at 3 different native resolutions (10m/20m/60m per
    Sentinel-2's own sensor design), so every band is bilinearly resized to
    img_size (default 120, S1/10m-S2's native patch size) to stack into a
    single tensor.

    Not every metadata row necessarily has both S1 and S2 data present on
    disk (e.g. a partial download) — rows missing either directory are
    dropped at init (see verify_paths).
    """

    S1_BANDS = ["VV", "VH"]
    S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    LABELS = [
        "Agro-forestry areas", "Arable land", "Beaches, dunes, sands", "Broad-leaved forest",
        "Coastal wetlands", "Complex cultivation patterns", "Coniferous forest",
        "Industrial or commercial units", "Inland waters", "Inland wetlands",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Marine waters", "Mixed forest", "Moors, heathland and sclerophyllous vegetation",
        "Natural grassland and sparsely vegetated areas", "Pastures", "Permanent crops",
        "Transitional woodland, shrub", "Urban fabric",
    ]

    def __init__(
        self,
        root: str,
        split: str,
        seed: int = 42,
        img_size: int = 120,
        lf_modality: str = "s1",
        verify_paths: bool = True,
    ):
        assert split in ["train", "test", "val"], "split must be 'train', 'test', or 'val'"
        assert lf_modality in ["s1", "s2"], "lf_modality must be 's1' or 's2'"
        # seed is accepted (unused) for interface parity with build_dataset's
        # uniform per-dataset call signature - BigEarthNet ships fixed
        # train/validation/test splits in metadata.parquet, so there's
        # nothing left for it to seed.

        self.root = root
        self.img_size = img_size
        self.lf_modality = lf_modality
        self.hf_modality = "s2" if lf_modality == "s1" else "s1"
        self.s1_root = path.join(root, "BigEarthNet-S1")
        self.s2_root = path.join(root, "BigEarthNet-S2")
        self.label_to_idx = {label: i for i, label in enumerate(self.LABELS)}

        split_map = {"train": "train", "val": "validation", "test": "test"}
        df = pd.read_parquet(path.join(root, "metadata.parquet"))
        df = df[df["split"] == split_map[split]]

        self.samples = []
        for row in df.itertuples(index=False):
            patch_id = row.patch_id
            s1_name = row.s1_name

            # S2 tile dir: strip last 2 tokens (H-Order, V-Order)
            s2_tile = "_".join(patch_id.split("_")[:-2])
            s2_dir = path.join(self.s2_root, s2_tile, patch_id)

            # S1 tile dir: strip last 3 tokens (Tile, H-Order, V-Order) - scene-only dir name
            s1_tile = "_".join(s1_name.split("_")[:-3])
            s1_dir = path.join(self.s1_root, s1_tile, s1_name)

            if verify_paths and not (path.isdir(s2_dir) and path.isdir(s1_dir)):
                continue

            self.samples.append((patch_id, s2_dir, s1_name, s1_dir, list(row.labels)))

        if not self.samples:
            raise RuntimeError(f"No BigEarthNet samples found on disk for split={split!r} under root={root!r}")

    def get_num_classes(self):
        return len(self.LABELS)

    def get_lf_input_size(self):
        return len(self.S1_BANDS) if self.lf_modality == "s1" else len(self.S2_BANDS)

    def get_hf_input_size(self):
        return len(self.S2_BANDS) if self.hf_modality == "s2" else len(self.S1_BANDS)

    def __len__(self):
        return len(self.samples)

    def _read_band_raw(self, fpath):
        arr = tif.imread(fpath).astype(np.float32)
        return torch.from_numpy(arr)

    def _load_band(self, fpath):
        band = self._read_band_raw(fpath)
        if band.shape != (self.img_size, self.img_size):
            band = F.interpolate(
                band[None, None], size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )[0, 0]
        return band

    def _load_s1(self, s1_dir, s1_name):
        bands = [self._load_band(path.join(s1_dir, f"{s1_name}_{b}.tif")) for b in self.S1_BANDS]
        return torch.stack(bands, dim=0)  # (2, H, W)

    def _load_s2(self, s2_dir, patch_id):
        bands = [self._load_band(path.join(s2_dir, f"{patch_id}_{b}.tif")) for b in self.S2_BANDS]
        return torch.stack(bands, dim=0)  # (12, H, W)

    def _load_label(self, labels):
        label = torch.zeros(len(self.LABELS), dtype=torch.float32)
        for l in labels:
            label[self.label_to_idx[l]] = 1.0
        return label

    def load_raw_bands(self, idx):
        """Returns every native-resolution S1+S2 band for sample idx as a
        {band_name: (H, W) float32 tensor} dict - unresized (each Sentinel
        band keeps its own native 10m/20m/60m resolution) and unnormalized.

        This is a separate entry point from __getitem__ (which stacks/resizes
        into the fixed lf/hf tensors the rest of this codebase's models
        expect): it's meant for models.BigEarthNetFidelityModel.preprocess,
        which - given any single-modality reben_publication checkpoint -
        knows on its own which bands/order/resize/normalization to apply, so
        the dataset doesn't need to know which checkpoint will consume it.
        A DataLoader batching a Dataset built around this method auto-collates
        it into {band_name: (B, H, W)} since each band has a fixed native
        resolution across all patches.
        """
        patch_id, s2_dir, s1_name, s1_dir, _ = self.samples[idx]
        bands = {}
        for b in self.S1_BANDS:
            bands[b] = self._read_band_raw(path.join(s1_dir, f"{s1_name}_{b}.tif"))
        for b in self.S2_BANDS:
            bands[b] = self._read_band_raw(path.join(s2_dir, f"{patch_id}_{b}.tif"))
        return bands

    def __getitem__(self, idx):
        patch_id, s2_dir, s1_name, s1_dir, labels = self.samples[idx]

        modality_images = {
            "s1": lambda: self._load_s1(s1_dir, s1_name),
            "s2": lambda: self._load_s2(s2_dir, patch_id),
        }
        lf_img = modality_images[self.lf_modality]()
        hf_img = modality_images[self.hf_modality]()
        label = self._load_label(labels)

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
    # Only cache in memory if the estimated footprint stays under this fraction
    # of currently available RAM, leaving headroom for the OS/rest of the process.
    MAX_MEMORY_FRACTION = 0.7

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = sorted(os.listdir(folder_path))  # Sorted list of .pt file names

        self._in_memory = False
        self._cache = None

        if self.files:
            # Measure the real on-disk size of one sample rather than deriving it
            # from config (e.g. latent_size) — that formula breaks for datasets
            # like "crop" where the cached tensors are spatial maps, not vectors.
            sample_bytes = os.path.getsize(os.path.join(folder_path, self.files[0]))
            estimated_bytes = sample_bytes * len(self.files)
            available_bytes = psutil.virtual_memory().available
            budget_bytes = available_bytes * self.MAX_MEMORY_FRACTION

            if estimated_bytes <= budget_bytes:
                logger.info(
                    f"FE_Dataset({folder_path}): caching {len(self.files)} samples "
                    f"(~{estimated_bytes / 1e6:.1f} MB) in memory "
                    f"({available_bytes / 1e6:.1f} MB available)"
                )
                self._cache = [
                    torch.load(os.path.join(folder_path, fname))
                    for fname in self.files
                ]
                self._in_memory = True
            else:
                logger.info(
                    f"FE_Dataset({folder_path}): estimated size ~{estimated_bytes / 1e6:.1f} MB "
                    f"exceeds {self.MAX_MEMORY_FRACTION:.0%} of available RAM "
                    f"({available_bytes / 1e6:.1f} MB available); falling back to per-sample disk loads"
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self._in_memory:
            data = self._cache[idx]
        else:
            file_path = os.path.join(self.folder_path, self.files[idx])
            data = torch.load(file_path)

        lf_latent = data['lf_latent']
        lf_output = data['lf_output']
        hf_output = data['hf_output']
        label = data['label']

        return lf_latent, lf_output, hf_output, label
    
