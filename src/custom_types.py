from pydantic import BaseModel
from typing import Literal, List

class SplitData(BaseModel):
    train: List[float]
    test: List[float]
    val: List[float]

class FidelityAugmentations(BaseModel):
    aug_name: Literal["none", "noise", "blur", "rotation"]
    strength: float

class DatasetSettings:
    name: str
    folder: str
    high_fidelity: FidelityAugmentations
    low_fidelity: FidelityAugmentations

class ClassifierSettings:
    epochs: int
    batch_size: int
    loss_fun: Literal["CE"]

class ThresholdSettings:
    start: float
    end: float
    num_steps: int
    scale: Literal["linear", "logorithmic"]
    direction: Literal["normal", "reversed"]

class FESettings:
    epochs: int
    batch_size: int
    r_range: ThresholdSettings

class ConfigOptions:
    dataset: DatasetSettings
    latent_size: int
    random_seed: int
    num_reruns: int
    classifier_training: ClassifierSettings
    fe_training: FESettings