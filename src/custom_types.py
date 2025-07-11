from pydantic import BaseModel
from typing import Literal, List

class SplitData(BaseModel):
    train: List[float]
    test: List[float]
    val: List[float]

class FidelityAugmentations(BaseModel):
    aug_name: Literal["none", "noise", "blur", "rotation"]
    strength: float

class DatasetSettings(BaseModel):
    name: str
    folder: str
    high_fidelity: FidelityAugmentations
    low_fidelity: FidelityAugmentations

class ClassifierSettings(BaseModel):
    epochs: int
    batch_size: int
    loss_fun: Literal["CE"]

class ThresholdSettings(BaseModel):
    start: float
    stop: float
    num_steps: int
    scale: Literal["linear", "logarithmic"]
    direction: Literal["normal", "reversed"]

class FESettings(BaseModel):
    epochs: int
    batch_size: int
    r_range: ThresholdSettings

class ConfigOptions(BaseModel):
    dataset: DatasetSettings
    latent_size: int
    random_seed: int
    num_reruns: int
    classifier_training: ClassifierSettings
    fe_training: FESettings