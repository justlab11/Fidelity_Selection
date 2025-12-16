from pydantic import BaseModel
from typing import Literal, List

class SplitData(BaseModel):
    train: List[float]
    test: List[float]
    val: List[float]

class DatasetSettings(BaseModel):
    name: Literal["toy_2d", "toy_5d", "mnist_noise", "mnist_rotation", "bird_grayscale", "bird_color", "crop"]
    folder: str

class ClassifierSettings(BaseModel):
    epochs: int
    batch_size: int
    loss_fun: Literal["CE"]
    lf_model: Literal["mlp", "resnet", "vit", "unet"]
    hf_model: Literal["mlp", "resnet", "vit", "unet"]

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