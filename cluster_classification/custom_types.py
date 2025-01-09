from typing import Literal, Optional
from pydantic import BaseModel, Field

class Augmentation(BaseModel):
    augmentation: str
    strength: float
    steps: int

class Settings(BaseModel):
    radius: int
    num_dims: int
    num_samples: int
    num_classes: int
    wrong_class_prob: float

class Dataset(BaseModel):
    name: str
    folder: str
    settings: Settings
    augmentations: dict[str, Augmentation]

class Parameters(BaseModel):
  latent_representation_size: int
  random_seed: int
  num_reruns: int

class EarlyStop(BaseModel):
    patience: int
    min_delta: float

class Stage1(BaseModel):
    epochs: int
    hf_save_location: str
    lf_save_location: str
    results_save_location: str
    batch_size: int
    loss_fun: str
    early_stop: EarlyStop

class Optimizer(BaseModel):
    lr: float
    weight_decay: float

class Scheduler(BaseModel):
    gamma: float

class FeModel(BaseModel):
    type: Literal["mlp", "svm"]
    regularization: Optional[float] = Field(None, description="not used if type != svm")
    kernel: Optional[str] = Field(None, description="not used if type != svm")
    optimizer: Optimizer
    scheduler: Scheduler
    early_stop: EarlyStop

class RRange(BaseModel):
    start: float
    stop: float
    num_steps: int
    scale: str
    direction: Literal["normal", "reversed"]

class Stage2(BaseModel):
    epochs: int
    fe_save_location: str
    results_save_location: str
    batch_size: int
    loss: Literal["fe_original", "fe_alternative"]
    fe_model: FeModel
    r_range: RRange

class ClusterClassificationConfig(BaseModel):
    dataset: Dataset
    parameters: Parameters
    stage1: Stage1
    stage2: Stage2
