from typing import Literal, Optional
from pydantic import BaseModel, Field

class Augmentation(BaseModel):
    augmentation: str
    strength: float
    steps: int
    schedule: str

class Dataset(BaseModel):
    name: str
    folder: str
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
    optimizer: Optimizer
    scheduler: Scheduler
    early_stop: EarlyStop

class RRange(BaseModel):
    start: int
    stop: int
    num_steps: int
    scale: str
    direction: Literal["normal", "reversed"]

class Stage2(BaseModel):
    epochs: int
    fe_save_location: str
    results_save_location: str
    batch_size: int
    fe_model: FeModel
    r_range: RRange

class Config(BaseModel):
    dataset: Dataset
    parameters: Parameters
    stage1: Stage1
    stage2: Stage2
