from typing import Literal, Optional
from pydantic import BaseModel, Field

class Augmentation(BaseModel):
    augmentation: str
    strength: float
    steps: int

class Dataset(BaseModel):
    name: str
    folder: str
    parameters: dict
    augmentations: dict[str, Augmentation]

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
    parameters: dict
    stage1: Stage1
    stage2: Stage2
