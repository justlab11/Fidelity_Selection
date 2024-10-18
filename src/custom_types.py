from pydantic import BaseModel, Field
from typing import Optional, Literal

class ToyDatasetParameters:
    radius: int
    num_dims: int
    num_samples: int
    num_classes: int
    wrong_class_prob: float

class DatasetSettings(BaseModel):
    name: str
    augmentation: str
    augmentation_level: int
    toy_dataset_parameters: Optional[ToyDatasetParameters] = None

class Parameters(BaseModel):
    latent_representation_size: int
    random_seed: int
    num_reruns: int

class Classifier(BaseModel):
    type: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "mlp"]
    pretrained: bool
    num_layers: Optional[int] = None

class Optimizer(BaseModel):
    lr: float
    weight_decay: float

class Scheduler(BaseModel):
    gamma: float

class EarlyStop(BaseModel):
    patience: int
    min_delta: int

class ModelConfig(BaseModel):
    classifier: Classifier
    train: bool
    validate: bool
    save_location: str
    load_file: Optional[str] = None
    optimizer: Optimizer
    scheduler: Scheduler
    early_stop: EarlyStop

class Stage1(BaseModel):
    epochs: int
    batch_size: int
    hf_model: ModelConfig
    lf_model: ModelConfig

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
    scale: Literal["linear", "logarithmic"]
    direction: Literal["reversed", "normal"]

class Stage2(BaseModel):
    epochs: int
    save_location: Optional[str] = None
    fe_model: FeModel
    r_range: RRange

class Options(BaseModel):
    dataset: DatasetSettings
    parameters: Parameters
    stage1: Stage1
    stage2: Stage2