from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict

class PerformanceData(BaseModel):
    train: List[float]
    test: List[float]
    val: List[float]

class MetaData(BaseModel):
    file_name: Optional[str] = None
    loss: PerformanceData
    acc: PerformanceData
    augmentation: str
    augmentation_degree: float
    dataset: str

class ToyDatasetParameters(BaseModel):
    radius: float
    num_dims: int
    num_samples: int
    num_classes: int
    wrong_class_prob: float

class AugmentationSettings(BaseModel):
    augmentation: Literal["none", "noise", "blur", "rotation", "degradation"]
    strength: float
    steps: int
    schedule: Literal["linear", "decay"]

class DatasetSettings(BaseModel):
    name: Literal["toy", "mnist", "cifar10", "cifar100", "crop"]
    folder: Optional[str] = None
    augmentations: Dict[Literal["high_fidelity", "low_fidelity"], AugmentationSettings]
    toy_dataset_parameters: Optional[ToyDatasetParameters] = None

class Parameters(BaseModel):
    latent_representation_size: int
    random_seed: int
    num_reruns: int

class Optimizer(BaseModel):
    lr: float
    weight_decay: float

class Scheduler(BaseModel):
    gamma: float

class EarlyStop(BaseModel):
    patience: int
    min_delta: int

class FeModel(BaseModel):
    type: Literal["mlp", "svm"]
    
class Stage1(BaseModel):
    epochs: int 
    save_location: Optional[str] = None 
    batch_size: int 
    loss_fun: str 
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
   batch_size: int 
   fe_model : FeModel 
   r_range : RRange 

class Options(BaseModel):
   dataset : DatasetSettings 
   parameters : Parameters 
   stage1 : Stage1 
   stage2 : Stage2 