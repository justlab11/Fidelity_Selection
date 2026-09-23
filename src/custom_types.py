from pydantic import BaseModel, ConfigDict, model_validator
from typing import Literal, List
import numpy as np

class SplitData(BaseModel):
    train: List[float]
    test: List[float]
    val: List[float]

class DatasetSettings(BaseModel):
    name: Literal["toy_2d", "toy_5d", "mnist_noise", "mnist_rotation", "bird_grayscale", "bird_color", "crop", "llvip", "bigearthnet"]
    folder: str

class ClassifierSettings(BaseModel):
    epochs: int
    batch_size: int
    loss_fun: Literal["CE"]
    lf_model: Literal["mlp", "resnet", "vit", "unet", "yolo"]
    hf_model: Literal["mlp", "resnet", "vit", "unet", "yolo"]
    trained_lf_model: str | None
    trained_hf_model: str | None
    hf_input_mode: Literal["concat", "hf_only"] = "concat"
    # Default matches the value every config before this field existed was
    # implicitly using (main.py hardcoded lr=1e-5 for lf/hf/selnet/sat
    # training) - appropriate for fine-tuning a pretrained CNN (resnet/vit),
    # but far too slow for a small model trained from scratch (e.g. "mlp" on
    # a toy dataset) to visibly converge within a reasonable epoch budget.
    lr: float = 1e-5

    @model_validator(mode="after")
    def check_yolo_requires_trained_model(self):
        if self.lf_model == "yolo" and self.trained_lf_model is None:
            raise ValueError("trained_lf_model must be set when lf_model is 'yolo'")
        if self.hf_model == "yolo" and self.trained_hf_model is None:
            raise ValueError("trained_hf_model must be set when hf_model is 'yolo'")
        return self

class ThresholdSettings(BaseModel):
    start: float
    stop: float
    num_steps: int
    scale: Literal["linear", "logarithmic"]
    direction: Literal["normal", "reversed"]

class FESettings(BaseModel):
    epochs: int
    batch_size: int
    reruns: int = 1
    r_range: ThresholdSettings

class ConfigOptions(BaseModel):
    dataset: DatasetSettings
    latent_size: int
    random_seed: int
    run_comparisons: bool
    train_body: bool
    classifier_training: ClassifierSettings
    fe_training: FESettings

    @model_validator(mode="after")
    def check_yolo_compatible_settings(self):
        is_yolo = self.classifier_training.lf_model == "yolo" or self.classifier_training.hf_model == "yolo"
        if is_yolo and not self.train_body:
            raise ValueError("train_body must be True when lf_model/hf_model is 'yolo' - YOLO has no separate head to precompute body outputs for")
        if is_yolo and self.run_comparisons:
            raise ValueError("run_comparisons must be False when lf_model/hf_model is 'yolo' - the SelectiveNet/SAT/SR baselines assume classification-shaped models")
        return self

class FEResult(BaseModel):
    r: float
    loss: float
    usage: float
    accuracy: float
