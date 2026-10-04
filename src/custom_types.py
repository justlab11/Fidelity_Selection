from pydantic import BaseModel, ConfigDict, model_validator
from typing import Literal, List
import numpy as np

class SplitData(BaseModel):
    train: List[float]
    test: List[float]
    val: List[float]

class DatasetSettings(BaseModel):
    name: Literal["toy_2d", "toy_2d_high_ceiling", "toy_5d", "mnist_noise", "mnist_rotation", "bird_grayscale", "bird_color", "crop", "llvip", "bigearthnet", "adni_mrf_bmcamri", "adni_mrf_mri", "adni_bmca_mri", "adni_libra_mri", "adni_libra_bmca", "adni_mri_bmca"]
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

class FESettings(BaseModel):
    epochs: int
    batch_size: int
    reruns: int = 1
    # LogSeededGreedySearch's search params (replaces AdaptiveGridSearch's
    # min_bracket_width, which has no equivalent concept here - see
    # helpers.LogSeededGreedySearch's docstring for the full algorithm).
    # Defaults match the budget validated on CUB/LLVIP's 5-rerun sweeps
    # (7 seed points + 23 refinement = 30 total, matching
    # AdaptiveGridSearch's old fixed 30-model grid budget).
    log_seed_points: List[float] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    refinement_budget: int = 23
    gap_tolerance: float = 0.02
    wide_bracket_ratio: float = 4.0
    # Override for main.py's gate_lr (normally 5e-5 if imbalanced_gate else
    # 3e-4). None = keep that old behavior - only set explicitly per-config
    # once validated there (see config.yml/config_cub_resnet_vit_color_256.yml's
    # comments): a tmp_experiments/epoch40_highlr_sweep grid search over both
    # CUB experiments found gate_lr=5e-4 + disagreement_weight_cap=5.0 beat
    # the old 5e-5/uncapped combo on best-achieved val_loss at a 40-epoch
    # budget for both. Left as None (unvalidated) for every other dataset.
    gate_lr: float | None = None
    # Caps build_gate_dataloaders_with_reweighting's disagreement-oversampling
    # weight (normally uncapped num_agree/num_disagree, which was ~72x for
    # bird_color - a handful of disagreeing train samples oversampled that
    # hard overfits/collapses the gate within a couple epochs; see
    # tmp_experiments/gate_overfit_fix/experiment.py). None = old uncapped
    # behavior.
    disagreement_weight_cap: float | None = None
    # Post-sweep step: per-rerun per-sample routing CSVs (filename/class,
    # fe_0.1..fe_1.0) plus the three diagnostics.py analyses (entry-usage-vs-
    # gain, never-escalated gain distribution, class/scene clustering).
    # Default True - datasets without a registered sample_identity mapping
    # (toy_2d and friends) are skipped gracefully with a warning rather than
    # failing, so this default is safe for every dataset; set to False
    # explicitly per-config to suppress the warning/skip entirely instead.
    run_diagnostic_suite: bool = True

class SelectiveNetTrainingSettings(BaseModel):
    # SelectiveNet trains one model per c (= 1-usage target); this was
    # hardcoded to 40 in main.py regardless of dataset/model - set this to
    # override just for SelectiveNet. Falls back to that same 40 when unset,
    # so every existing config keeps its current behavior unchanged.
    epochs: int | None = None

class SATSettings(BaseModel):
    # SAT's training length defaults to classifier_training.epochs (set this
    # to override just for SAT without changing LF/HF/SelectiveNet's shared
    # epoch budget). With alpha=0.99 the EMA target's effective memory is
    # ~100 epochs (1/(1-alpha)) - main.py logs a warning if the resolved
    # epoch count is below that, since the abstain head may not have had
    # enough updates to adapt.
    epochs: int | None = None

class ConfigOptions(BaseModel):
    dataset: DatasetSettings
    latent_size: int
    random_seed: int
    run_comparisons: bool
    train_body: bool
    classifier_training: ClassifierSettings
    fe_training: FESettings
    selectivenet_training: SelectiveNetTrainingSettings = SelectiveNetTrainingSettings()
    sat_training: SATSettings = SATSettings()

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
