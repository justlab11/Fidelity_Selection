import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.svm import SVC

class CustomMLP(nn.Module):
    def __init__(self, input_size, num_layers, output_size, hidden_size=64):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Create the sequential model
        self.model = nn.Sequential(*layers)

        # extra selectivenet heads
        self.aux_head = nn.Linear(hidden_size, output_size)
        self.selective_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        layers = {}

        latent = self.model(x)
        layers["latent"] = latent

        output = self.output_layer(latent)
        layers["output"] = output

        return layers  

    def selective_forward(self, x):    
        layers = {}

        latent = self.model(x)
        layers["latent"] = latent

        output = self.output_layer(latent)
        layers["output"] = output

        aux = self.aux_head(latent)
        layers["aux"] = aux

        select = self.selective_head(latent)
        layers["select"] = select

        return layers 
    
class CustomResNet18(nn.Module):
    def __init__(self, latent_size, output_size, num_channels=3):
        super().__init__()
        model = models.resnet18(weights="DEFAULT")
        if num_channels != 3:
            old_conv1 = model.conv1
            new_conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                # Tile the pretrained RGB filters across the extra channel groups instead of
                # randomly reinitializing, so the stem starts from meaningful features rather
                # than noise (e.g. num_channels=6 for grayscale+RGB fusion). Divide by the
                # number of tiled groups to keep the output magnitude roughly calibrated to
                # what the rest of the pretrained backbone expects.
                reps = -(-num_channels // 3)  # ceil division
                tiled = old_conv1.weight.repeat(1, reps, 1, 1)[:, :num_channels, :, :]
                new_conv1.weight.copy_(tiled / reps)
            model.conv1 = new_conv1

        num_ftrs = model.fc.in_features
        model.fc = nn.Identity()

        self.model = model
        self.latent_rep = nn.Sequential(
            nn.Linear(num_ftrs, latent_size),
            nn.Dropout(p=0.6)
        )
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(latent_size, output_size),
        )

        # extra selectivenet heads
        self.aux_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(latent_size, output_size),
        )
        self.selective_head_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(latent_size, 1),
        )

    def freeze_body(self, freeze=True):
        for param in self.model.parameters():
            param.requires_grad = not freeze

    def head(self, x):
        layers = {}

        x = self.latent_rep(x)
        layers["latent"] = x

        x = self.output_layer(x)
        layers["output"] = x

        return layers

    def forward(self, x):
        layers = {}

        x = self.model(x)
        layers["body_output"] = x

        x = self.latent_rep(x)
        layers["latent"] = x

        x = self.output_layer(x)
        layers["output"] = x

        return layers

    def selective_forward(self, x):
        layers = {}

        body = self.model(x)
        layers["body_output"] = body

        latent = self.latent_rep(body)
        layers["latent"] = latent

        output = self.output_layer(latent)
        layers["output"] = output

        aux = self.aux_head(latent)
        layers["aux"] = aux

        select = self.selective_head_layer(latent)
        layers["select"] = select

        return layers

    def selective_head(self, x):
        layers = {}

        latent = self.latent_rep(x)
        layers["latent"] = latent

        output = self.output_layer(latent)
        layers["output"] = output

        aux = self.aux_head(latent)
        layers["aux"] = aux

        select = self.selective_head_layer(latent)
        layers["select"] = select

        return layers 
    
class CustomViT(nn.Module):
    def __init__(self, latent_size, output_size):
        super().__init__()
        # Load vision transformer backbone with pretrained weights
        model = models.vit_b_16(weights="DEFAULT")  # or weights=None for no pretrained
        
        # Remove the original classification head
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Identity()
        
        self.model = model
        self.latent_rep = nn.Sequential(
            nn.Linear(num_ftrs, latent_size),
            nn.Dropout(p=0.6)
        )
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(latent_size, output_size)
        )

        # extra selectivenet heads
        self.aux_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(latent_size, output_size),
        )
        self.selective_head_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(latent_size, 1),
        )

    def freeze_body(self, freeze=True):
        for param in self.model.parameters():
            param.requires_grad = not freeze

    def head(self, x):
        layers = {}

        x = self.latent_rep(x)
        layers["latent"] = x

        x = self.output_layer(x)
        layers["output"] = x

        return layers

    def forward(self, x):
        layers = {}

        x = self.model(x)  # forward through ViT body
        layers["body_output"] = x

        x = self.latent_rep(x)
        layers["latent"] = x

        x = self.output_layer(x)
        layers["output"] = x

        return layers

    def selective_forward(self, x):
        layers = {}

        body = self.model(x)
        layers["body_output"] = body

        latent = self.latent_rep(body)
        layers["latent"] = latent

        output = self.output_layer(latent)
        layers["output"] = output

        aux = self.aux_head(latent)
        layers["aux"] = aux

        select = self.selective_head_layer(latent)
        layers["select"] = select

        return layers

    def selective_head(self, x):
        layers = {}

        latent = self.latent_rep(x)
        layers["latent"] = latent

        output = self.output_layer(latent)
        layers["output"] = output

        aux = self.aux_head(latent)
        layers["aux"] = aux

        select = self.selective_head_layer(latent)
        layers["select"] = select

        return layers


def build_svm(C=1.0, kernel='rbf'):
    valid_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    
    if kernel not in valid_kernels and not callable(kernel):
        raise ValueError(f"Invalid kernel. Choose from {valid_kernels} or provide a callable.")
    
    if C <= 0:
        raise ValueError("C must be strictly positive.")
    
    svm_model = SVC(C=C, kernel=kernel)
    
    return svm_model


from typing import List

import torch
from torch import nn


class DoubleConvBlock(nn.Module):
    """ A convolutional block in the UNet architecture.

    This block consists of two convolutional layers with batch normalization followed by a ReLU 
    activation function.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        padding (int): The padding to be applied to the input.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        batchnorm (nn.BatchNorm2d): The batch normalization layer.
        relu (nn.ReLU): The ReLU activation function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """Initializes DoubleConvBlock with specified input and output channels, kernel size, and padding.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
            padding (int, optional): The padding to be applied to the input. Defaults to 1.
        """
        super(DoubleConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        return x


class Encoder(nn.Module):
    """The encoder part of the UNet architecture.

    This consists of a series of convolutional blocks followed by maxpooling operations
    with increasing number of channels.

    Args:
        channels (List[int]): A list of channels for convolutionals block.

    Attributes:
        encoder_blocks (nn.ModuleList): A list of convolutional blocks followed by maxpooling.

    """

    def __init__(self, channels: List[int]) -> None:
        super(Encoder, self).__init__()
        self.encoder_blocks = nn.ModuleList()

        # Add a convolutional block followed by maxpooling(except last one) for each channel
        for i in range(len(channels)-1):
            self.encoder_blocks.append(
                DoubleConvBlock(channels[i], channels[i+1])),

            # Add a max pooling layer after each convolutional block except the last one
            if i < len(channels)-2:
                self.encoder_blocks.append(nn.MaxPool2d(kernel_size=2))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            List[torch.Tensor]: A list of tensors from each encoder block.
        """
        encoder_features = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

            # Save the output of each convolutional block
            if isinstance(encoder_block, DoubleConvBlock):
                encoder_features.append(x)

        return encoder_features

class Decoder(nn.Module):
    """The decoder part of the UNet architecture.

    This consists of a series of convolutional blocks with decreasing number of channels.

    Args:
        channels (List[int]): A list of channels for convolutionals block.

    Attributes:
        decoder_blocks (nn.ModuleList): A list of convolutional blocks.

    """

    def __init__(self, channels: List[int]) -> None:
        super(Decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList()

        # Add a upconvolutional block followed by a convolutional block for each channel
        for i in range(len(channels)-1):
            self.decoder_blocks.append(nn.ConvTranspose2d(
                channels[i], channels[i+1], 2, 2))
            self.decoder_blocks.append(
                DoubleConvBlock(channels[i], channels[i+1]))

    def _center_crop(self, feature: torch.Tensor, target_size: torch.Tensor) -> torch.Tensor:
        """Crops the input tensor to the target size.

        Args:
            feature (torch.Tensor): The input tensor.
            target_size (torch.Tensor): The target size.

        Returns:
            torch.Tensor: The cropped tensor.
        """
        _, _, H, W = target_size.shape
        _, _, h, w = feature.shape

        # Calculate the starting indices for the crop
        h_start = (h - H) // 2
        w_start = (w - W) // 2

        # Crop and returns the tensor
        return feature[:, :, h_start:h_start+H, w_start:w_start+W]

    def forward(self, x: torch.Tensor, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of the decoder.

        Args:
            x (torch.Tensor): The input tensor.
            encoder_features (List[torch.Tensor]): A list of tensors from each encoder block.

        Returns:
            torch.Tensor: The output tensor.
        """
        for i, decoder_block in enumerate(self.decoder_blocks):

            # Concatenate the output of the encoder with the output of the decoder
            if isinstance(decoder_block, DoubleConvBlock):
                encoder_feature = self._center_crop(encoder_features[i//2], x)
                x = torch.cat([x, encoder_feature], dim=1)

            # Apply the upconv or convolutional block
            x = decoder_block(x)
        return x

class UNet(nn.Module):
    """The UNet architecture.   

    Args:
        out_channels (int): The number of output channels.
        channels (List[int]): A list of channels for convolutionals block.

    Attributes:
        encoder (Encoder): The encoder part of the UNet architecture.
        decoder (Decoder): The decoder part of the UNet architecture.
        output (nn.Conv2d): The output layer.

    Example:
        >>> model = UNet(channels=[3, 64, 128, 256, 512], out_channels=1)
    """

    def __init__(
        self,
        channels: List[int],
        out_channels: int,
    ) -> None:
        super(UNet, self).__init__()
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels[::-1][:-1])
        self.output = nn.Conv2d(channels[1], out_channels, kernel_size=1)

        # extra selectivenet heads
        self.aux_head = nn.Conv2d(channels[1], out_channels, kernel_size=1)
        self.selective_head = nn.Conv2d(channels[1], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = {}

        # "body" is the true encoder bottleneck (smallest spatial size, most
        # channels - e.g. 1024 x 14 x 14 for a 224x224 input through this
        # architecture's 4 maxpools): the "middle of the U", analogous to the
        # small projected vector every other model in this file calls
        # "latent" (see CustomResNet18.forward's body_output -> latent_rep).
        #
        # "latent" here is instead the *decoder's* near-final feature map,
        # already upsampled back to the input's full spatial resolution -
        # structurally the segmentation head's pre-logit features, not a
        # bottleneck. It's kept spatial (not pooled) on purpose, so a future
        # gate could learn per-pixel/region-level routing instead of one
        # decision per image - see helpers.save_latent's latent_key and
        # main.py's gate_latent_key for where this is currently overridden
        # back to "body" for crop, since saving this full-resolution map per
        # sample for a ~90k-sample dataset needs ~1.6TB of disk. Re-enabling
        # it also needs LatentCNNHead (models.py) to stop global-average-
        # pooling its input before the routing decision.
        encoder_features = self.encoder(x)[::-1]
        layers["body"] = encoder_features[0]

        x = self.decoder(encoder_features[0], encoder_features[1:])
        layers["latent"] = x

        x = self.output(x)
        layers["output"] = x

        return layers

    def selective_forward(self, x):    
        layers = {}

        encoder_features = self.encoder(x)[::-1]
        layers["body"] = encoder_features[0]

        latent = self.decoder(encoder_features[0], encoder_features[1:])
        layers["latent"] = latent

        output = self.output(latent)
        layers["output"] = output

        aux = self.aux_head(latent)
        layers["aux"] = aux

        select = self.selective_head(latent)
        layers["select"] = select

        return layers 


def build_unet(num_channels, num_classes):
    # img size should be 224x224
    unet = UNet(channels=[num_channels, 64, 128, 256, 512, 1024], out_channels=num_classes)
    return unet

class LatentCNNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Output: [batch, 256, 1, 1]
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return {"output": self.fc(x)}
    
class SelectiveNet(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.proj_layer = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
        )

        self.selection_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.pred_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

        self.aux_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, lf_latent):
        proj = self.proj_layer(lf_latent)

        selection_logits = self.selection_head(proj)
        pred_logits = self.pred_head(proj)
        aux_logits = self.aux_head(proj)

        return selection_logits, pred_logits, aux_logits
    
    def body(self, lf_latent):
        proj = self.proj_layer(lf_latent)

        return proj


def _load_yolov5_checkpoint(weights_path, device="cpu"):
    """Loads a YOLOv5 checkpoint (.pt) via the `yolov5` pip package and returns
    the raw DetectionModel (unwrapped from the AutoShape/DetectMultiBackend
    inference wrapper yolov5.load() normally returns).

    Two footguns worth knowing about, both handled here:

    1. yolov5 checkpoints pickle their classes under bare `models.yolo` /
       `models.common` module paths (the original ultralytics/yolov5 repo's
       own flat layout), so yolov5.load() registers sys.modules['models']
       (and 'models.common', 'models.yolo') pointing at *its* package
       internals during unpickling. This project's own models.py — this file
       — is also imported as bare `models`, so left alone that would silently
       clobber it for the rest of the process. We snapshot and restore those
       sys.modules entries around the call so the collision never escapes
       this function (safe to do — already-instantiated objects keep direct
       references to their class objects regardless of sys.modules changes
       made afterward).
    2. This yolov5 package predates torch's 2.6 default flip of
       `torch.load(weights_only=...)` from False to True, so its internal
       load call fails the new default safe-unpickling check. We patch
       torch.load to force weights_only=False for the duration of this call
       only (fine — these are the user's own trained checkpoints).
    """
    import sys
    import functools
    import torch
    import yolov5
    import yolov5.models.yolo
    import yolov5.models.common
    import yolov5.models.experimental

    # This project's own models.py is normally already cached as bare
    # 'models' by the time this runs (e.g. main.py's `from models import
    # ...`), and src/ typically sits ahead of yolov5's install dir on
    # sys.path — so even popping the cached entry isn't enough, Python would
    # just re-resolve bare 'models' back to *this* file via the sys.path
    # scan. Instead we pre-register the exact bare aliases the checkpoint's
    # pickled classes expect (models, models.yolo, models.common,
    # models.experimental), pointed at yolov5's own already-importable
    # namespaced modules, so unpickling resolves them straight from the
    # sys.modules cache regardless of sys.path order.
    collision_keys = [k for k in sys.modules if k == "models" or k.startswith("models.")]
    snapshot = {k: sys.modules.pop(k) for k in collision_keys}
    sys.modules["models"] = yolov5.models
    sys.modules["models.yolo"] = yolov5.models.yolo
    sys.modules["models.common"] = yolov5.models.common
    sys.modules["models.experimental"] = yolov5.models.experimental

    # yolov5's device string parsing only strips a "cuda:" prefix (e.g.
    # "cuda:0" -> "0"); bare "cuda" survives unstripped and is misread as N
    # requested GPU indices (one per character), so normalize it here. The
    # unmodified `device` (whatever the caller passed) is still what we
    # .to(...) below — torch itself is fine with bare "cuda".
    yolo_device = "cuda:0" if str(device) == "cuda" else str(device)

    orig_torch_load = torch.load
    torch.load = functools.partial(orig_torch_load, weights_only=False)
    try:
        wrapped = yolov5.load(weights_path, device=yolo_device)
    finally:
        torch.load = orig_torch_load
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[k]
        sys.modules.update(snapshot)

    detection_model = wrapped.model.model  # AutoShape -> DetectMultiBackend -> DetectionModel
    detection_model = detection_model.to(device).eval()
    return detection_model


class YOLOv5FidelityModel(nn.Module):
    """Wraps a pretrained YOLOv5 checkpoint (an LF or HF backbone for the
    LLVIP gate task) as a plain nn.Module, in the same spirit as the other
    fidelity models in this file.

    forward(x) returns:
      - "output": raw per-anchor detection predictions, shape
        (B, num_anchors, 5 + num_classes) = [cx, cy, w, h, objectness,
        class_probs...], already sigmoid'd, *before* NMS — YOLOv5's own
        inference-mode output. x must be a (B, 3, H, W) float tensor scaled
        to [0, 1] (YOLOv5's own preprocessing convention — no ImageNet
        mean/std normalization, unlike CustomResNet18/CustomViT above), with
        H and W multiples of `self.stride` (32).
      - "latent": the deepest backbone feature map (the P5 scale, captured
        via a forward hook just before the Detect head), as the raw (B, C, H, W)
        spatial map — NOT pooled. Pooling here would throw away exactly the
        region-level structure (e.g. a small/occluded/distant person) that a
        downstream gate model needs to learn "which regions does LF tend to
        miss" rather than a single global summary. This is *not* the full
        multi-scale representation Detect actually reads from (P3/P4/P5) —
        just the coarsest scale, kept simple since nothing downstream consumes
        a YOLO latent besides the gate FE model; revisit if a specific use
        needs the finer scales too.

    num_classes is 1 ("person") for the LLVIP checkpoints this wraps.
    """

    def __init__(self, weights_path, device="cpu"):
        super().__init__()
        self.model = _load_yolov5_checkpoint(weights_path, device)
        self.names = self.model.names
        self.stride = int(torch.as_tensor(self.model.stride).max())

        self._latent_feat = None
        self.model.model[-2].register_forward_hook(self._capture_latent)

    def _capture_latent(self, module, inputs, output):
        self._latent_feat = output

    def forward(self, x):
        preds = self.model(x)
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        return {"latent": self._latent_feat, "output": preds}


def build_yolov5(weights_path, device="cpu"):
    return YOLOv5FidelityModel(weights_path, device=device)


def _load_bigearthnet_checkpoint(model_name, device="cpu"):
    """Loads a pretrained BigEarthNetv2.0 checkpoint (reben_publication) from
    the Huggingface Hub, e.g. "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.1.1".

    reben_publication currently lives at src/reben-training-scripts/reben_publication
    rather than directly under src/ - the import is tried bare first (so this
    keeps working once it's moved to sit directly under src/) and falls back
    to adding that folder to sys.path.
    """
    try:
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
    except ImportError:
        import sys
        from pathlib import Path
        fallback_path = Path(__file__).resolve().parent / "reben-training-scripts"
        if fallback_path.is_dir() and str(fallback_path) not in sys.path:
            sys.path.insert(0, str(fallback_path))
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_name)
    model = model.to(device).eval()
    return model


class BigEarthNetFidelityModel(nn.Module):
    """Wraps a pretrained single-modality BigEarthNetv2.0 checkpoint
    (reben_publication's BigEarthNetv2_0_ImageClassifier) as a plain
    nn.Module, in the same spirit as YOLOv5FidelityModel above. Only "s1"-only
    and "s2"-only checkpoints are supported (not the multimodal "all"
    variants) - name them "<...>-s1-<version>" / "<...>-s2-<version>",
    matching the Huggingface Hub naming convention used by this model zoo
    (e.g. "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.1.1").

    Confusingly, the band order these checkpoints expect depends on which
    version trained them and does NOT match the Sentinel-2 technical
    documentation order used elsewhere in this codebase (e.g.
    BigEarthNetDataset's S2_BANDS):
      - v0.1.x checkpoints: S1 ["VH", "VV"], S2 10m+20m bands
        ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]
      - v0.2.x+ checkpoints: S1 ["VV", "VH"], S2 10m+20m bands
        ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
        (this is BigEarthNetDataset's own S2_BANDS order minus the two 60m
        bands, B01/B09, that these checkpoints don't use at all)
    `self.band_order` is resolved from the checkpoint's name (this is what
    makes everything below checkpoint-independent - swap model_name for any
    other single-modality reben_publication checkpoint and band_order,
    normalization stats, and image_size all re-derive themselves correctly).

    Input handling is fully generic: `preprocess` takes a
    {band_name: (B, H, W) raw tensor} dict - any superset of self.band_order
    is fine (e.g. always pass every S1+S2 band you have; unused ones are
    ignored) - and internally selects, reorders, nearest-upsamples to
    self.image_size, and normalizes exactly like reben_publication's own
    training/eval pipeline (configilm.extra.BENv2_utils.stack_and_interpolate
    + BENv2DataModule's default eval transform), independent of which
    checkpoint is loaded:
      - resize: reben_publication always nearest-upsamples each band to a
        common size before stacking (never bilinear/bicubic) - band_order
        and image_size alone are enough to redo this correctly for any
        checkpoint.
      - normalization: mean/std come from
        configilm.extra.BENv2_utils.band_combi_to_mean_std(self.band_order),
        which looks stats up purely by band *name* (computed once from the
        official BigEarthNet v2 train split, independent of channel count/
        order/checkpoint) - so passing our own (possibly legacy v0.1) band
        order still returns the right per-channel stats in the right order.
      The one assumption this can't verify from the checkpoint alone: that
      whatever model_name resolves to was actually trained with reben_publication's
      default "120_nearest" preprocessing (true for every released checkpoint
      in this model zoo as of this writing).

    forward(x) returns:
      - "output": raw (B, 19) multi-label logits (BCEWithLogitsLoss target
        convention, matching BigEarthNetDataset's multi-hot label) - pass
        through sigmoid for per-class probabilities.
      - "latent": the pooled feature vector immediately before the final
        classification layer, i.e. timm's own forward_head(..., pre_logits=True)
        - a (B, num_features) vector, in the same spirit as
        CustomResNet18/CustomViT's "latent" above (not the finer spatial
        feature maps YOLOv5FidelityModel exposes, since this is a
        single-label-per-image classification task rather than detection).

    x passed to forward() must already be preprocessed (see `preprocess`).
    """

    S1_BAND_ORDER_V01 = ["VH", "VV"]
    S2_BAND_ORDER_V01 = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]

    S1_BAND_ORDER_V02 = ["VV", "VH"]
    S2_BAND_ORDER_V02 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

    # matches reben_publication's own stack_and_interpolate default (nearest
    # upsampling to a common size) and BENv2DataModule's default eval
    # transform (Normalize using this same interpolation's stats)
    NORMALIZATION_INTERPOLATION = "120_nearest"

    def __init__(self, model_name, device="cpu"):
        super().__init__()
        self.model = _load_bigearthnet_checkpoint(model_name, device)
        self.modality, self.band_order = self._resolve_modality_and_band_order(model_name)

        vision_encoder = self._get_vision_encoder()
        if not (hasattr(vision_encoder, "forward_features") and hasattr(vision_encoder, "forward_head")):
            raise ValueError(
                f"BigEarthNetFidelityModel expects a timm-style backbone with "
                f"forward_features/forward_head (e.g. resnet, rdnet); "
                f"model_name={model_name!r} does not expose one (got {type(vision_encoder)})"
            )

        expected_channels = self.model.config.channels
        if len(self.band_order) != expected_channels:
            raise ValueError(
                f"Resolved band_order {self.band_order} has {len(self.band_order)} bands "
                f"but checkpoint {model_name!r} expects config.channels={expected_channels}"
            )
        self.image_size = self.model.config.image_size

        from configilm.extra.BENv2_utils import band_combi_to_mean_std
        mean, std = band_combi_to_mean_std(self.band_order, interpolation=self.NORMALIZATION_INTERPOLATION)
        self.register_buffer("input_mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("input_std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def _resolve_modality_and_band_order(self, model_name):
        name = model_name.lower()
        if "-s1-" in name:
            modality = "s1"
        elif "-s2-" in name:
            modality = "s2"
        else:
            raise ValueError(
                f"BigEarthNetFidelityModel only supports single-modality "
                f"('...-s1-...'/'...-s2-...') checkpoints, got model_name={model_name!r}"
            )

        is_v01 = "-v0.1" in name
        if modality == "s1":
            band_order = self.S1_BAND_ORDER_V01 if is_v01 else self.S1_BAND_ORDER_V02
        else:
            band_order = self.S2_BAND_ORDER_V01 if is_v01 else self.S2_BAND_ORDER_V02
        return modality, band_order

    def _get_vision_encoder(self):
        # self.model is the BigEarthNetv2_0_ImageClassifier LightningModule;
        # self.model.model is a ConfigILM.ConfigILM wrapping the actual timm
        # backbone as .vision_encoder (DINOv3 checkpoints wire this
        # differently and aren't supported here - see class docstring).
        return self.model.model.vision_encoder

    def preprocess(self, band_tensors: dict):
        """Generic, checkpoint-independent input assembly. band_tensors is a
        {band_name: (B, H, W) raw tensor} dict at each band's own native
        resolution (e.g. BigEarthNetDataset.load_raw_bands's combined S1+S2
        dict) - extra keys beyond self.band_order are ignored, so the same
        dict can feed any single-modality checkpoint this class wraps.
        Selects/reorders the needed bands, nearest-upsamples each to
        self.image_size, stacks into (B, C, H, W), and normalizes with this
        checkpoint's band combination's mean/std. Returns a tensor ready for
        forward().
        """
        bands = []
        for b in self.band_order:
            band = band_tensors[b].float()
            if band.shape[-2:] != (self.image_size, self.image_size):
                band = F.interpolate(
                    band.unsqueeze(1), size=(self.image_size, self.image_size), mode="nearest"
                ).squeeze(1)
            bands.append(band)

        x = torch.stack(bands, dim=1)  # (B, C, H, W)
        x = (x - self.input_mean) / self.input_std
        return x

    def forward(self, x):
        vision_encoder = self._get_vision_encoder()
        feats = vision_encoder.forward_features(x)
        latent = vision_encoder.forward_head(feats, pre_logits=True)
        output = vision_encoder.forward_head(feats)

        return {"latent": latent, "output": output}


def build_bigearthnet(model_name, device="cpu"):
    return BigEarthNetFidelityModel(model_name, device=device)
