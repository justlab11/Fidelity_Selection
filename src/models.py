import torch
import torch.nn as nn
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
