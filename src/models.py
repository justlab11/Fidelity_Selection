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
            model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

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
        self.selective_head = nn.Sequential(
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

        select = self.selective_head(latent)
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

        select = self.selective_head(latent)
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
        self.selective_head = nn.Sequential(
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

        select = self.selective_head(latent)
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

        select = self.selective_head(latent)
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
        return self.fc(x)
    
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
