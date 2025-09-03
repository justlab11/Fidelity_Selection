import torch
import torch.nn as nn
from torchvision import models
from sklearn.svm import SVC
from typing import List
import torch.optim as optim


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet architecture.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        layers = []

        encoder_features = self.encoder(x)[::-1]
        layers.append(encoder_features[0])

        x = self.decoder(encoder_features[0], encoder_features[1:])
        x = self.output(x)
        layers.append(x)
        
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

from src.datasets import CUBDataset, FE_Dataset, CropDataset
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


train_ds = CropDataset(
    root="data",
    split="train"
)

val_ds = CropDataset(
    root="data",
    split="val"
)

test_ds = CropDataset(
    root="data",
    split="test"
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=16
)

test_loader = DataLoader(
    test_ds,
    batch_size=16
)

lf_model = build_unet(
    num_channels=3,
    num_classes=14
)

hf_model = build_unet(
    num_channels=6,
    num_classes=14
)

lf_model.load_state_dict(torch.load("best_lf_model_seg.pt", weights_only=True))
hf_model.load_state_dict(torch.load("best_hf_model_seg.pt", weights_only=True))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
lf_model = lf_model.to(DEVICE)
hf_model = hf_model.to(DEVICE)

def create_fe_dataset(dataloader, lf_model, hf_model, device):
    lf_embeddings = []
    lf_preds = []
    hf_preds = []
    labels = []

    # Set models to eval mode
    lf_model.eval()
    hf_model.eval()

    lf_model.to(device)
    hf_model.to(device)

    with torch.no_grad():
        for lf_sample, hf_sample, label in dataloader:
            if device is not None:
                lf_sample = lf_sample.to(device, torch.float)
                hf_sample = hf_sample.to(device, torch.float)
                label = label.to(device)

            # LF model
            lf_outs = lf_model(lf_sample)
            lf_embed = lf_outs[0]
            lf_pred = lf_outs[-1]

            # HF model (concatenate lf_sample and hf_sample along last dim)
            hf_input = torch.cat([lf_sample, hf_sample], dim=1)
            hf_outs = hf_model(hf_input)
            hf_pred = hf_outs[-1]

            lf_embeddings.append(lf_embed.cpu())
            lf_preds.append(lf_pred.cpu())
            hf_preds.append(hf_pred.cpu())
            labels.append(label.cpu())

    # Concatenate all batches
    lf_embeddings = torch.cat(lf_embeddings, dim=0)
    lf_preds = torch.cat(lf_preds, dim=0)
    hf_preds = torch.cat(hf_preds, dim=0)
    labels = torch.cat(labels, dim=0)

    return FE_Dataset(lf_embeddings, lf_preds, hf_preds, labels)

fe_train_ds = create_fe_dataset(
    dataloader=train_loader,
    lf_model=lf_model,
    hf_model=hf_model,
    device=DEVICE
)

fe_val_ds = create_fe_dataset(
    dataloader=val_loader,
    lf_model=lf_model,
    hf_model=hf_model,
    device=DEVICE
)

fe_test_ds = create_fe_dataset(
    dataloader=test_loader,
    lf_model=lf_model,
    hf_model=hf_model,
    device=DEVICE
)

fe_train_loader = DataLoader(
    fe_train_ds,
    batch_size=64,
    shuffle=True
)

fe_val_loader = DataLoader(
    fe_val_ds,
    batch_size=64
)

fe_test_loader = DataLoader(
    fe_test_ds,
    batch_size=64
)

class MetaLossFunction(nn.Module):
    def __init__(self, ch, cw: float, device: str, loss_fun: str="CE"):
        '''
        Loss function for the FE model
        Parameters:
            ch - cost of using high fidelity
            cw - cost of being wrong
            loss_fun - the choice of loss function used to determine wrong predictions
        '''
        super().__init__()
        self.ch = ch
        self.ch.insert(0, 0) # first element is zero because no cost for LF sample usage

        self.cw = cw
        self.device = device

        match loss_fun:
            case "CE":
                self.loss_fun = nn.CrossEntropyLoss(reduction="none")
            case "binary":
                self.loss_fun = nn.L1Loss(reduction="none")
            case default:
                raise ValueError("Invalid loss function")

    def forward(self, y_true: torch.tensor, y_preds: torch.tensor, choices: torch.tensor):
        choices = nn.Softmax(dim=1)(choices)
        choices = choices.to(self.device)

        model_losses = []
        for i,pred in enumerate(y_preds):
            loss = self.loss_fun(pred, y_true)
            model_losses.append(loss)

        model_losses_tensor = torch.stack(model_losses, dim=1)
        model_losses_tensor = model_losses_tensor.to(self.device)

        ch_vec = torch.tensor(self.ch, device=self.device).float()

        total_costs = model_losses_tensor + ch_vec

        expected_costs = torch.sum(choices * total_costs, dim=1)

        return expected_costs.mean()
    
def classifier_one_run(model, dataloader, criterion, fidelity, optimizer=None, scheduler=None):
    """
    Perform one run through the DataLoader.

    Args:
    - model (torch.nn.Module): The neural network model.
    - dataloader (DataLoader): The DataLoader for iterating over the dataset.
    - criterion (callable): The loss function.
    - optimizer (torch.optim.Optimizer, optional): The optimizer for updating model weights.

    Returns:
    - float: Average loss over the run.
    - int: Correct predictions for calculating accuracy if needed.
    """
    if fidelity not in ["lf", "hf", "gate"]:
        return

    model.train(mode=bool(optimizer))
    device = next(model.parameters()).device
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_high = 0
    if fidelity == "lf":
        for data, _, target in dataloader:
            target = target.type(torch.LongTensor)
            data, target = data.to(device, torch.float), target.to(device)
            output = model(data)[-1]

            loss = criterion(output, target)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            predicted = torch.argmax(output, dim=1)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)
        if scheduler:
            scheduler.step()

    elif fidelity == "hf":
        for lf_data, hf_data, target in dataloader:
            target = target.type(torch.LongTensor)
            data = torch.cat([lf_data, hf_data], dim=1)

            data, target = data.to(device, torch.float), target.to(device)
            output = model(data)[-1]
            loss = criterion(output, target)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            predicted = torch.argmax(output, dim=1)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)
        if scheduler:
            scheduler.step()

    elif fidelity == "gate":
        for lf_embeddings, lf_preds, hf_preds, target in dataloader:
            lf_embeddings = lf_embeddings.to(device, torch.float)
            lf_preds = lf_preds.to(device, torch.float)      # [B, C, H, W]
            hf_preds = hf_preds.to(device, torch.float)      # [B, C, H, W]

            target = target.long().to(device)                # [B, H, W]

            outputs = model(lf_embeddings)[-1]               # [B, 2] for gate
            choices = torch.argmax(outputs, dim=1)           # [B]

            lf_pixel_pred = torch.argmax(lf_preds, dim=1)    # [B, H, W]
            hf_pixel_pred = torch.argmax(hf_preds, dim=1)    # [B, H, W]

            # Select per-image output: if choice==1 select hf_pixel_pred, else lf_pixel_pred
            final_pred = torch.where(
                choices[:, None, None] == 1,
                hf_pixel_pred,
                lf_pixel_pred
            )

            loss = criterion(target, lf_preds, hf_preds, outputs)   # Or your custom loss

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Pixel-wise (mean) accuracy: correct pixels over total pixels
            correct = (final_pred == target).float().sum().item()
            num_pixels = torch.numel(target)
            total_correct += correct
            total_samples += num_pixels
            total_loss += loss.item() * lf_preds.size(0)
            total_high += choices.sum().item()

        if scheduler:
            scheduler.step()

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    if fidelity == "gate":
        high_count = total_high / dataloader.dataset.__len__()
        return average_loss, accuracy, high_count
    
    return average_loss, accuracy



for r in range(0, 101, 1):
    r_val = r / 100
    loss = MetaLossFunction(
        ch=[r_val],
        cw=1,
        device=DEVICE
    )

    fe_model = LatentCNNHead(
        in_channels=1024,
        num_classes=2
    )

    optimizer_fe = optim.Adam(fe_model.parameters(), lr=3e-4)

    val_min = 100000

    for i in range(20):
        train_loss, train_acc, train_count = classifier_one_run(
            model=fe_model,
            dataloader=fe_train_loader,
            criterion=loss,
            fidelity="gate",
            optimizer=optimizer_fe
        )

        val_loss, val_acc, val_count = classifier_one_run(
            model=fe_model,
            dataloader=fe_val_loader,
            criterion=loss,
            fidelity="gate",
        )

        if val_loss < val_min:
            torch.save(fe_model.state_dict(), 'best_fe_model_seg.pt')

    fe_model.load_state_dict(torch.load('best_fe_model.pt', weights_only=True))

    test_loss, test_acc, test_count = classifier_one_run(
        model=fe_model,
        dataloader=fe_test_loader,
        criterion=loss,
        fidelity="gate",
    )

    print(
        f"{r_val:.3f}\t{test_count:.4f}\t{test_loss:.4f}\t{test_acc * 100:.2f}"
    )
