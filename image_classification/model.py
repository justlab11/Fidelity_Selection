import torch.nn as nn
from torchvision import models

def build_resnet(latent_size, output_size, device='cpu'):
    # Dictionary mapping resnet_size to the corresponding model function
    
    # Check if the requested ResNet size is valid    
    # Get the appropriate ResNet model
    model = models.resnet18(weights="DEFAULT")
    
    # Remove the original fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Identity()
    
    # Create a new sequential module for the final layers
    new_head = nn.Sequential(
        nn.Linear(num_ftrs, latent_size),
        nn.ReLU(),
        nn.Linear(latent_size, output_size)
    )
    
    # Create a new sequential model combining ResNet and the new head
    full_model = nn.Sequential(
        model,
        new_head
    )
    
    # Move the model to the specified device
    full_model = full_model.to(device)
    
    return full_model
