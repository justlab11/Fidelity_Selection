import torch
import torch.nn as nn
from torchvision import models
from sklearn.svm import SVC

def build_mlp(input_size, num_layers, output_size, hidden_size=64, device='cpu'):
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.ReLU())
    
    # Hidden layers
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    
    # Output layer
    layers.append(nn.Linear(hidden_size, output_size))
    
    # Create the sequential model
    model = nn.Sequential(*layers).to(device)
    
    return model

def build_resnet(resnet_size, latent_size, output_size, pretrained=True, device='cpu'):
    # Dictionary mapping resnet_size to the corresponding model function
    resnet_models = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152
    }
    
    # Check if the requested ResNet size is valid
    if resnet_size not in resnet_models:
        raise ValueError(f"Invalid ResNet size. Choose from {list(resnet_models.keys())}")
    
    # Get the appropriate ResNet model
    model = resnet_models[resnet_size](weights="DEFAULT")
    
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

def build_svm(C=1.0, kernel='rbf'):
    valid_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    
    if kernel not in valid_kernels and not callable(kernel):
        raise ValueError(f"Invalid kernel. Choose from {valid_kernels} or provide a callable.")
    
    if C <= 0:
        raise ValueError("C must be strictly positive.")
    
    svm_model = SVC(C=C, kernel=kernel)
    
    return svm_model

