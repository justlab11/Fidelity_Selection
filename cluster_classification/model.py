import torch.nn as nn

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
