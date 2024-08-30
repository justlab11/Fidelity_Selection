import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
from typing import *