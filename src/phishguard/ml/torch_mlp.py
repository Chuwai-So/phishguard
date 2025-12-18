from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

#Handles data entry calculation
class MLPBinary(nn.Module):

    #Minimal MLP for classification
    #Input(N, 11)
    #Output(N, 1) logits

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            #Maps 11 features into 16 hidden features
            nn.Linear(in_dim, 16),
            #Prevent linear stacking
            nn.ReLU(),
            #Regularization -- turn off 20% of hidden so no dependency on certain neuron
            #This prevents overfitting
            #Should turn everything back on during evaulation
            nn.Dropout(p=0.2),
            #Collapse the 16 hidden features into one score
            nn.Linear(16, 1),
        )
    #Defining behavior
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@dataclass
class TorchTrainResult:
    model: MLPBinary
    val_probs: np.ndarray

