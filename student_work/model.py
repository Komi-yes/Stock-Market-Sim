from __future__ import annotations
import torch
from torch import nn
from shared.contract import OBSERVATION_SIZE, ACTION_SIZE

class QNetwork(nn.Module):
    """Starter network. You may redesign the internal architecture.

    The only fixed requirement is:
      input  : float32 tensor [batch, 16]
      output : float32 tensor [batch, 9]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
