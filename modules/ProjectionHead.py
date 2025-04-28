##############################################
# Projection Head:
#  maximize constrastrive learning's effectiveness
##############################################

import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps the input features to a lower-dimensional space.
    """
    def __init__(self, input_dim, proj_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
        )
    def forward(self, x):
        return self.proj(x)
