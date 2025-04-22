#############################################
# ChunkAuxClassifier: 
#  An auxiliary classifier for chunk-level CE loss
#############################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkAuxClassifier(nn.Module):
    """
    ChunkAuxClassifier (Auxiliary Classifier for Chunk-level CE Loss)

    Purpose:
        - Provide auxiliary supervision at the chunk level using session-level GT labels
        - Guide the TemporalBranch to learn discriminative embeddings even before full session aggregation
        - Used only during ramp-up phase (e.g., epoch 20~34)

    Input:
        - x: Tensor of shape (B, D) or (B, T, D), where T is the number of chunks

    Output:
        - logits: Tensor of shape (B, C) or (B * T, C), for classification
    """

    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()

        # Lightweight classifier: Linear layer + activation + output layer
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),        # Normalization for stability
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),                      # Non-linear activation
            nn.Dropout(dropout_rate),       # Regularization
            nn.Linear(input_dim // 2, num_classes)  # Final output layer
        )

    def forward(self, x):
        """
        Forward pass through auxiliary classifier.

        Args:
            x (Tensor): shape (B, D) or (B, T, D)

        Returns:
            logits (Tensor): shape (B, C) or (B * T, C)
        """
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.view(B * T, D)  # Flatten chunks for classification
        return self.classifier(x)

