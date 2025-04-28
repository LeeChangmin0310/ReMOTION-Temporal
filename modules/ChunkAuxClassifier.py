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
    def __init__(self, input_dim, num_classes, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.aux_classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        # He init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

        return self.aux_classifier(x)  # (B, C)