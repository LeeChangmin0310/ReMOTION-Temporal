#############################################
# Gated Attention Pooling
# Combines soft attention and hard gating mechanisms to pool informative chunk embeddings.
#############################################


import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Pooling Layer

    Combines soft attention weighting with a gating mechanism:
    - Learns attention scores over chunk embeddings via MLP
    - Learns per-chunk gates to modulate chunk contribution
    - Applies softmax-attn × sigmoid-gate × input for weighted aggregation

    Args:
        input_dim (int): Dimensionality of each chunk embedding
        temperature (float): Softmax temperature for controlling attention sharpness

    Input:
        x (Tensor): (B, T, D) = batch of chunk-level embeddings

    Output:
        pooled (Tensor): (B, D) = session-level embedding
        attn_weights (optional): (B, T, 1) attention scores
        entropy (optional): Scalar tensor for attention entropy regularization
    """
    def __init__(self, input_dim, temperature=0.1):
        super(GatedAttentionPooling, self).__init__()
        self.temperature = temperature

        # MLP for attention score: (B, T, D) → (B, T, 1)
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

        # MLP for gating mechanism: (B, T, D) → (B, T, D)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, return_weights=False, return_entropy=False):
        """
        Forward pass for gated attention pooling.

        Args:
            x (Tensor): (B, T, D) chunk embeddings
            return_weights (bool): whether to return attention weights
            return_entropy (bool): whether to return attention entropy

        Returns:
            pooled (Tensor): (B, D)
            attn_weights (optional): (B, T, 1)
            entropy (optional): scalar entropy term
        """
        attn_scores = self.attn(x)  # (B, T, 1)
        attn_scores = attn_scores - attn_scores.mean(dim=1, keepdim=True)
        attn_weights = F.softmax(attn_scores / self.temperature, dim=1)  # (B, T, 1)

        gate = self.gate(x)          # (B, T, D)
        gated_x = x * gate           # (B, T, D)
        pooled = torch.sum(attn_weights * gated_x, dim=1)  # (B, D)

        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=1).mean()

        if return_weights and return_entropy:
            return pooled, attn_weights, entropy
        elif return_weights:
            return pooled, attn_weights
        elif return_entropy:
            return pooled, entropy
        else:
            return pooled
