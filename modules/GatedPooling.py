#############################################
# Gated Pooling
# Gating mechanisms to pool informative chunk embeddings.
#############################################

import torch
import entmax
import torch.nn as nn
import torch.nn.functional as F

class GatedPooling(nn.Module):
    """
    Gated Pooling Layer

    Combines soft attention weighting with a gating mechanism:
    - Learns attention scores over chunk embeddings via AttnScorer(pre-softmax attn score)'s entimax
    - Learns per-chunk gates to modulate chunk contribution
    - Applies entimax-attn(from AttnScorer) × sigmoid-gate × input for weighted aggregation

    Args:
        input_dim (int): Dimensionality of each chunk embedding

    Input:
        x (Tensor): (B, T, D) = batch of chunk-level embeddings

    Output:
        pooled (Tensor): (B, D) = session-level embedding
        attn_weights (optional): (B, T, 1) attention scores
        entropy (optional): Scalar tensor for attention entropy regularization
    """
    def __init__(self, input_dim):
        super(GatedPooling, self).__init__()
        # MLP for gating mechanism: (B, T, D) → (B, T, D)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, attn_scores, epoch=None, return_weights=False, return_entropy=False):
        """
        Forward pass for gated attention pooling.

        Args:
            x (Tensor): (B, T, D) chunk embeddings
            attn_scores (
            return_weights (bool): whether to return attention weights
            return_entropy (bool): whether to return attention entropy

        Returns:
            pooled (Tensor): (B, D)
            attn_weights (optional): (B, T, 1)
            entropy (optional): scalar entropy term
        """
        # attn_weights = entmax.entmax15(attn_scores, dim=1)  # (B, T, 1)
        # attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, 1)
        if self.training:
            alpha_g = 1.7 + 0.3 * min((epoch - 30) / 19, 1.0)                       # 1.7 → 2.0 over epochs 30→50
        elif epoch == None or not self.training: 
            alpha_g = 2.0
        attn_weights = entmax.entmax_bisect(attn_scores, alpha=alpha_g, dim=1)      # (B, T, 1)

        gate = self.gate(x)                                                         # (B, T, D)
        gated_x = x * gate                                                          # (B, T, D)
        pooled = torch.sum(attn_weights * gated_x, dim=1)                           # (B, D)

        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=1).mean()

        if return_weights and return_entropy:
            return pooled, attn_weights, entropy
        elif return_weights:
            return pooled, attn_weights
        elif return_entropy:
            return pooled, entropy
        else:
            return pooled
