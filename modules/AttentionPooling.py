#############################################
# AttentionPooling (No Projection Version):
# Simpler version of attention pooling that directly uses input embeddings
# to compute attention scores without linear projection.
#############################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """
    Simpler attention pooling layer that directly attends over input chunk embeddings
    without projecting them to a higher dimension. Useful when the input embeddings 
    are already expressive (e.g., from a strong TemporalBranch).

    Inputs:
        - x: Tensor of shape (B, T, D_in), chunk-level embeddings

    Outputs:
        - pooled: Tensor of shape (B, D_in), session-level embedding
        - attn_weights (optional): Tensor of shape (B, T, 1)
        - entropy (optional): Scalar tensor representing average attention entropy
    """
    def __init__(self, input_dim, temperature=0.1):
        """
        Args:
            input_dim (int): Dimensionality of input embeddings
            temperature (float): Temperature for softmax scaling
        """
        super(AttentionPooling, self).__init__()
        self.temperature = temperature

        # Attention MLP over input chunks (no projection)
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x, return_weights=False, return_entropy=False):
        """
        Args:
            x (Tensor): Input embeddings, shape (B, T, D_in)
            return_weights (bool): Whether to return attention weights
            return_entropy (bool): Whether to return entropy of attention weights

        Returns:
            pooled (Tensor): (B, D_in), session-level embedding
            attn_weights (Tensor, optional): (B, T, 1)
            entropy (Tensor, optional): scalar
        """
        # Step 1: Compute attention scores -> (B, T, 1)
        attn_scores = self.attn(x)

        # Step 2: Apply Stable temperature-scaled softmax -> (B, T, 1)
        attn_scores = attn_scores - attn_scores.mean(dim=1, keepdim=True)
        attn_weights = F.softmax(attn_scores / self.temperature, dim=1)

        # Step 3: Weighted sum of original inputs -> (B, D_in)
        pooled = torch.sum(attn_weights * x, dim=1)

        # Step 4 (optional): Entropy for sparsity regularization
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=1).mean()

        if return_weights and return_entropy:
            return pooled, attn_weights, entropy
        elif return_weights:
            return pooled, attn_weights
        elif return_entropy:
            return pooled, entropy
        else:
            return pooled