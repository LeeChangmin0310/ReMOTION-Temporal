#############################################
# TopKSoftPooling:
# Simpler version of attention pooling that directly uses input embeddings
# to compute attention scores without linear projection.
#############################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSoftPooling(nn.Module):
    """
    Learnable soft pooling over Top-K selected chunk embeddings.
    Used to aggregate selected chunks into a pooled representation
    for KL alignment with GatedAttentionPooling outputs.
    
    Input: (K, D) selected chunk embeddings
    Output: (D,) pooled session-level embedding
    """
    def __init__(self, input_dim, temperature=1.0):
        super(TopKSoftPooling, self).__init__()
        self.score_layer = nn.Linear(input_dim, 1)
        self.temperature = temperature

    def forward(self, x):
        """
        Args:
            x (Tensor): (K, D) Top-K selected chunk embeddings
        Returns:
            pooled (Tensor): (D,) pooled representation
            attn (Tensor): (K, 1) attention weights
        """
        scores = self.score_layer(x)  # (K, 1)
        attn = F.softmax(scores / self.temperature, dim=0)  # (K, 1)
        pooled = torch.sum(attn * x, dim=0)  # (D,)
        return pooled, attn