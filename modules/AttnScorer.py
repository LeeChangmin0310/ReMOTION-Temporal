###################################################
# Attention Scorer
###################################################

import torch
import entmax
import torch.nn as nn
import torch.nn.functional as F

class AttnScorer(nn.Module):
    """
    AttnScorer is a lightweight attention module designed to score temporal chunks
    based on their relevance to the emotion classification task.
    
    The scores are typically used for selecting Top-K chunks per session for 
    supervised contrastive learning (SupConLossTopK).
    """

    def __init__(self, input_dim):
        """
        Initializes the attention scorer module.

        Args:
            input_dim (int): The dimensionality of each chunk embedding (D).
            temperature (float): Temperature scaling factor for softmax sharpness.
        """
        super().__init__()

        # The scorer is a 2-layer feedforward network with Tanh activation in between.
        # It compresses the D-dimensional input to a single scalar score per chunk.
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),  # Reduces dimension
            nn.Tanh(),                             # Non-linear activation
            nn.Linear(input_dim // 2, 1)           # Outputs a scalar score
        )

    def forward(self, z, temperature=0.5, epoch=0):
        """
        Args:
            z (Tensor): (B, T, D) chunk embeddings
            return_entropy (bool): whether to return entropy

        Returns:
            attn_weights (Tensor): (B, T, 1)
            entropy (optional): scalar tensor for entropy loss
        """
        scores = self.scorer(z)  # Raw attention scores: (B, T, 1)
        raw_scores = scores - scores.mean(dim=1, keepdim=True) # Center the scores (zero-mean across time dimension) for stability
        
        if epoch < 10: # Phase 0.0
            return F.softmax(raw_scores / temperature, dim=1), raw_scores
        elif 10 <= epoch < 35: # Phase 0.5 and 1
            # return entmax.entmax15(raw_scores, dim=1), raw_scores
            return F.softmax(raw_scores / temperature, dim=1), raw_scores
        else:
            return None, raw_scores # return pre-softmax raw attn score
        
        """
        attn_scores = F.softmax(scores / self.temperature, dim=1) # Apply softmax across time (T) with temperature scaling: (B, T, 1)

        if return_entropy:
            entropy = -torch.sum(attn_scores * torch.log(attn_scores + 1e-8), dim=1).mean()
            return attn_scores, entropy
        else:
            return attn_scores # These scores are only used to rank/select Top-K chunks; not used for pooling.
        """
        
        