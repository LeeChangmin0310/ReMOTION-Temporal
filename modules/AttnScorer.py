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
            nn.GELU(),                             # smoother, no saturation
            # nn.Tanh(),                             # Non-linear activation
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
        
        # --- scale factor γ --------------------------------
        if epoch < 20:
            sigma_star = 1.2
        else:
            sigma_star = 1.5
        gamma = torch.tensor(
            sigma_star / (self.running_score_std + 1e-4),
            device=raw_scores.device).clamp_(0.5, 3.0)
        raw_scaled = raw_scores * gamma
        
        # --- choose attention kernel ----------------------
        if epoch < 5:                        # Phase 0-a:  Softmax explore
            attn = torch.softmax(raw_scaled / temperature, dim=1)
        
        elif epoch < 20:                     # Phase 0-b:  α-Entmax warm-up
            # α 1.9→1.6 linearly
            alpha = 2.0 - 0.1 * max(0, epoch - 4)
            attn  = entmax.entmax_bisect(raw_scaled, alpha=alpha, dim=1)
        
        elif epoch < 35:                     # Phase 1:    Entmax15 sparse
            attn  = entmax.entmax15(raw_scaled, dim=1)
        
        else:                                # Phase 2:    scorer frozen
            attn  = None                     # use raw_scores only
        
        return attn, raw_scores
        
        """
        attn_scores = F.softmax(scores / self.temperature, dim=1) # Apply softmax across time (T) with temperature scaling: (B, T, 1)

        if return_entropy:
            entropy = -torch.sum(attn_scores * torch.log(attn_scores + 1e-8), dim=1).mean()
            return attn_scores, entropy
        else:
            return attn_scores # These scores are only used to rank/select Top-K chunks; not used for pooling.
        """
        
        