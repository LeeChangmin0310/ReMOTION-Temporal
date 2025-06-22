###############################################################
# GatedPooling – fixed-α option
###############################################################
import torch, entmax, torch.nn as nn, torch.nn.functional as F
from typing import Optional

class GatedPooling(nn.Module):
    """
    Entmax-attention × sigmoid-gate × dropout.
    If `fixed_alpha` is given, that value is always used;
    otherwise the default schedule (1.7→1.9) is applied.
    """

    def __init__(self,
                 input_dim: int,
                 p_drop: float = 0.1,
                 fixed_alpha: Optional[float] = None):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(p_drop)
        self.fixed_alpha = fixed_alpha       # ← new hyper-param

    # ----------------------------------------------------------
    def _get_alpha(self, epoch: Optional[int], training: bool) -> float:
        """Return entmax α according to schedule or fixed value."""
        if self.fixed_alpha is not None:
            return self.fixed_alpha          # constant α
        if (not training) or epoch is None:
            return 1.9                       # inference
        if epoch < 30:                       # Phase-0/1
            return 1.7
        return 1.7 + 0.2 * min((epoch - 30) / 19, 1.0)  # 1.7→1.9

    # ----------------------------------------------------------
    def forward(self, 
                x: torch.Tensor, 
                raw_scores: torch.Tensor,
                epoch: Optional[int] = None,
                return_weights: bool = False,
                return_entropy: bool = False) -> torch.Tensor:
        """
        x          : (B, T, D)
        raw_scores : (B, T, 1) logits from AttnScorer
        """
        alpha_g = self._get_alpha(epoch, self.training)

        a = entmax.entmax_bisect(raw_scores, alpha=alpha_g, dim=1)  # (B,T,1)
        g = self.gate(x)                                            # (B,T,D)

        pooled = torch.sum(a * self.drop(x * g), dim=1)             # (B,D)
        ent = -(a * (a + 1e-8).log()).sum(dim=1).mean()

        if return_weights and return_entropy:
            return pooled, a, ent
        if return_weights:
            return pooled, a
        if return_entropy:
            return pooled, ent
        return pooled
