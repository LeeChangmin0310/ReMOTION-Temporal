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
        # -------- running std (EMA) buffer registering --------
        self.register_buffer("running_score_std", torch.tensor(1.0))

    @torch.no_grad()
    def _update_running_std(self, batch_std):
        self.running_score_std.mul_(0.9).add_(0.1 * batch_std)
    
    def forward(self, z, temperature=0.5, epoch=0):
        """
        Args:
            z (Tensor): (B, T, D) chunk embeddings
            return_entropy (bool): whether to return entropy

        Returns:
            attn_weights (Tensor): (B, T, 1)
            entropy (optional): scalar tensor for entropy loss
            running_score_std (float): running std of raw attention scores
                                        (EMA for std-aware adaptive temperature adjustment)
        """
        # 1) zero-mean raw scores
        scores     = self.scorer(z)                           # (B,T,1)
        raw_scores = scores - scores.mean(1, keepdim=True)    # zero-mean

        # 2) EMA σ  update
        self._update_running_std(raw_scores.detach().std().item())
        sigma = self.running_score_std.item()

        sigma_star = 0.8 if epoch < 12 else 1.2               # target σ*
        gamma      = (sigma_star / (sigma + 1e-4))
        gamma      = float(max(0.5, min(2.0, gamma)))         # clamp
        raw_scaled = raw_scores * gamma                       # scale
        alpha = None
        
        # --- choose attention kernel ----------------------
        if epoch < 15:                                                      # Phase 0:  Softmax diversity explore
            attn = torch.softmax(raw_scaled / temperature, dim=1)
            
        elif 15 <= epoch < 25:                                              # Phase 1:  α-Entmax discriminativity explore        
            alpha = 1.9 - 0.03 * (epoch - 15)                                     # α: 1.9 → 1.6  (10 ep ×0.03) linearly over 10ep
            attn = entmax.entmax_bisect(raw_scaled, alpha=alpha, dim=1)           # (B,T)
            
        else:                                                               # Phase 2:    scorer fine-tune
            attn  = None                                                          # use raw_scores only
        
        # 4) DEBUG
        if self.training and (torch.rand(1).item() < 0.05):   # 5 % debug
            print(f"[AttnScorer] epoch={epoch:02d} | σ={sigma:.3f} "
                  f"→ γ={gamma:.2f} | kernel={('soft','α','15','raw')[min(3,epoch//15)]}")

        return attn, raw_scaled, alpha