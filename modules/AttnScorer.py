###############################################################
# 1) AttnScorer – entropy-aware scaling + smoother α-schedule
###############################################################
import torch, entmax, torch.nn as nn, torch.nn.functional as F


class AttnScorer(nn.Module):
    """Two-layer MLP → scalar score per chunk.
    Kernel schedule:
        • P0  (0-14) :   softmax   (exploration)
        • P1  (15-29):   α-entmax  α 1.0→1.4  (sparsity ramp)
        • P2  (≥30) :   raw logits (hand off to GatedPooling)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1)
        )
        # Running std-dev of raw scores (EMA) for dynamic scaling
        self.register_buffer("running_score_std", torch.tensor(1.0))

    # ---------- helpers ----------
    @torch.no_grad()
    def _update_running_std(self, batch_std: float):
        # EMA: 0.9·prev + 0.1·new
        self.running_score_std.mul_(0.9).add_(0.1 * batch_std)

    # ---------- forward ----------
    def forward(self, z: torch.Tensor,
                temperature: float = 1.0,
                epoch: int = 0):
        """
        z : (B, T, D)
        Returns: attn (or None), scaled_logits, alpha
        """
        scores = self.scorer(z)                            # (B,T,1)
        scores = scores - scores.mean(1, keepdim=True)     # zero-mean

        # --- adaptive scale  -----------------------------------------
        batch_std = scores.detach().std().item()
        self._update_running_std(batch_std)
        sigma = max(self.running_score_std.item(), 1e-4)
        gamma = max(0.6, min(1.6, 0.9 / sigma))            # target σ* ≈0.9
        scaled = scores * gamma

        # --- kernel switch  ------------------------------------------
        attn, alpha = None, None
        if self.training:
            if epoch < 15:                                 # Phase-0
                attn = torch.softmax(scaled / temperature, dim=1)
            elif epoch < 30:                               # Phase-1
                sub   = epoch - 15       # 0‥14
                alpha = 1.0 + 0.4 * (sub / 14)             # 1.0→1.4
                attn  = entmax.entmax_bisect(scaled, alpha=alpha, dim=1)
            # Phase-2 : attn=None → raw logits only
        else:                                              # inference
            attn = entmax.entmax15(scaled, dim=1)

        # --- debug (5 % chance) ---------------------------
        if self.training and torch.rand(1).item() < 0.05:
            ker = "soft" if epoch < 15 else ("ent" if epoch < 30 else "raw")
            print(f"[Attn] ep:{epoch:02d} γ:{gamma:.2f} ker:{ker}")

        return attn, scaled, alpha
