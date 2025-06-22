# ===============================================
# Lightweight auxiliary classifier for chunk CE
# ===============================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChunkAuxClassifier(nn.Module):
    """
    Chunk-level auxiliary head.
    Only used during Phase-1 to provide weak CE on Top-K chunks.
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dim: int = 48,
                 p_drop: float = 0.25):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        # Kaiming-normal is fine with GELU + LayerNorm
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, D) or (B, T, D) – flatten if needed.
        """
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
        return self.net(x)
