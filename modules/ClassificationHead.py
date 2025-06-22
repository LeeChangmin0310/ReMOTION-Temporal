# ===============================================
# Session-level classifier head
# ===============================================
import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Simple 2-layer MLP for session embeddings.
    Receives the pooled representation from GatedPooling.
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dim: int = 96,
                 p_drop: float = 0.20):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, D)
        return self.mlp(x)
