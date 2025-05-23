'''
# =============================================================
# Slim‑MTDE (4‑Branch, Physiology‑tuned)
# =============================================================
import torch, torch.nn as nn, torch.nn.functional as F

# -------- utility -----------------------------------------------------------
def same_conv(cin, cout, k, d):
    """Same‑length Conv1d with automatic padding."""
    return nn.Conv1d(cin, cout, k, padding=int((k-1)*d/2), dilation=d)

# -------- pooling: softmax + channel gate -----------------------------------
class SoftmaxGatePool(nn.Module):
    """Temporal Softmax attention + channel‑wise sigmoid gate."""
    def __init__(self, C):
        super().__init__()
        self.attn = nn.Conv1d(C, 1, 1)   # time‑attention scores
        self.gate = nn.Linear(C, C)      # channel gate
    def forward(self, x):                # x:(B,C,T)
        w = torch.softmax(self.attn(x), dim=-1)
        pooled = (x * w).sum(-1)         # (B,C)
        return pooled * torch.sigmoid(self.gate(pooled))  # (B,C)

# -------- multi‑scale block --------------------------------------------------
class MultiScaleTemporalBlock(nn.Module):
    """
    RFs after MaxPool(2) :  6 | 34 | 60 | 120 frames
    Channel map          :  8 | 16 | 24 | 32  = 80 (GroupNorm 8)
    """
    def __init__(self, cin=24, emb=256, dp=0.2):
        super().__init__()
        self.br_us = nn.Sequential(            # Ultra‑short 6f
            same_conv(cin,  8, k=3, d=1), nn.GELU(), nn.MaxPool1d(2))
        self.br_s  = nn.Sequential(            # Short 34f
            same_conv(cin, 16, k=5, d=4), nn.GELU(), nn.MaxPool1d(2))
        self.br_m  = nn.Sequential(            # Medium 60f
            same_conv(cin, 24, k=5, d=15), nn.GELU(), nn.MaxPool1d(2))
        self.br_l  = nn.Sequential(            # Long 120f
            same_conv(cin, 32, k=3, d=30), nn.GELU(), nn.MaxPool1d(2))

        C_total = 80
        self.norm = nn.GroupNorm(8, C_total)   # 80/8 = 10
        self.drop = nn.Dropout(dp)
        self.pool = SoftmaxGatePool(C_total)
        self.proj = nn.Linear(C_total, emb)

    def forward(self, x):                      # (B,C,T)
        cat = torch.cat([self.br_us(x),
                         self.br_s (x),
                         self.br_m (x),
                         self.br_l (x)], dim=1)     # (B,80,T')
        cat = self.drop(self.norm(cat))
        return F.gelu(self.proj(self.pool(cat)))    # (B,emb)

# -------- slim stem ----------------------------------------------------------
class SlimStem(nn.Sequential):
    """Conv5 (upslope) + Conv3‑stride2 (learnable downsampling)."""
    def __init__(self, cin=1, cout=24, dp=0.1):
        super().__init__(
            nn.Conv1d(cin, cout, 5, padding=2),
            nn.GroupNorm(4, cout), nn.GELU(),
            nn.Conv1d(cout, cout, 3, padding=1, stride=2),
            nn.Dropout(dp))

# -------- full encoder -------------------------------------------------------
class MTDE(nn.Module):
    """
    rPPG chunk (128 samples) -> 256‑D embedding.
    """
    def __init__(self,
                 in_channels: int = 1,
                 cnn_out_channels: int = 24,
                 embedding_dim: int = 256,
                 dropout_rate: float = 0.1):        
        super().__init__()
        self.stem = SlimStem(in_channels, cnn_out_channels, dropout_rate)
        self.mstb = MultiScaleTemporalBlock(cnn_out_channels, embedding_dim, dropout_rate)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)      # (B,1,T)
        elif x.shape[2] == 1: x = x.transpose(1,2)
        return self.mstb(self.stem(x))

'''
# =============================================================
# Slim‑MTDE  (3‑branch RF = 6 / 60 / 120  +  Softmax‑Gate pool)
# =============================================================
import torch, torch.nn as nn, torch.nn.functional as F

# -------------------------------------------------------------
# 1.  Softmax over time  +  channel‑wise sigmoid gate
# -------------------------------------------------------------
class SoftmaxGatePool(nn.Module):
    """
    Temporal soft‑attention followed by a per‑channel gate.
    x: (B, C, T)  ->  out: (B, C)
    """
    def __init__(self, C):
        super().__init__()
        self.attn = nn.Conv1d(C, 1, kernel_size=1)   # time‑score
        self.gate = nn.Linear(C, C)                  # channel gate

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=-1)      # (B,1,T)
        pooled = (x * w).sum(-1)                     # (B,C)
        return pooled
        # gated  = pooled * torch.sigmoid(self.gate(pooled))
        # return gated                                 # (B,C)

# -------------------------------------------------------------
# 2.  Multi‑scale block : RF 6 / 60 / 120 frames
# -------------------------------------------------------------
def same_conv(Cin, Cout, k, d):
    """Length‑preserving Conv1d (padding = (k‑1)*d/2)."""
    p = int((k - 1) * d / 2)
    return nn.Conv1d(Cin, Cout, kernel_size=k, padding=p, dilation=d)

class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, Cin=24, emb_dim=256, dropout=0.2):
        super().__init__()
        # Short  RF = 3 (Conv)  -> 6 after MaxPool
        self.br_short = nn.Sequential(
            same_conv(Cin, 16, k=3, d=1),
            nn.GELU(), nn.MaxPool1d(2)
        )
        # Medium RF = 29 -> 58 (≈2 s)
        self.br_mid = nn.Sequential(
            same_conv(Cin, 24, k=5, d=15),
            nn.GELU(), nn.MaxPool1d(2)
        )
        # Long   RF = 121 -> 242 (covers whole 128‑frame chunk)
        self.br_long = nn.Sequential(
            same_conv(Cin, 32, k=3, d=60),
            nn.GELU(), nn.MaxPool1d(2)
        )

        C_total = 16 + 24 + 32
        self.norm  = nn.GroupNorm(8, C_total)
        self.drop  = nn.Dropout(dropout)
        self.pool  = SoftmaxGatePool(C_total)
        self.proj  = nn.Linear(C_total, emb_dim)

    def forward(self, x):                 # x: (B,C,T)
        cat = torch.cat([self.br_short(x),
                         self.br_mid  (x),
                         self.br_long (x)], dim=1)   # Same T′ by design
        cat = self.drop(self.norm(cat))
        return F.gelu(self.proj(self.pool(cat)))     # (B, emb_dim)

# -------------------------------------------------------------
# 3.  Slim Stem  (Conv5 → Conv3 stride‑2)
# -------------------------------------------------------------
class SlimStem(nn.Sequential):
    def __init__(self, Cin=1, Cout=24, drop=0.1):
        super().__init__(
            nn.Conv1d(Cin, Cout, kernel_size=5, padding=2),
            nn.GroupNorm(4, Cout), nn.GELU(),
            nn.Conv1d(Cout, Cout, kernel_size=3, padding=1, stride=2),
            nn.Dropout(drop)
        )

# -------------------------------------------------------------
# 4.  Full MTDE encoder
# -------------------------------------------------------------
class MTDE(nn.Module):
    """
    rPPG chunk (128 samples) -> 256‑D embedding.
    """
    def __init__(self,
                 in_channels: int = 1,
                 cnn_out_channels: int = 24,
                 embedding_dim: int = 256,
                 dropout_rate: float = 0.1):        
        super().__init__()
        self.stem = SlimStem(in_channels, cnn_out_channels, dropout_rate)
        self.mstb = MultiScaleTemporalBlock(cnn_out_channels, embedding_dim, dropout_rate)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):                 # x: (B,T) or (B,T,1)
        if x.dim() == 2:
            x = x.unsqueeze(1)            # (B,1,T)
        elif x.shape[2] == 1:
            x = x.transpose(1, 2)         # (B,1,T)
        z = self.stem(x)                  # (B,C,T/2)
        return self.mstb(z)               # (B, emb_dim)

'''
# =============================================
# MTDEv13: Optimized Multi-Scale Temporal Dynamics Encoder
# =============================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedTemporalPooling(nn.Module):
    """
    Learnable weighted pooling over the temporal dimension.
    Computes attention scores per time step via a 1x1 convolution,
    applies softmax across time, and returns the weighted sum.
    Input: x of shape (B, C, T)
    Output: Tensor of shape (B, C)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            # 1x1 conv reduces channel dim to 1 for score computation
            nn.Conv1d(input_dim, 1, kernel_size=1),
            # softmax over temporal dimension
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        attn_weights = self.attn(x)         # (B, 1, T)
        # weighted sum over time → (B, C)
        return torch.sum(x * attn_weights, dim=-1)

class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_channels=24, embedding_dim=256, dropout_rate=0.2):
        super().__init__()

        # Short-range: RF ≈ 5
        self.branch_short = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.MaxPool1d(2)
        )

        # Medium-range: RF ≈ (5-1)*2+1 = 9
        self.branch_med = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.MaxPool1d(2)
        )

        # Long-range: RF ≈ (3-1)*15+1 = 31
        self.branch_long = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=15, dilation=15),
            nn.GELU(),
            nn.MaxPool1d(2)
        )

        self.norm = nn.GroupNorm(num_groups=8, num_channels=72)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = WeightedTemporalPooling(input_dim=72)  # attention over time
        self.fc = nn.Linear(72, embedding_dim)

    def forward(self, x):
        out_s = self.branch_short(x)
        out_m = self.branch_med(x)
        out_l = self.branch_long(x)
        cat = torch.cat([out_s, out_m, out_l], dim=1)  # (B, 72, T)
        cat = self.norm(cat)
        cat = self.dropout(cat)
        pooled = self.pool(cat)  # (B, 72)
        return F.gelu(self.fc(pooled))  # (B, embedding_dim)
 


class MTDE(nn.Module):
    """
    Multi-Temporal Dynamics Encoder (MTDE) for rPPG-based emotion recognition.
    - Stem: two Conv1d layers + stride-2 downsampling → reduces T by half
    - MultiScaleTemporalBlock: captures multi-scale temporal features
    Output: session-level embedding of size embedding_dim
    """
    def __init__(self,
                 in_channels: int = 1,
                 cnn_out_channels: int = 24,
                 embedding_dim: int = 256,
                 dropout_rate: float = 0.1):
        super().__init__()
        # ------------------------------------------
        # Stem: local feature extraction + downsampling
        # ------------------------------------------
        self.stem = nn.Sequential(
            # capture local pulse upslope (RF≈7)
            nn.Conv1d(in_channels,
                      cnn_out_channels,
                      kernel_size=7,
                      padding=3),
            nn.GroupNorm(num_groups=max(1, cnn_out_channels//4),
                         num_channels=cnn_out_channels),
            nn.GELU(),
            # deeper local features
            nn.Conv1d(cnn_out_channels,
                      cnn_out_channels,
                      kernel_size=5,
                      padding=2),
            nn.GroupNorm(num_groups=max(1, cnn_out_channels//4),
                         num_channels=cnn_out_channels),
            nn.GELU(),
            # downsample temporal dim by 2
            nn.Conv1d(cnn_out_channels,
                      cnn_out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=2),
            nn.Dropout(dropout_rate)
        )

        # Multi-scale block for mid/long-range
        self.multi_scale_block = MultiScaleTemporalBlock(
            in_channels=cnn_out_channels,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate
        )

        # Initialize weights for Conv and Linear
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) or (B, T, 1)
        if x.dim() == 2:
            x = x.unsqueeze(1)       # → (B, 1, T)
        elif x.shape[2] == 1:
            x = x.transpose(1, 2)    # → (B, 1, T)

        # Stem: local conv + downsampling → (B, C, T/2)
        x_cnn = self.stem(x)
        # Multi-scale temporal encoding → (B, embedding_dim)
        emb = self.multi_scale_block(x_cnn)
        return emb
'''
'''
# =============================================
# MTDEv11: Optimized Multi-Scale Temporal Dynamics Encoder
# =============================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedTemporalPooling(nn.Module):
    """
    Learnable weighted pooling over temporal dimension.
    Allows the model to softly attend to important time steps.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            # 1x1 conv to compute attention score per time step
            nn.Conv1d(input_dim, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):  # x: (B, C, T)
        attn = self.attn(x)       # (B, 1, T)
        return torch.sum(x * attn, dim=-1)  # (B, C)


class MultiScaleTemporalBlock(nn.Module):
    """
    Multi-scale temporal encoder that captures
      • short RF ≈ 3 frames
      • medium RF ≈  (3−1)*4+1 = 9 frames
      • long   RF ≈ (3−1)*16+1 = 33 frames
    using dilation and varied kernel sizes.
    """
    def __init__(self, in_channels=24, embedding_dim=256, dropout_rate=0.2):
        super().__init__()
        # Branch for short-range patterns (kernel=3)
        self.branch_short = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.MaxPool1d(2)
        )
        # Branch for medium-range patterns (kernel=5)
        self.branch_med = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=5, padding=2, dilation=1),
            nn.GELU(),
            nn.MaxPool1d(2)
        )
        # Branch for long-range patterns (kernel=3, dilation=4)
        self.branch_long = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.MaxPool1d(2)
        )

        # combine branch outputs: total channels = 16+24+32 = 72
        self.norm = nn.GroupNorm(num_groups=8, num_channels=72)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = WeightedTemporalPooling(input_dim=72)
        self.fc = nn.Linear(72, embedding_dim)

    def forward(self, x):  # x: (B, C, T)
        out_s = self.branch_short(x)
        out_m = self.branch_med(x)
        out_l = self.branch_long(x)
        x = torch.cat([out_s, out_m, out_l], dim=1)  # (B, 72, T')
        x = self.norm(x)
        x = self.dropout(x)
        x = self.pool(x)  # (B, 72)
        x = self.fc(x)    # (B, embedding_dim)
        return F.gelu(x)


class MTDE(nn.Module):
    def __init__(self, in_channels=1, cnn_out_channels=24, embedding_dim=256, dropout_rate=0.1):
        super().__init__()
        # --------------------------------------------
        # Stem: two convolutional layers + downsampling
        # --------------------------------------------
        self.stem = nn.Sequential(
            # wider receptive field to capture pulse upswing
            nn.Conv1d(in_channels, cnn_out_channels, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups=max(1, cnn_out_channels//4), num_channels=cnn_out_channels),
            nn.GELU(),
            # additional conv for deeper local features
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=max(1, cnn_out_channels//4), num_channels=cnn_out_channels),
            nn.GELU(),
            # stride conv for downsampling instead of maxpool
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1, stride=2),
            nn.Dropout(dropout_rate)
        )

        # Multi-scale temporal block for mid/long-range patterns
        self.multi_scale_block = MultiScaleTemporalBlock(
            in_channels=cnn_out_channels,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B, T, 1)
        # reshape to (B, C, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.shape[2] == 1:
            x = x.transpose(1, 2)
        # local feature extraction + downsampling
        x_cnn = self.stem(x)
        # multi-scale temporal encoding
        emb = self.multi_scale_block(x_cnn)
        return emb
'''
'''
# =============================================
# MTDEv10: Multi-Scale Temporal Feature Extraction(16, 24, 32) using CNN + WeightedTemporalPooling
# Multi-scale Temporal Dynamics Encoder
# --------------------------------------------------------------
# Purpose:
#   - Extract rich and discriminative temporal features from rPPG signals.
#   - Replace static GAP with learnable temporal pooling to preserve emotion-relevant features.
#   - Discard SEBlock due to negligible gain in temporal domain.
#   - Designed for chunk-length 128 rPPG input (e.g., MAHNOB-HCI).
#   - Utilize dropout and GELU for regularization and non-linearity.
#   - GroupNorm for small-batch training stability.
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedTemporalPooling(nn.Module):
    """
    Learnable weighted pooling over temporal dimension.
    Allows the model to softly attend to important time steps.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(input_dim, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):  # x: (B, C, T)
        attn = self.attn(x)  # (B, 1, T)
        return torch.sum(x * attn, dim=-1)  # (B, C)


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_channels=24, embedding_dim=128, dropout_rate=0.2):
        super(MultiScaleTemporalBlock, self).__init__()

        # Multi-scale CNN branches to capture different temporal receptive fields
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=5, padding=2),
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
        ])

        self.norm = nn.GroupNorm(num_groups=4, num_channels=72)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = WeightedTemporalPooling(input_dim=72)
        self.fc = nn.Linear(72, embedding_dim)

    def forward(self, x):  # x: (B, C, T)
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1)  # (B, 72, T')
        x = self.norm(x)
        x = self.dropout(x)
        x = self.pool(x)  # (B, 72)
        x = self.fc(x)    # (B, embedding_dim)
        x = F.gelu(x)
        return x


class MTDE(nn.Module):
    def __init__(self, in_channels=1, cnn_out_channels=24, embedding_dim=256, dropout_rate=0.1):
        super(MTDE, self).__init__()

        # Initial convolutional layer to extract local temporal patterns
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=cnn_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )

        # Multi-scale temporal encoder block
        self.multi_scale_block = MultiScaleTemporalBlock(
            in_channels=cnn_out_channels,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B, T, 1)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, T) → (B, 1, T)
        elif x.shape[2] == 1:
            x = x.transpose(1, 2)  # (B, T, 1) → (B, 1, T)

        x_cnn = self.stem(x)
        emb = self.multi_scale_block(x_cnn)  # (B, embedding_dim)
        return emb
'''

'''
# =============================================
# TemporalBranch_v9: Multi-Scale Temporal Feature Extraction using CNN(24, 24, 24) + WeightedTemporalPooling
# --------------------------------------------------------------
# Purpose:
#   - Extract rich and discriminative temporal features from rPPG signals.
#   - Replace static GAP with learnable temporal pooling to preserve emotion-relevant features.
#   - Discard SEBlock due to negligible gain in temporal domain.
#   - Designed for chunk-length 128 rPPG input (e.g., MAHNOB-HCI).
#   - Utilize dropout and GELU for regularization and non-linearity.
#   - GroupNorm for small-batch training stability.
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedTemporalPooling(nn.Module):
    """
    Learnable weighted pooling over temporal dimension.
    Allows the model to softly attend to important time steps.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(input_dim, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):  # x: (B, C, T)
        attn = self.attn(x)  # (B, 1, T)
        return torch.sum(x * attn, dim=-1)  # (B, C)


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_channels=24, embedding_dim=128, dropout_rate=0.2):
        super(MultiScaleTemporalBlock, self).__init__()

        # Multi-scale CNN branches to capture different temporal receptive fields
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=3, padding=1),
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=5, padding=2),
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=7, padding=3),
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
        ])

        self.norm = nn.GroupNorm(num_groups=4, num_channels=72)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = WeightedTemporalPooling(input_dim=72)
        self.fc = nn.Linear(72, embedding_dim)

    def forward(self, x):  # x: (B, C, T)
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1)  # (B, 72, T')
        x = self.norm(x)
        x = self.dropout(x)
        x = self.pool(x)  # (B, 72)
        x = self.fc(x)    # (B, embedding_dim)
        x = F.gelu(x)
        return x


class TemporalBranch(nn.Module):
    def __init__(self, in_channels=1, cnn_out_channels=24, embedding_dim=256, dropout_rate=0.1):
        super(TemporalBranch, self).__init__()

        # Initial convolutional layer to extract local temporal patterns
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=cnn_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )

        # Multi-scale temporal encoder block
        self.multi_scale_block = MultiScaleTemporalBlock(
            in_channels=cnn_out_channels,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B, T, 1)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, T) → (B, 1, T)
        elif x.shape[2] == 1:
            x = x.transpose(1, 2)  # (B, T, 1) → (B, 1, T)

        x_cnn = self.stem(x)
        emb = self.multi_scale_block(x_cnn)  # (B, embedding_dim)
        return emb
'''
'''
# =============================================
# TemporalBranch_v8: Multi-Scale Temporal Feature Extraction using CNN + (SE Block) for 256 emb dim
# SE Block is not adequate
# --------------------------------------------------------------
# Purpose:
#   - Extract rich and discriminative temporal features from rPPG signals.
#   - Capture local and global temporal patterns for emotion recognition.
#   - Designed for chunk-length 128 rPPG input (e.g., MAHNOB-HCI).
#   - Incorporate dropout for regularization.
#   - Utilize GELU activation for non-linearity.
#   - Employ batch normalization for stable training.
#   - Squeeze-and-Excitation (SE) block for channel-wise attention.
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for 1D temporal features.

    This block adaptively recalibrates channel-wise feature responses
    by explicitly modeling interdependencies between channels.
    Useful for emphasizing emotion-relevant filters in rPPG-based signals.

    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for bottleneck in FC layers.
    """
    def __init__(self, channels, reduction=4):
        super(SEBlock1D, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T) → SE weights: (B, C, 1)
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y  # Channel-wise reweighting


class MultiScaleTemporalBlock(nn.Module):
    """
    Multi-Scale Temporal Block for rPPG feature extraction.

    This block captures temporal dependencies at multiple scales
    (local, mid-range, global) using different kernel sizes.
    Additionally, it incorporates an SE block to emphasize 
    emotion-relevant filters in the temporal domain.

    Args:
        in_channels (int): Input channel dimension.
        embedding_dim (int): Output embedding size.
        dropout_rate (float): Dropout probability.
    """
    def __init__(self, in_channels=24, embedding_dim=128, dropout_rate=0.2):
        super(MultiScaleTemporalBlock, self).__init__()
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=3, padding=1),  # Local
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=5, padding=2), # Mid-range
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=7, padding=3), # Global
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
        ])

        # self.norm = nn.BatchNorm1d(48)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=72)
        self.dropout = nn.Dropout(dropout_rate)

        # self.se = SEBlock1D(channels=72, reduction=4)  # Channel attention

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(72, embedding_dim)

    def forward(self, x):
        """
        Forward pass for multi-scale temporal extraction with SE.

        Input:
            x: (B, C, T)
        Returns:
            embedding: (B, embedding_dim)
        """
        outs = [branch(x) for branch in self.branches]  # [(B, C_i, T')]
        x = torch.cat(outs, dim=1)                      # (B, 48, T')
        x = self.norm(x)
        x = self.dropout(x)
        # x = self.se(x)                                  # Channel reweighting
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        x = F.gelu(x)
        return x




# Final Temporal Branch using Multi-Scale CNN
class TemporalBranch(nn.Module):
    def __init__(self, in_channels=1, cnn_out_channels=24, embedding_dim=256, dropout_rate=0.1):
        """
        Temporal Branch v5 for extracting temporal features from rPPG signals using a multi-scale CNN approach.

        Args:
            in_channels (int): Number of input channels (1 for rPPG).
            cnn_out_channels (int): Number of output channels for CNN feature extraction.
            embedding_dim (int): Dimensionality of the final embedding.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(TemporalBranch, self).__init__()

        # CNN path for local temporal feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=cnn_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )

        # Multi-Scale Temporal Block to capture diverse temporal patterns
        self.multi_scale_block = MultiScaleTemporalBlock(
            in_channels=cnn_out_channels, 
            embedding_dim=embedding_dim, 
            dropout_rate=dropout_rate
        )
        
    
        # Final Fully Connected layer to project the multi-scale features into the final embedding space
        # self.fc = nn.Linear(embedding_dim, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier uniform distribution.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass for extracting temporal features.

        Args:
            x (Tensor): Input tensor of shape (B, T, 1), where B is batch size and T is sequence length.

        Returns:
            Tensor: Final embedding of shape (B, embedding_dim).
        """
        # Input x: (B, T, 1) -> (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)     # (B, T) → (B, 1, T) for Conv1d
        elif x.shape[2] == 1:
            x = x.transpose(1, 2)  # (B, T, 1) → (B, 1, T) for Conv1d
                                   # → (B, T, 1) for LSTM, Transformer 

        # Apply CNN for local temporal feature extraction
        x_cnn = self.stem(x)
        
        # Apply multi-scale temporal feature extraction
        emb = self.multi_scale_block(x_cnn)  # (B, embedding_dim)

        # Apply final FC layer to refine the embedding
        # embedding = self.fc(embedding)  # (B, embedding_dim)

        return emb
'''

'''
# =============================================
# TemporalBranch_v7: Multi-Scale Temporal Feature Extraction using CNN + (SE Block) for 256 emb dim
# SE Block is not adequate
# --------------------------------------------------------------
# Purpose:
#   - Extract rich and discriminative temporal features from rPPG signals.
#   - Capture local and global temporal patterns for emotion recognition.
#   - Designed for chunk-length 128 rPPG input (e.g., MAHNOB-HCI).
#   - Incorporate dropout for regularization.
#   - Utilize GELU activation for non-linearity.
#   - Employ batch normalization for stable training.
#   - Squeeze-and-Excitation (SE) block for channel-wise attention.
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for 1D temporal features.

    This block adaptively recalibrates channel-wise feature responses
    by explicitly modeling interdependencies between channels.
    Useful for emphasizing emotion-relevant filters in rPPG-based signals.

    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for bottleneck in FC layers.
    """
    def __init__(self, channels, reduction=4):
        super(SEBlock1D, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T) → SE weights: (B, C, 1)
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y  # Channel-wise reweighting


class MultiScaleTemporalBlock(nn.Module):
    """
    Multi-Scale Temporal Block for rPPG feature extraction.

    This block captures temporal dependencies at multiple scales
    (local, mid-range, global) using different kernel sizes.
    Additionally, it incorporates an SE block to emphasize 
    emotion-relevant filters in the temporal domain.

    Args:
        in_channels (int): Input channel dimension.
        embedding_dim (int): Output embedding size.
        dropout_rate (float): Dropout probability.
    """
    def __init__(self, in_channels=24, embedding_dim=128, dropout_rate=0.2):
        super(MultiScaleTemporalBlock, self).__init__()
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=3, padding=1),  # Local
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=5, padding=2), # Mid-range
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=7, padding=3), # Global
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
        ])

        # self.norm = nn.BatchNorm1d(48)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=72)
        self.dropout = nn.Dropout(dropout_rate)

        self.se = SEBlock1D(channels=72, reduction=4)  # Channel attention

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(72, embedding_dim)

    def forward(self, x):
        """
        Forward pass for multi-scale temporal extraction with SE.

        Input:
            x: (B, C, T)
        Returns:
            embedding: (B, embedding_dim)
        """
        outs = [branch(x) for branch in self.branches]  # [(B, C_i, T')]
        x = torch.cat(outs, dim=1)                      # (B, 48, T')
        x = self.norm(x)
        x = self.dropout(x)
        x = self.se(x)                                  # Channel reweighting
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        x = F.gelu(x)
        return x




# Final Temporal Branch using Multi-Scale CNN
class TemporalBranch(nn.Module):
    def __init__(self, in_channels=1, cnn_out_channels=24, embedding_dim=256, dropout_rate=0.1):
        """
        Temporal Branch v5 for extracting temporal features from rPPG signals using a multi-scale CNN approach.

        Args:
            in_channels (int): Number of input channels (1 for rPPG).
            cnn_out_channels (int): Number of output channels for CNN feature extraction.
            embedding_dim (int): Dimensionality of the final embedding.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(TemporalBranch, self).__init__()

        # CNN path for local temporal feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=cnn_out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )

        # Multi-Scale Temporal Block to capture diverse temporal patterns
        self.multi_scale_block = MultiScaleTemporalBlock(
            in_channels=cnn_out_channels, 
            embedding_dim=embedding_dim, 
            dropout_rate=dropout_rate
        )
        
    
        # Final Fully Connected layer to project the multi-scale features into the final embedding space
        # self.fc = nn.Linear(embedding_dim, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier uniform distribution.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass for extracting temporal features.

        Args:
            x (Tensor): Input tensor of shape (B, T, 1), where B is batch size and T is sequence length.

        Returns:
            Tensor: Final embedding of shape (B, embedding_dim).
        """
        # Input x: (B, T, 1) -> (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)     # (B, T) → (B, 1, T) for Conv1d
        elif x.shape[2] == 1:
            x = x.transpose(1, 2)  # (B, T, 1) → (B, 1, T) for Conv1d
                                   # → (B, T, 1) for LSTM, Transformer 

        # Apply CNN for local temporal feature extraction
        x_cnn = self.stem(x)
        
        # Apply multi-scale temporal feature extraction
        emb = self.multi_scale_block(x_cnn)  # (B, embedding_dim)

        # Apply final FC layer to refine the embedding
        # embedding = self.fc(embedding)  # (B, embedding_dim)

        return emb
'''
'''
# =============================================
# TemporalBranch_v6: Multi-Scale Temporal Feature Extraction using CNN + SE Block
# --------------------------------------------------------------
# Purpose:
#   - Extract rich and discriminative temporal features from rPPG signals.
#   - Capture local and global temporal patterns for emotion recognition.
#   - Designed for chunk-length 128 rPPG input (e.g., MAHNOB-HCI).
#   - Incorporate dropout for regularization.
#   - Utilize GELU activation for non-linearity.
#   - Employ batch normalization for stable training.
#   - Squeeze-and-Excitation (SE) block for channel-wise attention.
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for 1D temporal features.

    This block adaptively recalibrates channel-wise feature responses
    by explicitly modeling interdependencies between channels.
    Useful for emphasizing emotion-relevant filters in rPPG-based signals.

    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for bottleneck in FC layers.
    """
    def __init__(self, channels, reduction=4):
        super(SEBlock1D, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T) → SE weights: (B, C, 1)
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y  # Channel-wise reweighting


class MultiScaleTemporalBlock(nn.Module):
    """
    Multi-Scale Temporal Block for rPPG feature extraction.

    This block captures temporal dependencies at multiple scales
    (local, mid-range, global) using different kernel sizes.
    Additionally, it incorporates an SE block to emphasize 
    emotion-relevant filters in the temporal domain.

    Args:
        in_channels (int): Input channel dimension.
        embedding_dim (int): Output embedding size.
        dropout_rate (float): Dropout probability.
    """
    def __init__(self, in_channels=48, embedding_dim=128, dropout_rate=0.3):
        super(MultiScaleTemporalBlock, self).__init__()
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, 8, kernel_size=3, padding=1),  # Local
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 16, kernel_size=5, padding=2), # Mid-range
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=7, padding=3), # Global
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
        ])

        # self.norm = nn.BatchNorm1d(48)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=48)
        self.dropout = nn.Dropout(dropout_rate)

        self.se = SEBlock1D(channels=48, reduction=8)  # Channel attention

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, embedding_dim)

    def forward(self, x):
        """
        Forward pass for multi-scale temporal extraction with SE.

        Input:
            x: (B, C, T)
        Returns:
            embedding: (B, embedding_dim)
        """
        outs = [branch(x) for branch in self.branches]  # [(B, C_i, T')]
        x = torch.cat(outs, dim=1)                      # (B, 48, T')
        x = self.norm(x)
        x = self.dropout(x)
        # x = self.se(x)                                  # Channel reweighting
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x




# Final Temporal Branch using Multi-Scale CNN
class TemporalBranch(nn.Module):
    def __init__(self, in_channels=1, cnn_out_channels=48, embedding_dim=128, dropout_rate=0.3):
        """
        Temporal Branch v5 for extracting temporal features from rPPG signals using a multi-scale CNN approach.

        Args:
            in_channels (int): Number of input channels (1 for rPPG).
            cnn_out_channels (int): Number of output channels for CNN feature extraction.
            embedding_dim (int): Dimensionality of the final embedding.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(TemporalBranch, self).__init__()

        # CNN path for local temporal feature extraction
        self.conv = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2)  # Single-scale CNN
        self.norm = nn.GroupNorm(num_groups=4, num_channels=cnn_out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

        # Multi-Scale Temporal Block to capture diverse temporal patterns
        self.multi_scale_block = MultiScaleTemporalBlock(
            in_channels=cnn_out_channels, 
            embedding_dim=embedding_dim, 
            dropout_rate=dropout_rate
        )

        # Final Fully Connected layer to project the multi-scale features into the final embedding space
        # self.fc = nn.Linear(embedding_dim, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier uniform distribution.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass for extracting temporal features.

        Args:
            x (Tensor): Input tensor of shape (B, T, 1), where B is batch size and T is sequence length.

        Returns:
            Tensor: Final embedding of shape (B, embedding_dim).
        """
        # Input x: (B, T, 1) -> (B, 1, T)
        if x.dim() == 2:
            x_cnn = x.unsqueeze(1)     # (B, T) → (B, 1, T) for Conv1d
        elif x.shape[2] == 1:
            x_cnn = x.transpose(1, 2)  # (B, T, 1) → (B, 1, T) for Conv1d
                                   # → (B, T, 1) for LSTM, Transformer 

        # Apply CNN for local temporal feature extraction
        x_cnn = self.pool(self.act(self.norm(self.conv(x_cnn))))  # (B, C, T//2)
        x_cnn = self.dropout(x_cnn)
        
        # Apply multi-scale temporal feature extraction
        embedding = self.multi_scale_block(x_cnn)  # (B, embedding_dim)

        # Apply final FC layer to refine the embedding
        # embedding = self.fc(embedding)  # (B, embedding_dim)

        return embedding
'''
'''
# =============================================
# TemporalBranch_v5: Multi-Scale Temporal Feature Extraction using CNN
# --------------------------------------------------------------
# Purpose:
#   - Extract rich and discriminative temporal features from rPPG signals.
#   - Capture local and global temporal patterns for emotion recognition.
#   - Designed for chunk-length 128 rPPG input (e.g., MAHNOB-HCI).
#   - Incorporate dropout for regularization.
#   - Utilize GELU activation for non-linearity.
#   - Employ batch normalization for stable training.
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Scale Temporal Feature Extraction using CNNclass MultiScaleTemporalBlock(nn.Module):
class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_channels=48, embedding_dim=128, dropout_rate=0.3):
        """
        Multi-Scale Temporal Block for feature extraction.

        Args:
            in_channels (int): Number of input channels (64 after CNN feature extraction).
            embedding_dim (int): Dimensionality of the final feature embedding.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(MultiScaleTemporalBlock, self).__init__()
        # Define multiple branches with different kernel sizes to capture temporal features at different scales
        # MultiScaleTemporalBlock
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, 8, kernel_size=3, padding=1), # Local
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),  # Mid-range
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=7, padding=3), # Global
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
        ])

        # Normalization layer to normalize across the 3 branches
        self.norm = nn.BatchNorm1d(48)  # 3 branches * 16 channels each
        self.dropout = nn.Dropout(dropout_rate)

        # Global Average Pooling to compress features along the time dimension
        self.gap = nn.AdaptiveAvgPool1d(1)  # (B, 48, 1)

        # Final fully connected layer to project features into the desired embedding dimension
        self.fc = nn.Linear(48, embedding_dim)

    def forward(self, x):
        """
        Forward pass for the multi-scale temporal feature extraction.

        Args:
            x (Tensor): Input tensor of shape (B, T, 64), where B is batch size and T is sequence length.

        Returns:
            Tensor: Final embedding of shape (B, embedding_dim).
        """
        # Reshape input to match (B, 1, T)
        # x = x.transpose(1, 2)  # (B, 64, T)

        # Apply each branch to the input to extract multi-scale features
        outs = [branch(x) for branch in self.branches]  # List[(B, 16, T/2)]
        
        # Concatenate outputs from all branches along the channel dimension
        x = torch.cat(outs, dim=1)  # (B, 48, T/2)
        
        # Normalize across channels
        x = self.norm(x)
        
        # Apply dropout regularization
        x = self.dropout(x)
        
        # Apply Global Average Pooling across the temporal dimension (T)
        x = self.gap(x).squeeze(-1)  # (B, 48)

        # Apply the fully connected layer to project features into the final embedding space
        x = self.fc(x)  # (B, embedding_dim)

        return x



# Final Temporal Branch using Multi-Scale CNN
class TemporalBranch(nn.Module):
    def __init__(self, in_channels=1, cnn_out_channels=48, embedding_dim=128, dropout_rate=0.3):
        """
        Temporal Branch v5 for extracting temporal features from rPPG signals using a multi-scale CNN approach.

        Args:
            in_channels (int): Number of input channels (1 for rPPG).
            cnn_out_channels (int): Number of output channels for CNN feature extraction.
            embedding_dim (int): Dimensionality of the final embedding.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(TemporalBranch, self).__init__()

        # CNN path for local temporal feature extraction
        self.conv = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2)  # Single-scale CNN
        self.norm = nn.GroupNorm(num_groups=4, num_channels=cnn_out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

        # Multi-Scale Temporal Block to capture diverse temporal patterns
        self.multi_scale_block = MultiScaleTemporalBlock(
            in_channels=cnn_out_channels, 
            embedding_dim=embedding_dim, 
            dropout_rate=dropout_rate
        )

        # Final Fully Connected layer to project the multi-scale features into the final embedding space
        # self.fc = nn.Linear(embedding_dim, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier uniform distribution.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass for extracting temporal features.

        Args:
            x (Tensor): Input tensor of shape (B, T, 1), where B is batch size and T is sequence length.

        Returns:
            Tensor: Final embedding of shape (B, embedding_dim).
        """
        # Input (B, T, 1) -> (B, 1, T)
        x_cnn = x.transpose(1, 2)

        # Apply CNN for local temporal feature extraction
        x_cnn = self.pool(self.act(self.norm(self.conv(x_cnn))))  # (B, C, T//2)
        x_cnn = self.dropout(x_cnn)
        
        # Apply multi-scale temporal feature extraction
        embedding = self.multi_scale_block(x_cnn)  # (B, embedding_dim)

        # Apply final FC layer to refine the embedding
        # embedding = self.fc(embedding)  # (B, embedding_dim)

        return embedding
'''

'''
# ==============================================
# TemporalBranch_v4Lite: Simplified Temporal Feature Extractor for rPPG
# --------------------------------------------------------------
# Purpose:
#   - Extract compact and discriminative features from 1D rPPG input (T=128)
#   - Reduce model complexity for small datasets like MAHNOB-HCI
#   - Focus on extracting local temporal features while maintaining generalization
#
# Components:
#   - Single-Scale CNN for local temporal encoding
#   - Shallow Residual Path with SE block (optional)
#   - Final projection FC layer for embedding
#
# Input:
#   - x: Tensor of shape (B, T, 1), where T = 128 (rPPG chunk)
# Output:
#   - embedding: Tensor of shape (B, embedding_dim), per-chunk representation
# ==============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation Block (Optional Channel Attention)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // reduction, 1),
            nn.GELU(),
            nn.Conv1d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)  # (B, C, 1)
        return x * scale

# Basic Dilated Residual Block (with GELU and Dropout)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, dilation, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=in_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(self.norm1(out))
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        
        return self.act(out + x)

# Lightweight TemporalBranch
class TemporalBranch(nn.Module):
    def __init__(self,
                 in_channels=1,
                 cnn_out_channels=32,
                 res_channels=32,
                 embedding_dim=128,
                 dropout_rate=0.1):
        super(TemporalBranch, self).__init__()

        # --- CNN Path: Local temporal pattern encoding ---
        self.conv = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2)  # Single-scale
        self.norm = nn.GroupNorm(num_groups=4, num_channels=cnn_out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

        # --- Residual Path: Capture temporal dependencies ---
        self.res_proj = nn.Conv1d(in_channels, res_channels, kernel_size=1)
        self.res_blocks = nn.Sequential(
            ResidualBlock(res_channels, res_channels, dilation=1, dropout_rate=dropout_rate),
        )
        self.se_block = SEBlock(res_channels, reduction=8)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # --- Final Projection ---
        fusion_dim = cnn_out_channels + res_channels
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim * 2, embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input: (B, T, 1)

        # --- CNN Path ---
        x_cnn = x.transpose(1, 2)  # (B, 1, T)
        x_cnn = self.pool(self.act(self.norm(self.conv(x_cnn))))  # (B, C, T//2)
        x_cnn = self.dropout(x_cnn)
        feat_cnn = x_cnn.mean(dim=2)  # (B, C)

        # --- Residual Path ---
        x_res = self.res_proj(x.transpose(1, 2))  # (B, res_channels, T)
        x_res = self.res_blocks(x_res)
        x_res = self.se_block(x_res)
        x_res = self.dropout(x_res)
        feat_res = self.gap(x_res).squeeze(-1)  # (B, res_channels)

        # --- Feature Fusion & Projection ---
        fused = torch.cat([feat_cnn, feat_res], dim=1)  # (B, fusion_dim)
        embedding = self.fc(fused)  # (B, embedding_dim)
        return embedding
'''
'''
# ==============================================
# TemporalBranch (v4): Dilated Residual + Multi-Scale CNN + SE Fusion
# --------------------------------------------------------------
# Purpose:
#   - Enhance chunk-level rPPG embeddings to be more discriminative.
#   - Increase intra-class diversity and inter-class separability.
#   - Incorporate global temporal dynamics with deep residual blocks.
#
# Components:
#   - Multi-Scale CNN (kernel 3/5/7) for local temporal pattern extraction
#   - Dilated Residual Blocks with increasing dilation rates (1,2,4)
#   - Squeeze-and-Excitation (SE) block for channel-wise attention
#   - Final projection FC with increased width for richer embedding
#
# Input:
#   - x: Tensor of shape (B, T, 1), where T = 128 (rPPG chunk)
# Output:
#   - embedding: Tensor of shape (B, embedding_dim), per-chunk representation
# ==============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // reduction, 1),
            nn.GELU(),
            nn.Conv1d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)  # (B, C, 1)
        return x * scale

# Dilated Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, dilation, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.act = nn.GELU()
        self.droupout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(self.bn1(out))
        out = self.droupout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.droupout(out)
        return self.act(out + x)

# Main TemporalBranch
class TemporalBranch(nn.Module):
    def __init__(self,
                 in_channels=1,
                 cnn_out_channels=32,
                 res_channels=32,
                 num_res_blocks=3,
                 embedding_dim=128,
                 dropout_rate=0.1):
        super(TemporalBranch, self).__init__()

        # === Multi-Scale CNN Path ===
        self.conv3 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=7, padding=3)
        self.norm_cnn = nn.GroupNorm(4, cnn_out_channels * 3)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

        # === Residual Path with Dilated Convs + SE ===
        self.res_proj = nn.Conv1d(in_channels, res_channels, kernel_size=1)
        self.res_blocks = nn.Sequential(
            ResidualBlock(res_channels, res_channels * 2, dilation=1, dropout_rate=dropout_rate),
            ResidualBlock(res_channels, res_channels * 2, dilation=2, dropout_rate=dropout_rate),
            ResidualBlock(res_channels, res_channels * 2, dilation=4, dropout_rate=dropout_rate),
        )
        self.se_block = SEBlock(res_channels, reduction=8)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # === Final Embedding Projection ===
        fusion_dim = (cnn_out_channels * 3) + res_channels
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),    # Increased width
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim * 2, embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input: (B, T, 1)
        # print(f"[TEMPORAL] Input requires_grad: {x.requires_grad}")

        # --- Local CNN Path ---
        x_cnn = x.transpose(1, 2)  # (B, 1, T)
        x3 = self.conv3(x_cnn)
        x5 = self.conv5(x_cnn)
        x7 = self.conv7(x_cnn)
        x_cat = torch.cat([x3, x5, x7], dim=1)               # (B, C*3, T)
        x_cat = self.pool(self.act(self.norm_cnn(x_cat)))    # (B, C*3, T//2)
        x_cat = self.dropout(x_cat)                          # (B, C*3, T//2)
        feat_cnn = x_cat.mean(dim=2)                         # (B, C*3)

        # --- Residual Path ---
        x_res = self.res_proj(x_cnn)                         # (B, res_channels, T)
        x_res = self.res_blocks(x_res)                       # (B, res_channels, T)
        x_res = self.se_block(x_res)                         # (B, res_channels, T)
        x_res = self.dropout(x_res)                          # (B, res_channels, T)
        feat_res = self.gap(x_res).squeeze(-1)               # (B, res_channels)

        # --- Feature Fusion ---
        fused = torch.cat([feat_cnn, feat_res], dim=1)       # (B, fusion_dim)
        embedding = self.fc(fused)                           # (B, embedding_dim)
        # print(f"[TEMPORAL] Input requires_grad: {x.requires_grad}")
        
        return embedding
'''

'''
# =============================================
# Temporal Branch (v3): Enhanced Multi-Scale CNN + Deep Residual + Projection
# Purpose: Maximize chunk-wise embedding diversity for discriminative temporal modeling
# Optimized for rPPG (T=128) input (MAHNOB-HCI, etc.)
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(in_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)

class TemporalBranch(nn.Module):
    def __init__(self,
                 in_channels=1,
                 cnn_out_channels=32,        # output channels per conv
                 res_channels=16,            # internal channels for residual
                 num_res_blocks=2,           # stacked residuals
                 embedding_dim=128,
                 dropout_rate=0.1):
        super(TemporalBranch, self).__init__()

        # === Multi-Scale CNN for Local Temporal Patterns ===
        self.conv3 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=7, padding=3)

        self.norm_cnn = nn.GroupNorm(4, cnn_out_channels * 3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # === Deep Residual Block for Global Dynamics ===
        self.res_proj = nn.Conv1d(in_channels, res_channels, kernel_size=1)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(res_channels, hidden_channels=res_channels * 2)
            for _ in range(num_res_blocks)
        ])

        # === Temporal Aggregation via Global Average Pooling ===
        self.gap = nn.AdaptiveAvgPool1d(1)

        # === Final Projection ===
        fusion_dim = (cnn_out_channels * 3) + res_channels
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, embedding_dim),
            # nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input x: (B, T, 1)

        # --- Local CNN Path ---
        x_cnn = x.transpose(1, 2)  # (B, 1, T)
        x3 = self.conv3(x_cnn)
        x5 = self.conv5(x_cnn)
        x7 = self.conv7(x_cnn)
        x_cat = torch.cat([x3, x5, x7], dim=1)               # (B, C*3, T)
        x_cat = self.pool(self.act(self.norm_cnn(x_cat)))    # (B, C*3, T//2)
        feat_cnn = x_cat.mean(dim=2)                         # (B, C*3)

        # --- Residual Global Path ---
        x_res = x.transpose(1, 2)                             # (B, 1, T)
        x_res = self.res_proj(x_res)                         # (B, res_channels, T)
        x_res = self.res_blocks(x_res)                       # (B, res_channels, T)
        feat_res = self.gap(x_res).squeeze(-1)               # (B, res_channels)

        # --- Feature Fusion and Embedding ---
        fused = torch.cat([feat_cnn, feat_res], dim=1)       # (B, fusion_dim)
        embedding = self.fc(fused)                           # (B, embedding_dim)

        return embedding, 0
'''

'''
# =============================================
# Temporal Branch (v2): Multi-Scale CNN + Dilated Residual + Self-Attention
# Purpose: Extract rich local-global temporal features from rPPG sequences
# Optimized for chunk-length 128 rPPG input (MAHNOB-HCI, etc.)
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBranch(nn.Module):
    def __init__(self,
                 in_channels=1,
                 cnn_out_channels=32,         # Output channels per conv
                 hidden_dim=64,               # Hidden size for residual block
                 embedding_dim=128,           # Final output embedding dim
                 dropout_rate=0.1):
        super(TemporalBranch, self).__init__()

        # === Multi-Scale CNN: Short-term pattern capture at multiple receptive fields ===
        self.conv3 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=7, padding=3)

        # Normalize and activate concatenated multi-scale features
        self.norm_cnn = nn.GroupNorm(4, cnn_out_channels * 3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Temporal downsampling

        # === Dilated Residual Block: Global temporal modeling ===
        self.dilated_conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv1d(hidden_dim, in_channels, kernel_size=3, padding=1)
        self.norm_res = nn.BatchNorm1d(in_channels)

        # === Self-Attention Pooling: Focus on important time steps ===
        self.attn_proj = nn.Linear(in_channels, 1)

        # === Final projection (CNN + AttentionPooled Residual) → Embedding ===
        fusion_dim = (cnn_out_channels * 3) + in_channels
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for conv and linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input: x shape = (B, T, 1) → rPPG sequence

        # --- CNN Path (local multi-scale) ---
        x_cnn = x.transpose(1, 2)  # (B, 1, T) for Conv1d
        x3 = self.conv3(x_cnn)
        x5 = self.conv5(x_cnn)
        x7 = self.conv7(x_cnn)

        x_cat = torch.cat([x3, x5, x7], dim=1)               # (B, C*3, T)
        x_cat = self.pool(self.act(self.norm_cnn(x_cat)))    # (B, C*3, T//2)
        feat_cnn = x_cat.mean(dim=2)                         # (B, C*3)

        # --- Residual Path with Dilated Conv (global) ---
        x_res = x.transpose(1, 2)                             # (B, 1, T)
        residual = x_res
        out = self.dilated_conv1(x_res)                      # (B, H, T)
        out = F.relu(out)
        out = self.dilated_conv2(out)                        # (B, 1, T)
        out = out + residual                                 # Residual connection
        out = self.norm_res(out)                             # (B, 1, T)

        # --- Self-Attention Pooling ---
        out_t = out.transpose(1, 2)                           # (B, T, 1)
        attn_scores = self.attn_proj(out_t)                  # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)         # (B, T, 1)
        pooled_res = (out_t * attn_weights).sum(dim=1)       # (B, 1)

        # --- Feature Fusion & Projection ---
        fused = torch.cat([feat_cnn, pooled_res], dim=1)     # (B, fusion_dim)
        embedding = self.fc(fused)                           # (B, embedding_dim)

        return embedding, attn_weights.squeeze(-1)           # Return embedding + attn weights
'''

'''
# =============================================
# Temporal Branch (v1): Multi-Scale CNN + Bi-LSTM Hybrid (Parallel structure)
# Purpose: Extract discriminative local (via CNN) and global (via Bi-LSTM) temporal features from rPPG signals
# Optimized for chunk-length 128 rPPG input in emotion recognition (e.g., MAHNOB-HCI)
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBranch(nn.Module):
    def __init__(self,
                 in_channels=1,
                 cnn_out_channels=32,       # Lowered for balance with LSTM
                 hidden_dim=64,             # BiLSTM hidden size (per direction)
                 lstm_layers=1,
                 embedding_dim=128,
                 dropout_rate=0.1):
        super(TemporalBranch, self).__init__()

        # Multi-Scale CNN: captures short-term temporal patterns
        self.conv3 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, cnn_out_channels, kernel_size=7, padding=3)

        self.norm = nn.GroupNorm(4, cnn_out_channels * 3)  # group size adjusted for 96 channels
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Reduce temporal resolution

        # BiLSTM: captures long-range sequential dependencies
        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )

        # Final projection: concat(CNN_out, LSTM_out) → projection
        fusion_dim = (cnn_out_channels * 3) + (2 * hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self._init_weights()

    def _init_weights(self):
        # Xavier init for conv + fc
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, 1)
        x_cnn = x.transpose(1, 2)  # (B, 1, T)

        # Multi-scale CNN branches
        x3 = self.conv3(x_cnn)
        x5 = self.conv5(x_cnn)
        x7 = self.conv7(x_cnn)

        x_cat = torch.cat([x3, x5, x7], dim=1)             # (B, C*3, T)
        x_cat = self.pool(self.act(self.norm(x_cat)))      # (B, C*3, T//2)
        feat_cnn = x_cat.mean(dim=2)                       # (B, C*3)

        # BiLSTM output
        lstm_out, _ = self.bilstm(x)                       # (B, T, 2*H)
        feat_lstm = lstm_out.mean(dim=1)                  # (B, 2*H)

        # Concatenate CNN + LSTM features
        fused = torch.cat([feat_cnn, feat_lstm], dim=1)   # (B, fusion_dim)

        # Final projection to embedding
        embedding = self.fc(fused)                        # (B, embedding_dim)

        return embedding, 0
'''

'''
#############################################
# Temporal Branch: 1DCNN-LSTM 
#############################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBranch(nn.Module):
    """
    TemporalBranch: Extracts discriminative temporal embeddings from input rPPG signals.
    Architecture: 
      1) 1D CNN with increased channels for local feature extraction,
      2) Bi-LSTM (2-layer, hidden=128) for richer temporal modeling,
      3) Skip connection from CNN-pooled features,
      4) Final FC layer producing embedding_dim=128 with mild dropout.

    This design aims to prevent feature collapse by increasing capacity
    and preserving local & temporal information through skip connections.
    """
    def __init__(self, 
                 in_channels=1, 
                 cnn_out_channels=128,  # Increased channels
                 lstm_hidden=128,       # Larger hidden size
                 lstm_layers=2, 
                 embedding_dim=128, 
                 dropout_rate=0.1):
        super(TemporalBranch, self).__init__()

        # 1D CNN for local temporal feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2),
            # GroupNorm with group=8 for 128 channels
            nn.GroupNorm(8, cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

        # Bi-directional LSTM with 2 layers and hidden=128
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        # LayerNorm for stable training
        self.lstm_layernorm = nn.LayerNorm(lstm_hidden * 2)

        # Skip connection: transform pooled CNN features to match LSTM output dimension (2H)
        self.skip_fc = nn.Linear(cnn_out_channels, lstm_hidden * 2)

        # Final embedding layer after combining LSTM output & skip connection
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout_rate)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initializes LSTM parameters with Xavier and Orthogonal schemes.
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, 1),
               representing chunk-level rPPG signal in time.

        Returns:
            embedding: Tensor (B, embedding_dim),
                       final chunk-level embedding.
            pooled_combined: Tensor (B, 2H),
                             intermediate representation before the final FC.
        """
        # (B, T, 1) -> (B, 1, T)
        x = x.transpose(1, 2)
        cnn_feat = self.cnn(x)                 # (B, C, T//4)
        cnn_feat = cnn_feat.transpose(1, 2)    # (B, T//4, C)

        # Skip connection from CNN-pooled features
        cnn_pooled = torch.mean(cnn_feat, dim=1)  # (B, C)
        skip = self.skip_fc(cnn_pooled)           # (B, 2H)

        # Process through Bi-LSTM
        lstm_out, _ = self.lstm(cnn_feat)         # (B, T//4, 2H)
        lstm_out = self.lstm_layernorm(lstm_out)  # (B, T//4, 2H)

        # Temporal average pooling
        pooled = torch.mean(lstm_out, dim=1)      # (B, 2H)

        # Combine LSTM-pooled features with skip connection
        pooled_combined = pooled + skip           # (B, 2H)

        # Final embedding
        embedding = self.fc(pooled_combined)      # (B, D)
        return embedding, pooled_combined
########################################################################################################
'''

'''
class TemporalBranch(nn.Module):
    """
    TemporalBranch: Extracts discriminative temporal embeddings from input rPPG signals.
    Architecture: 1D CNN → MaxPool → Bi-LSTM → (Self-Attention Pooling) → Fully Connected Layer → Embeddings
    Refactoring:
        - CNN + BiLSTM to prevent overfitting and enhance fine-grained feature extraction.
        - Optimized for short (T=128) 1D signals with limited dataset size.
        - Removed Self-Attention Pooling in Temporal Branch
    """
    def __init__(self, in_channels=1, cnn_out_channels=64, lstm_hidden=64, lstm_layers=1, embedding_dim=128, dropout_rate=0.2):
        super(TemporalBranch, self).__init__()

        # 1D CNN for local temporal feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2),
            # nn.BatchNorm1d(cnn_out_channels),
            nn.GroupNorm(4, cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool1d(kernel_size=2, stride=2),
            # nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm1d(cnn_out_channels),
            nn.GroupNorm(4, cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AvgPool1d(kernel_size=2, stride=2)
            # nn.MaxPool1d(kernel_size=2)
        )

        # Bidirectional LSTM to model longer-term temporal dependencies
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_layernorm = nn.LayerNorm(lstm_hidden * 2)

        # Self-attention pooling to summarize temporal sequence
        # self.attn_pool = SelfAttentionPooling(lstm_hidden * 2, temperature=0.1)

        # Final embedding layer with skip connection from CNN
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout_rate),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x):
        # Input: (B, T, 1) → (B, 1, T)
        x = x.transpose(1, 2)
        cnn_feat = self.cnn(x)                 # (B, C, T)
        cnn_feat = cnn_feat.transpose(1, 2)    # (B, T, C)

        lstm_out, _ = self.lstm(cnn_feat)      # (B, T, 2H)
        lstm_out = self.lstm_layernorm(lstm_out)    # (B, T, 2H)

        # Temporal average pooling before FC (optionally can replace with attention)
        pooled = torch.mean(lstm_out, dim=1)   # (B, 2H)
        embedding = self.fc(pooled)            # (B, D)

        return embedding, pooled  # Return both final and pre-FC pooled features
    
'''

'''
class SelfAttentionPooling(nn.Module):
    def __init__(self, lstm_dim, temperature=0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_dim // 2, 1)
        )
        self.temperature = temperature

    def forward(self, lstm_out):  # (B, T, D)
        attn_scores = self.attn(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores / (self.temperature + 1e-6), dim=1)
        pooled = torch.sum(attn_weights * lstm_out, dim=1)  # (B, D)
        return pooled
'''
'''
class TemporalBranch(nn.Module):
    """
    Extracts temporal features from the recovered rPPG signal
    using a 1D CNN followed by an LSTM with self-attention pooling
    to produce discriminative chunk-wise embeddings.
    """
    def __init__(self, in_channels=1, cnn_out_channels=64, lstm_hidden=128, lstm_layers=2, embedding_dim=256):
        super(TemporalBranch, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.attn_pool = SelfAttentionPooling(lstm_hidden * 2)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, embedding_dim),  # bidirectional → 2x hidden
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = x.transpose(1, 2)  # (batch, 1, seq_len)
        cnn_feat = self.cnn(x)  # (batch, cnn_out_channels, seq_len//4)
        cnn_feat = cnn_feat.transpose(1, 2)  # (batch, seq_len//4, cnn_out_channels)
        lstm_out, _ = self.lstm(cnn_feat)    # (batch, seq_len//4, 2 * lstm_hidden)
        pooled = self.attn_pool(lstm_out)    # (batch, 2 * lstm_hidden)
        embedding = self.fc(pooled)          # (batch, embedding_dim)
        return embedding
'''




'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineTemporalBranch(nn.Module):
    """
    BaselineTemporalBranch:
    논문 구조 기반. Chunk-level rPPG를 받아서 Flatten 후 FC 레이어로 임베딩을 생성.
    Conv1D + MaxPool + BatchNorm + Dropout → Flatten → FCs
    """
    def __init__(self, in_channels=1, embedding_dim=256):
        super(BaselineTemporalBranch, self).__init__()

        # Conv1D Layer 1
        self.conv1 = nn.Conv1d(in_channels, 512, kernel_size=50, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)

        # Conv1D Layer 2
        self.conv2 = nn.Conv1d(512, 256, kernel_size=25, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        # Flatten → FC layers
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 61, 256),
            nn.Sigmoid(),  # 논문은 sigmoid를 사용
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Sigmoid()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
        )

        self.out = nn.Linear(64, embedding_dim)  # 최종 임베딩 출력

    def forward(self, x):
        # x: (B, T, 1) → (B, 1, T)
        x = x.transpose(1, 2)

        x = self.conv1(x)      # (B, 512, T')
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.drop1(x)

        x = self.conv2(x)      # (B, 256, T'')
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.drop2(x)

        x = torch.flatten(x, start_dim=1)  # (B, 256*T)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)  # (B, embedding_dim)

        return x

'''