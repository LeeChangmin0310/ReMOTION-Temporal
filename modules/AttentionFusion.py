#############################################
# AttentionFusion: Attention-based multi-branch fusion
#############################################

import torch
import torch.nn as nn
class AttentionFusion(nn.Module):
    def __init__(self, input_dims=[512, 256, 128], fusion_dim=1024):
        super(AttentionFusion, self).__init__()
        self.num_branches = len(input_dims)
        self.fusion_dim = fusion_dim

        # Project all to fusion_dim for consistent shape
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, fusion_dim) for in_dim in input_dims
        ])

        # Compute attention scores
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            ) for in_dim in input_dims
        ])

    def forward(self, branches):
        # branches: list of [B, D_i] tensors
        projected = [proj(b) for proj, b in zip(self.projections, branches)]  # [(B, fusion_dim)]
        attn_scores = [attn(b) for attn, b in zip(self.attention, branches)]  # [(B, 1)]
        attn_scores = torch.stack(attn_scores, dim=1)  # (B, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, N, 1)
        
        # print(f"[DEBUG] attn_weights (mean/std): {attn_weights.squeeze(-1).mean(dim=0).detach().cpu().numpy()}")

        projected_stack = torch.stack(projected, dim=1)  # (B, N, fusion_dim)
        fused = (attn_weights * projected_stack).sum(dim=1)  # (B, fusion_dim)
        return fused
