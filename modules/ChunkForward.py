import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class ChunkForwardModule(nn.Module):
    """
    A helper module to wrap:
      - PhysMamba (encoder)
      - TemporalBranch
    Supports:
      - Gradient checkpointing for temporal branch
      - Optional encoder freeze
    """
    def __init__(self, encoder, temporal_branch, use_checkpoint=False, freeze_encoder=True):
        super(ChunkForwardModule, self).__init__()
        self.encoder = encoder
        self.temporal_branch = temporal_branch
        self.use_checkpoint = use_checkpoint
        self.freeze_encoder = freeze_encoder

        # Freeze encoder if flag is set
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def forward_temporal(self, rppg_norm):
        return self.temporal_branch(rppg_norm)

    def forward(self, chunk_data):
        # chunk_data: (1, C, T, H, W)

        # === Encoder (PhysMamba) ===
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            rppg = self.encoder(chunk_data)  # shape: (1, T)
        if self.freeze_encoder:
            # === rPPG -> Leaf Tensor===
            rppg = rppg.detach().requires_grad_()
        # print(f"[CHECK] rppg.requires_grad: {rppg.requires_grad}") 
        
        '''
        # === Normalize rPPG ===
        mean = rppg.mean(dim=1, keepdim=True)
        std = rppg.std(dim=1, keepdim=True)
        rppg_norm = (rppg - mean) / (std + 1e-6)
        rppg_norm = rppg_norm.unsqueeze(-1)  # shape: (1, T, 1)
        # print(f"[CHECK] rppg_norm grad_fn: {rppg_norm.grad_fn}")
        # print(f"[CHECK] rppg_norm.requires_grad: {rppg.requires_grad}")  
        '''
        
        # === TemporalBranch ===
        if self.use_checkpoint:
            emb = checkpoint.checkpoint(self.forward_temporal, rppg)
        else:
            emb = self.temporal_branch(rppg)

        return emb  # shape: (1, embedding_dim)
