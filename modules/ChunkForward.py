import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class ChunkForwardModule(nn.Module):
    """
    Feed one **chunk** (video or pre-extracted rPPG) through
        • PhysMamba (rPPG encoder) – always executed
        • TemporalBranch            – learns temporal features

    Key features
    -------------
    • The encoder can be *frozen*; in that case we:
        – wrap the forward pass with `torch.no_grad()`  
        – optionally slice the input into micro-batches  
          to cap peak GPU memory and avoid OOM.
    • Optional gradient-checkpointing on the TemporalBranch.
    """

    def __init__(
        self,
        extractor: nn.Module,
        encoder: nn.Module,
        use_checkpoint: bool = False,
        freeze_extractor: bool = True,
        micro_bs: int = 24,          # max chunks processed per encoder call
    ):
        super().__init__()
        self.extractor = extractor
        self.encoder = encoder
        self.use_checkpoint = use_checkpoint
        self.freeze_extractor = freeze_extractor
        self.micro_bs = micro_bs

        # Freeze encoder weights if requested
        if self.freeze_extractor:
            for p in self.extractor.parameters():
                p.requires_grad = False
            self.extractor.eval()

    # ------------------------------------------------------------------ #
    # Helper for gradient-checkpointing on MTDE only
    # ------------------------------------------------------------------ #
    def _encoder(self, rppg):
        return self.encoder(rppg)

    # ------------------------------------------------------------------ #
    # Extractor forward with memory-safe micro-batching
    # ------------------------------------------------------------------ #
    def _run_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            • shape (N, 1, 128)  – rPPG snippet batch
            • shape (1, C, T, H, W) – single video chunk

        Returns
        -------
        Tensor
            shape (N, T_rppg) – raw rPPG sequence per chunk
        """

        def _ex(sub):
            # Encoder is frozen – no gradients, no activation storage
            with torch.no_grad():
                return self.extractor(sub)

        # If batch is small enough, run once
        if x.size(0) <= self.micro_bs:
            return _ex(x).detach()               # detach breaks graph

        # Otherwise slice into micro-batches to cap peak memory
        outs = []
        for s in range(0, x.size(0), self.micro_bs):
            outs.append(_ex(x[s:s + self.micro_bs]))
        return torch.cat(outs, 0).detach()        # (N, T_rppg)

    # ------------------------------------------------------------------ #
    # Main forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts
        -------
        • (1, C, T, H, W)  : a single video chunk
        • (N, 1, 128)      : batch of pre-segmented rPPG snippets

        Returns
        -------
        Tensor – chunk embeddings with shape (N, D)
        """

        # 1) PhysMamba → rPPG
        if x.dim() == 5:                          # video input
            rppg = self._run_extractor(x)           # (1, T_rppg)
        else:                                     # rPPG batch
            rppg = self._run_extractor(x)           # (N, T_rppg)

        # 2) Make rPPG a leaf tensor so TemporalBranch gradients flow
        rppg = rppg.requires_grad_(True)
        #"""
        # 3) Normalize rPPG
        mean = rppg.mean(dim=1, keepdim=True)
        std = rppg.std(dim=1, keepdim=True)
        rppg_norm = (rppg - mean) / (std + 1e-6)
        rppg_norm = rppg_norm.unsqueeze(-1)
        #"""
        # 3) TemporalBranch (optionally checkpointed)
        if self.use_checkpoint:
            emb = checkpoint.checkpoint(self._temporal, rppg)
        else:
            emb = self.encoder(rppg)      # (N, D)

        return emb


'''
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
        
        """
        # === Normalize rPPG ===
        mean = rppg.mean(dim=1, keepdim=True)
        std = rppg.std(dim=1, keepdim=True)
        rppg_norm = (rppg - mean) / (std + 1e-6)
        rppg_norm = rppg_norm.unsqueeze(-1)  # shape: (1, T, 1)
        # print(f"[CHECK] rppg_norm grad_fn: {rppg_norm.grad_fn}")
        # print(f"[CHECK] rppg_norm.requires_grad: {rppg.requires_grad}")  
        """
        
        # === TemporalBranch ===
        if self.use_checkpoint:
            emb = checkpoint.checkpoint(self.forward_temporal, rppg)
        else:
            emb = self.temporal_branch(rppg)

        return emb  # shape: (1, embedding_dim)
'''