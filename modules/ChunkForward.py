import torch, torch.nn as nn, torch.utils.checkpoint as checkpoint
from scipy.signal import butter, filtfilt


class ChunkForwardModule(nn.Module):
    """
    One video/rPPG *chunk* passes through
      • PhysMamba  –> raw rPPG signal
      • MTDE       –> temporal embedding

    ✔  Optional: micro-batching for the extractor (OOM safety)  
    ✔  Optional: gradient-checkpointing on MTDE  
    ✔  Encoder can be frozen so the graph starts at MTDE
    """

    def __init__(
        self,
        extractor: nn.Module,
        encoder:   nn.Module,
        use_checkpoint:  bool = False,
        freeze_extractor: bool = True,
        micro_bs: int = 24          # max snippets per extractor call
    ):
        super().__init__()
        self.extractor = extractor
        self.encoder   = encoder
        self.use_checkpoint  = use_checkpoint
        self.freeze_extractor = freeze_extractor
        self.micro_bs = micro_bs

        if self.freeze_extractor:
            for p in self.extractor.parameters():
                p.requires_grad = False
            self.extractor.eval()              # inference mode → saves memory

    # ------------------------------------------------------------------ #
    # Helper so checkpoint can forward gate_on into MTDE
    # ------------------------------------------------------------------ #
    def _encoder(self, rppg_norm: torch.Tensor, gate_on: bool):
        return self.encoder(rppg_norm, gate_on=gate_on)   # (N, D)

    # ------------------------------------------------------------------ #
    # Memory–safe PhysMamba forward (optional micro-batching)
    # ------------------------------------------------------------------ #
    def _run_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw rPPG (N, T_rppg) detached from graph."""
        def _forward(sub):
            with torch.no_grad():
                return self.extractor(sub)      # (B, T)

        if x.size(0) <= self.micro_bs:
            return _forward(x).detach()

        chunks = [ _forward(x[s:s+self.micro_bs]) for s in range(0, x.size(0), self.micro_bs) ]
        return torch.cat(chunks, dim=0).detach()

    # ------------------------------------------------------------------ #
    # Optional physiologic pre-filter (unused in current pipeline)
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess_rppg(batch: torch.Tensor, fs: float = 30.0) -> torch.Tensor:
        """(B, T) → (B, 2, T)  band-pass + session min-max normalised."""
        b, a = butter(2, [0.7/(fs/2), 4/(fs/2)], 'band')
        filt = torch.tensor(
            filtfilt(b, a, batch.cpu().numpy(), axis=1),
            dtype=batch.dtype, device=batch.device
        )
        vmin, vmax = filt.min(1, keepdim=True).values, filt.max(1, keepdim=True).values
        norm = (filt - vmin) / (vmax - vmin + 1e-6)
        return torch.stack([norm, filt], dim=1)           # (B, 2, T)

    # ------------------------------------------------------------------ #
    # Main forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, *, gate_on: bool = True) -> torch.Tensor:
        """
        x shapes
        • (1, C, T, H, W)  – single video chunk
        • (N, 1, 128)      – batch of rPPG snippets
        Returns: (N, D)  chunk embeddings
        """

        # 1) PhysMamba → raw rPPG  (detached leaf tensor)
        rppg = self._run_extractor(x)           # (N, T_rppg)  no grad
        rppg = rppg.requires_grad_(True)        # start gradient flow

        # 2) z-score normalisation  → (N, T, 1)
        mu, sigma = rppg.mean(1, keepdim=True), rppg.std(1, keepdim=True)
        rppg_norm = ((rppg - mu) / (sigma + 1e-6)).unsqueeze(-1)

        # 3) MTDE   (checkpoint if requested)
        if self.use_checkpoint:
            emb = checkpoint.checkpoint(self._encoder, rppg_norm, gate_on)
        else:
            emb = self.encoder(rppg_norm, gate_on=gate_on)

        return emb                               # (N, D)



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
    
    def preprocess_rppg(batch, fs=30.0):
        # band‑pass
        b, a = signal.butter(2, [0.7/(fs/2), 4/(fs/2)], 'band')
        filt = torch.tensor(
            signal.filtfilt(b, a, batch.cpu().numpy(), axis=1),
            dtype=batch.dtype, device=batch.device
        )
        # 세션‑min‑max
        min_v = filt.min(dim=1, keepdim=True).values
        max_v = filt.max(dim=1, keepdim=True).values
        norm  = (filt - min_v) / (max_v - min_v + 1e-6)
        return torch.stack([norm, filt], dim=1)  # (B, 2, T)


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