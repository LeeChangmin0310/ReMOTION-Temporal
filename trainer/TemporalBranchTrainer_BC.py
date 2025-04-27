"""

rPPG → TemporalBranch → Chunk Embedding
                      ↓
               AttnScorer (raw score)
                      ↓
        ┌─────────────┴──────────────┐
        │                            │
   SupConLoss / ChunkCE         GatedPooling
        ↓                            ↓
    Representation             Session Embedding
                                 ↓
                         ClassificationHead


Video session  →  N temporal chunks  ──▶  PhysMamba(encoder, frozen)
                                         │  rPPG 1-D signal
                                         ▼
                                    TemporalBranch
                                         │  chunk-embed z_t  (D=256)
             ┌─────────────┬─────────────┴─────────────┐
             │             │                           │
      (Ph0) Projection  AttnScorer            (Ph2)   GatedPooling
             │             │                           │
     SupCon 128-dim   raw score s_t               session-embed Z_s
             ▼             ▼                           ▼
      SupConLoss       α-scheduler              Classifier(CE)
                                 (Phase-specific heads)





=======================================================
🧠 Emotion Recognition Model - End-to-End Architecture
=======================================================

          rPPG Chunk (from PhysMamba)
                        │
                ┌──────▼─────────┐
                │ TemporalBranch │ (Multi-Scale CNN → Embedding)
                └──────┬─────────┘
                       ▼
              Chunk Embedding (B, D)
                       │
      ┌────────────────┴─────────────────┐
      │                                  │
      ▼                                  ▼
AttnScorer (softmax/Top-K)      GatedPooling (CE path)
      │                                  │
Top-K + Threshold                      Session Embedding
      │                                  │
 ┌────┴───────────┐                      ▼
 │               ▼               ClassificationHead
 │          ProjectionHead               │
 │               ↓                       ▼
 │           Normalize         CrossEntropyLoss (session-level)
 │               ↓                       │
 │       SupConLossTopK                  │
 │         (weight decays)               ▼
 │                \                    Logits
 │                 \
 ▼                  ▼
TopKSoftPooling    ChunkAuxClassifier (per Top-K chunk)
        │                   │
  TopK Session Emb          └─ CE Loss (chunk-level, weak supervision)
        ▼
   KL AlignCosineLoss (pooled Top-K emb ↔ pooled Gated emb)

=======================================================
Summary:
- Dual attention paths: AttnScorer (SupCon/TopK) vs GatedPooling (CE)
- SupCon uses Top-K + projection (0–34 epoch), decays in importance
- Weak CE via ChunkAuxClassifier using Top-K chunks (20–34)
- GatedPooling guides main classifier via CE (30–49)
- KL AlignCosineLoss between Top-K embeds vs Gated embeds (epoch 25–35)
=======================================================


═══════════════════════════════════════════════════════════════════════
🔁 Emotion Recognition - Full Phase Description (by Epoch)
═══════════════════════════════════════════════════════════════════════

📌 Epoch 0–20: Exploration Phase
───────────────────────────────
rPPG Chunk → TemporalBranch → Chunk Embedding
                         ↓
                  AttnScorer (softmax, T=1.0)
                         ↓
               Attention Weights (no Top-K)
                         ↓
         Weighted Projection → Normalize → SupConLossTopK
                         ↓
                 SparsityLoss (attention entropy)

🟢 Active Modules:
- TemporalBranch, AttnScorer, ProjectionHead, SupConLossTopK
- GatedPooling (from epoch 15) → only pooling, no CE

❌ Frozen:
- AlignLoss, Classifier, ChunkAuxClassifier

🎯 Purpose:
- Encourage diverse attention during early exploration
- Learn global discriminative temporal representations
- Avoid premature Top-K filtering


📌 Epoch 20–24: Top-K Supervision Phase
────────────────────────────────
AttnScorer → raw scores only
Top-K + Threshold → selection

┌── Top-K selected embeddings
│      └→ Projection → Normalize → SupConLossTopK (lower weight)
│      └→ ChunkAuxClassifier (chunk-level CE)
│      └→ TopKSoftPooling (pooled embedding for KL later)
▼
GatedPooling → session embedding → Classifier (still frozen)

🟠 Active Modules:
- TemporalBranch, AttnScorer, ProjectionHead
- GatedPooling (from epoch 15)
- ChunkAuxClassifier (weak CE, epoch 20–34)
- SupConLossTopK (Top-K based, lower weight)
- SparsityLoss (Attn entropy)

❌ Frozen:
- Classifier (session-level CE)
- KL AlignLoss

🎯 Purpose:
- Begin weak CE supervision via Top-K chunks
- Shift representation learning from contrastive to CE
- Maintain sparse attention and chunk discriminability


📌 Epoch 25–34: Alignment Phase
───────────────────────────────
TopKSoftPooling → TopK Session Embedding
GatedPooling → Gated Session Embedding
                         ↓
         AlignCosineLoss (embedding vs embedding)
                         ↓
         Classifier (session-level CE)

🔵 Active Modules:
- Classifier (trainable)
- SupConLossTopK (weight decaying)
- GatedPooling, AlignCosineLoss
- ChunkAuxClassifier

🎯 Purpose:
- Align dual attention pathways's representations (Top-K → Gated)
- Transition to CE-dominant supervision


📌 Epoch 35–44: Fine-Tuning Phase
────────────────────────────────
SupConLoss (fixed 0.05 or off)
ProjectionHead, AttnScorer → frozen
GatedPooling → Gated Embedding
                ↓
        Classifier → CrossEntropy

🟣 Active Modules:
- TemporalBranch, GatedPooling, Classifier

❄️ Frozen:
- AttnScorer, ProjectionHead, SupConLossTopK (optionally off)
- ChunkAuxClassifier (off after epoch 34)

🎯 Purpose:
- Refine CE classification accuracy
- Final discriminative tuning


📌 Epoch 45–49: Stability Phase
───────────────────────────────
    TemporalBranch → GatedPooling → Classifier → CE Loss

⚪ Active Modules:
- TemporalBranch, GatedPooling, Classifier

❌ Off:
- SupConLossTopK, ChunkAuxClassifier, KL AlignLoss, AttnScorer, ProjectionHead

🎯 Purpose:
- Stable inference-ready attention + classifier
- Final model generalization and consolidation

═══════════════════════════════════════════════════════════════════════
"""

import os
import yaml
import wandb

import math
import entmax
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader

from tqdm import tqdm
from functools import partial
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from trainer.BaseTrainer import BaseTrainer
from neural_encoders.model.PhysMamba import PhysMamba

from decoders.TemporalBranch import TemporalBranch

from modules.AttnScorer import AttnScorer           # Returns pre-softmax raw attn score
from modules.GatedPooling import GatedPooling       # No attn inside, takes extrernal attn raw score from AttnScorer
from modules.ProjectionHead import ProjectionHead
from modules.ChunkForward import ChunkForwardModule
from modules.ClassificationHead import ClassificationHead
from modules.ChunkAuxClassifier import ChunkAuxClassifier
# from modules.TopKPooling import TopKSoftPooling
# from modules.GatedAttentionPooling import GatedAttentionPooling

from tools.utils import SupConLossTopK
from tools.utils import reconstruct_sessions, run_tsne_and_plot, straight_through_topk


class TemporalBranchTrainer_BC(BaseTrainer):
    """
    Session-Level Emotion Recognition Pipeline (TemporalBranchTrainer_BC)

    ==============================================
    🎯 Final Phase-based Training Strategy Overview
    ==============================================
    
    PHASE 1: Epoch 0–19 — "Exploration"
    - SupConLossTopK (attn-weighted all chunks)
    - AttnScorer: softmax only
    - ChunkAuxClassifier: off
    - Classifier: off
    - GatedPooling: on from epoch 15 (no CE)
    - Purpose: representation diversity, soft attention adaptation

    PHASE 2: Epoch 20–24 — "Top-K Transition"
    - SupConLossTopK: Top-K + threshold, active
    - ChunkAuxClassifier: on
    - KL AlignLoss: off
    - Purpose: sparse attention, Top-K selection, weak chunk-level CE

    PHASE 3: Epoch 25–35 — "Alignment Phase"
    - SupConLossTopK: Top-K, weight decays
    - ChunkAuxClassifier: on
    - AlignCosineLoss: active (TopK vs Gated embedding)
    - Classifier: on
    - Purpose: CE training + embedding alignment

    PHASE 4: Epoch 36–44 — "Fine-tuning"
    - CE only (Classifier, GatedPooling)
    - SupConLossTopK: fixed at 0.05 (optional anchor)
    - KL AlignLoss: T=1.0, optionally on
    - Purpose: maximize CE accuracy

    PHASE 5: Epoch 45–49 — "Stability"
    - Only CE Loss
    - All auxiliary paths off
    - Purpose: clean & generalizable inference

    Loss Composition:
    + SupConLossTopK(selected_proj, label) × contrastive_weight
    + CrossEntropyLoss(session_pred, label) × ce_weight
    + Chunk-level CE (chunk_pred, label) × chunk_ce_weight
    + SparsityLoss(attn_entropy) × sparsity_weight
    """

    def __init__(self, config, data_loader):
        super(TemporalBranchTrainer_BC, self).__init__()
        self.device = torch.device(config.DEVICE)
        self.config = config
        self.max_epoch = config.TRAIN.EPOCHS
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.warm_up = config.TRAIN.WARMUP_EPOCHS 
        self.optimizer_config = {}
        self.temperature = 1.0
        
        wandb.init(
            project="TemporalReMOTION",
            name=f"Exp_Arsl_FINALPIPE_CEsoftmax",
            # config=cfg_dict,
            dir="./wandb_logs"
        )

        # --------------------------- Encoder Initialization ---------------------------
        """Used to extract physiological signals from chunked video input (frozen during training)"""
        self.encoder = PhysMamba().to(self.device)
        self.encoder = torch.nn.DataParallel(self.encoder)
        pretrained_path = os.path.join("./pretrained_encoders", "UBFC-rPPG_PhysMamba_DiffNormalized.pth")
        self.encoder.load_state_dict(torch.load(pretrained_path, map_location=self.device))

        # Freeze encoder (by default, no fine-tuning)
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        """
        # Fine-tuning
        modules_to_finetune = [
            'Block3',           # Slow stream's low-level feature extraction block
            'Block6',           # Fast stream's high-level feature extraction block
            'ConvBlock6',       # Fast stream's last conv (selective)
            'ConvBlock3'        # stem's last conv (selective)
        ]
        for name, param in self.encoder.module.named_parameters():
            if any(key in name for key in modules_to_finetune):
                param.requires_grad = True
            else:
                param.requires_grad = False
        """

        # ------------------------------ Temporal Decoder ------------------------------
        """Multi-scale 1D CNN + SE blocks for extracting temporal features from rPPG"""
        self.temporal_branch = TemporalBranch(
            embedding_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM
        ).to(self.device)

        # ------------------------------ Attention Scorer ------------------------------
        self.attn_scorer = AttnScorer(
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM
        ).to(self.device)
        
        # ------------------------------ Projection Head -------------------------------
        """Maps chunk embeddings to a projection space for contrastive learning"""
        self.chunk_projection = ProjectionHead(
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
            proj_dim=128
        ).to(self.device)
        
        # ------------------------------ Chunk Auxiliary Classifier -------------------------
        """Auxiliary classifier for weak chunk-level CE loss.
        Trained with session-level GT label per chunk."""
        self.chunk_aux_classifier = ChunkAuxClassifier(
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
            num_classes=config.TRAIN.NUM_CLASSES
        ).to(self.device)
        
        # ------------------------------ Gated Attention Pooling ------------------------------
        """Aggregates chunk embeddings to a session-level embedding using Gated self-attention
           Includes entropy regularization (used in sparsity loss)"""
        self.pooling = GatedPooling(
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM
        ).to(self.device)

        # ----------------------------- Classification Head -----------------------------
        """Simple MLP head for predicting emotion labels from pooled session embedding"""
        self.classifier = ClassificationHead(
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
            num_classes=config.TRAIN.NUM_CLASSES
        ).to(self.device)

        # ----------------------------- Chunk Forward Module ----------------------------
        self.chunk_forward_module = ChunkForwardModule(
            encoder=self.encoder.module,
            temporal_branch=self.temporal_branch,
            use_checkpoint=False,
            freeze_encoder=True
        ).to(self.device)

        # -------------------------------- Loss Functions -------------------------------
        """SupConLossTopK: supervised contrastive loss using top-K attended chunks per session"""
        self.contrastive_loss_fn = SupConLossTopK(temperature=0.1)
        
        """CrossEntropyLoss: standard cross-entropy loss for chunk-level classification(phase1)"""
        self.aux_criterion = nn.CrossEntropyLoss(reduction='none')
        
        """CrossEntropyLoss: standard cross-entropy loss for session-level classification(phase2)"""
        self.criterion = nn.CrossEntropyLoss() # <================================ HOW ABOUT FOCAL LOSS???
        
        """
        AlignCosineLoss: cosine distance between Top-K pooled vs Gated pooled embeddings.
        Encourages representational alignment across attention streams.
        self.cosinse_criterion = AlignCosineLoss()
        """

        # ------------------------------ Optimizer & Scheduler -----------------------------
        self.optimizer = None
        self.scheduler = None
        
        # ---------------------------- Session Reconstruction ----------------------------
        self.reconstruct_sessions = partial(reconstruct_sessions, self)
        
        # ------------------------------ t-SNE visualization ------------------------------
        self.run_tsne_and_plot = partial(run_tsne_and_plot, self)
        
        # -------------------------------- History Tracking --------------------------------
        self.best_ce_loss = float('inf')
        self.best_cos_loss = float('inf')
        self.best_chunk_ce_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_val_acc_phase = [0.0 for _ in range(3)]     # Phase 0,1,2
        self.best_val_f1_phase = [0.0 for _ in range(3)]
        self.best_val_metric_phase = [0.0 for _ in range(3)]
        self.avg_entropy_attn = None
        
        self.train_losses = []
        self.val_losses = []
        self.loss_ce_per_epoch = []
        self.loss_contrastive_per_epoch = []
        self.loss_sparsity_per_epoch = []
        self.loss_chunk_ce_per_epoch = []

        # ---------------------------- t-SNE Embedding Containers ----------------------------
        self.chunk_embeddings_for_tsne = []
        self.chunk_labels_for_tsne = []
        self.session_embeddings_for_tsne = []
        self.session_labels_for_tsne = []
    
    def forward_chunk(self, chunk):
        """
        Forward pass for a single chunk.
        Args:
            chunk (Tensor): Input chunk tensor of shape (1, C, T, H, W)
        Returns:
            emb (Tensor): Output embedding tensor of shape (1, embed_dim)
        """
        return self.chunk_forward_module(chunk)
    
    def forward_single_chunk_checkpoint(self, chunk):
        """
        Forward pass a single chunk using the internal setting of use_checkpoint.
        """
        if self.chunk_forward_module.use_checkpoint:
            return checkpoint.checkpoint(self.forward_chunk, chunk)
        else:
            return self.forward_chunk(chunk)
        
    def update_training_state(self, epoch):
        """
        Unified phase-aware training control:
        - Sets loss weights, temperature, top-k ratio
        - Controls gradient flow
        - Configures optimizer/scheduler
        
        Phase 0 (0–19): SupCon + Entropy Temperature Control
        Phase 1 (20–34): Chunk CE + Top-K
        Phase 2 (35–49): Final CE + Classifier + GatedPooling
        """
        # ───────────────────── Phase-wise setup ─────────────────────
        PHASE0_END = 19
        PHASE1_END = 34
        PHASE2_END = self.max_epoch - 1
        
        # ───────────────────── Phase-0 ─────────────────────
        if epoch <= PHASE0_END:          # Phase-0   (0–19)
            phase, lr, wd, t_max = 0, 3e-4, 1e-4, 20 # (= phase length × 1.0)
            
            # ① SupCon λ : 0-4 (1→0.6) , 5-19 (0.6→0.3)
            if  epoch < 10:
                self.contrastive_weight = 1.0 - 0.05 * epoch        # 1.0→0.55
            else:
                self.contrastive_weight = 0.55 - 0.02 * (epoch-10)  # 0.55→0.35
            self.contrastive_weight = max(0.3, self.contrastive_weight)
            self.lambda_ent      = 0.1
            self.chunk_ce_weight = 0.0
            self.ce_weight       = 0.0
            
            # adaptive τ  (0.5 ≤ τ ≤ 1.2)
            if self.avg_entropy_attn is None:
                self.temperature = 1.0
            else:
                tau = 2.5 - self.avg_entropy_attn
                self.temperature = max(0.7, min(1.2, tau))
        
        # ───────────────────── Phase-1 ─────────────────────
        elif epoch <= PHASE1_END:        
            phase, lr, wd, t_max = 1, 2e-4, 5e-5, 20 # (= 15 × 1.3 ≈ 20)
            
            # ② Chunk-CE λ : 20-24 = 0.3, 25-34 = 0.5→0.7
            if epoch <= 24:
                self.chunk_ce_weight = 0.35
            else:
                self.chunk_ce_weight = 0.35 + 0.025 * (epoch-25)        # 0.35→0.60
            self.contrastive_weight = 0.0                               # Weak SupCon
            self.lambda_ent         = 0.0
            self.chunk_ce_weight    = min(0.60, self.chunk_ce_weight)
            self.ce_weight          = 0.0
            self.temperature        = 1.0                               # fixed
        
        # ───────────────────── Phase-2 ─────────────────────
        else:                          
            phase, lr, wd, t_max = 2, 1e-4, 5e-5, 15 # (= 15 × 1.0)
            self.lambda_ent         = 0.0
            self.contrastive_weight = 0.0
            self.chunk_ce_weight    = 0.0
            self.ce_weight          = 1.0
            self.temperature        = 1.0
            
        # ─────── Gradient gate ───────
        for p in self.attn_scorer.parameters():
            p.requires_grad = True
        for p in self.temporal_branch.parameters():
            p.requires_grad = True
        for p in self.chunk_projection.parameters():
            p.requires_grad = (phase < 1)
        for p in self.chunk_aux_classifier.parameters():
            p.requires_grad = (phase == 1)
        for p in self.pooling.parameters():
            p.requires_grad = (phase == 2)
        for p in self.classifier.parameters():
            p.requires_grad = (phase == 2)

        # Top-K Ratio
        self.top_k_ratio = max(0.3, 0.6 - 0.3 * (epoch / PHASE2_END))

        # Reconfigure optimizer/scheduler per phase
        self.configure_optimizer_scheduler(lr, wd, t_max, phase)
        
        # SupConLoss sampling scheduling (e.g., threshold, top-k mask, etc.)
        self.contrastive_loss_fn.schedule_params(epoch, self.max_epoch)
        
        return phase

    def configure_optimizer_scheduler(self, lr, weight_decay, t_max, phase):
        """
        Phase-wise optimizer & scheduler setup.
        Automatically resets learning rate and weight decay per phase.

        * phase 0 : AttnScorer LR = 1.00 × lr
        * phase 1 : AttnScorer LR = 1.00 × lr   
        * phase 2 : AttnScorer LR = 0.50 × lr   (fine-tuning)
        """
        # ------------------------------------------------------------------
        lr_scale_as = 1.0 if phase < 2 else 0.5
        lr_raw      = lr                     # default lr per phase
        lr_as       = lr * lr_scale_as       # lr for AttnScorer
        # ------------------------------------------------------------------

        pg_decay, pg_nodecay = [], []

        def add_param(param, name, lr_this):
            is_nd = ("bias" in name or "LayerNorm" in name)
            group  = pg_nodecay if is_nd else pg_decay
            group.append(
                {"params": param,
                "lr": lr_this,
                "weight_decay": 0.0 if is_nd else weight_decay}
            )

        # --------- params setup ---------
        modules = [
            (self.temporal_branch,     lr_raw),
            (self.attn_scorer,         lr_as),
            (self.pooling,             lr_raw),
            (self.chunk_projection,    lr_raw),
            (self.chunk_aux_classifier,lr_raw),
            (self.classifier,          lr_raw),
        ]

        for module, lr_mod in modules:
            for n, p in module.named_parameters():
                if p.requires_grad:
                    add_param(p, n, lr_mod)

        # --------- Optimizer / Scheduler ---------
        self.optimizer = optim.AdamW(pg_decay + pg_nodecay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max, eta_min=1e-6
        )
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    
    # ----------------------------------------------------------
    # helper (already imported in your module header once)
    # from utils.attention_utils import straight_through_topk
    # ----------------------------------------------------------

    def forward_batch(self, batch, epoch, phase):
        """
        One mini-batch forward pass (B sessions).
        Handles phase-wise:
        - Phase 0: SupConLossTopK + adaptive temperature
        - Phase 1: Chunk-level CE with Top-K + threshold
        - Phase 2: Session-level CE with GatedPooling
        
        Handles:
        - TemporalBranch → chunk-wise rPPG embeddings
        - AttnScorer: pre-softmax raw score
        - SupConLossTopK with Top-K + threshold selection from epoch ≥ 20
        - GatedPooling for CE loss
        - ChunkAuxClassifier (epoch 20–34 only) for chunk-level CE supervision
        - SparsityLoss based on entropy: AttnScorer vs Gated
        - t-SNE logging for chunk/session visualization
        - Full debug printouts for embeddings, attention, entropy, losses
        
        Additional:
        - t-SNE logging
        - Attention entropy debug
        - Full loss composition & debug prints
        """
        chunk_tensors, batch_labels, batch_sids, _ = batch
        B = len(batch_labels)

        # ───── collectors ────────────────────────────────────
        session_embeds, target_labels = [], []
        proj_for_supcon, labels_for_supcon = [], []
        chunk_ce_losses, entropy_list = [], []
        entropy_attn_list, entropy_gated_list = [], []
        # -----------------------------------------------------

        # ───── per-session loop ──────────────────────────────
        for b in range(B):
            chunks = chunk_tensors[b].float().to(self.device)      # (T,1,128)
            T      = chunks.size(0)
            label  = batch_labels[b]
            sid    = batch_sids[b]

            print(f"[DEBUG] Session {sid}  #chunks={T}")

            # 1) encode all chunks at once  (T,1,128) ➜ (1,T,D)
            chunk_emb = self.forward_single_chunk_checkpoint(chunks)  # (T,D)
            chunk_emb = chunk_emb.unsqueeze(0)                        # (1,T,D)
            
            # ──────── per-chunk debug & t-SNE ───────────────
            for idx, cemb in enumerate(chunk_emb[0]):                # iterate T chunks
                print(f"  Chunk {idx}: mean={cemb.mean():.4f}, std={cemb.std():.4f}")
                self.chunk_embeddings_for_tsne.append(cemb.detach().cpu().numpy())
                self.chunk_labels_for_tsne.append(label)
            # ────────────────────────────────────────────────

            # 2) attention scorer 
            attn_soft, raw_scaled = self.attn_scorer(
                chunk_emb,                               # (1,T,1)
                temperature=self.temperature,            # τ  (phase-adaptive)
                epoch=epoch,                             # epoch index
            )                    
            print(f"[DEBUG] Session {sid}  #chunks={T}, σ_running={self.attn_scorer.running_score_std.item():.3f}")
            
            # 3) build α  – soft in Phase-0, ST-Top-K in Phase-1
            if phase == 0:      # Phase-0 softmax
                alpha = attn_soft 
                entropy_attn_soft = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=1).mean()
            elif phase == 1:    # Phase-1 ST-Top-K
                if len(raw_scaled.squeeze(0).squeeze(-1)) < 1:
                    print("Empty attention scores – skip this session")
                    continue
                k  = min(T, max(6, int(T * self.top_k_ratio)))
                alpha_mask = straight_through_topk(raw_scaled, k=k) # (1, T)
                mask_bool  = alpha_mask.squeeze(0).bool()           # (T,)
                alpha = alpha_mask.unsqueeze(-1)                    # (1,T,1)
                print(f"[DEBUG] Top-K selected = {mask_bool.sum().item()} / {T}")
                entropy_attn_soft = -(attn_soft * attn_soft.clamp_min(1e-8).log()).sum(dim=1).mean()
            else:               # Phase-2 raw scores
                alpha = torch.ones_like(raw_scaled)   # (1,T,1)
                entropy_attn_soft = 0

            # 4) apply α before projection  (keeps grad path)
            gated_emb   = chunk_emb * alpha                           # (1,T,D)
            proj_embeds = self.chunk_projection(gated_emb)            # (1,T,128)

            # 5-A) SupCon collection
            if phase < 1:
                proj_for_supcon.append(proj_embeds.squeeze(0))
                labels_for_supcon.append(torch.full((T,), label,
                                    dtype=torch.long, device=self.device))

            # 5-B) Top-K chunk CE  (Phase-1)
            if phase == 1:
                logits_all = self.chunk_aux_classifier(chunk_emb.squeeze(0))  # (T,C)
                lbl_all    = torch.full((T,), label, device=self.device)      # (T)
                ce_per_chunk     = self.aux_criterion(logits_all, lbl_all)              # scalar
                weighted_ce  = (alpha_mask.squeeze(0) * ce_per_chunk).sum() / (
                                alpha_mask.squeeze(0).sum() + 1e-8)  # ← eps
                chunk_ce_losses.append(weighted_ce)

            # 6) session-level pooling
            pooled, attn_gated, ent_gated = self.pooling(
                gated_emb, raw_scaled, return_weights=True, return_entropy=True)
            # ★──────── session-level t-SNE ───────────────────
            sess_emb = pooled.squeeze(0)                              # (D,)
            self.session_embeddings_for_tsne.append(sess_emb.detach().cpu().numpy())
            self.session_labels_for_tsne.append(label)
            # ────────────────────────────────────────────────

            session_embeds.append(pooled.squeeze(0))
            target_labels.append(label)

            # 7) entropy bookkeeping
            entropy_attn_list.append(entropy_attn_soft)
            entropy_gated_list.append(ent_gated)
            entropy_list.append(entropy_attn_soft if phase < 2 else ent_gated)

            # 8) OPTIONAL chunk-level debug
            if phase < 2:
                attn_np = attn_soft.detach().cpu().squeeze().numpy()
                print(f"[DEBUG][Attn Weights] {attn_np.tolist()}")
                print(f"[DEBUG][Attn Sparsity] mean={attn_np.mean():.4f}, std={attn_np.std():.4f}, entropy={entropy_attn_soft.item():.4f}")
            else:
                gated_np = attn_gated.detach().cpu().squeeze().numpy()
                print(f"[DEBUG][Gated Attn Weights] {gated_np.tolist()}")
                print(f"[DEBUG][Gated Attn Sparsity] mean={gated_np.mean():.4f}, std={gated_np.std():.4f}, entropy={ent_gated.item():.4f}")

        # ───── phase-specific losses ─────────────────────────
        # SupCon (Phase-0)
        if phase < 1 and proj_for_supcon:
            proj_concat  = torch.cat(proj_for_supcon, 0)
            label_concat = torch.cat(labels_for_supcon, 0)
            if label_concat.unique().numel() >= 2:
                contrastive_term = self.contrastive_loss_fn(proj_concat, label_concat)
            else:
                contrastive_term = None
        else:
            contrastive_term = None

        # Session CE (always computed)
        sess_mat   = torch.stack(session_embeds, 0)
        label_vec  = torch.tensor(target_labels, dtype=torch.long, device=self.device)
        ce_logits  = self.classifier(sess_mat)
        ce_loss    = self.criterion(ce_logits, label_vec)

        # Chunk CE (Phase-1)
        chunk_ce_raw = (torch.stack(chunk_ce_losses).mean()
                        if chunk_ce_losses else torch.tensor(0.0, device=self.device))

        # Sparsity (for monitoring)
        sparsity_loss = torch.stack(entropy_list).mean()

        # ───── scaled terms (for debug table) ───────────────
        ce_loss_scaled          = self.ce_weight      * ce_loss
        contrastive_term_scaled = (self.contrastive_weight * contrastive_term
                                if contrastive_term is not None
                                else torch.tensor(0.0, device=self.device))
        chunk_ce_loss_scaled    = self.chunk_ce_weight * chunk_ce_raw

        # ───── total loss per phase ─────────────────────────
        if phase == 0:
            if contrastive_term is None:
                print("⚠️  Skip batch (no positive pairs)")
                return None
            loss_total = contrastive_term_scaled + self.lambda_ent * sparsity_loss
        elif phase == 1:
            loss_total = chunk_ce_loss_scaled
        else:                              # phase 2
            loss_total = ce_loss_scaled
        # ----------------------------------------------------

        # ───── FULL DEBUG PRINT (unchanged layout) ──────────
        print("\n📊 [Loss Breakdown & Prediction]")
        print("─" * 60)
        print(f"🔧 Loss Weights:\n   ▸ CE         : {self.ce_weight:.2f}"
            f"\n   ▸ Contrastive: {self.contrastive_weight:.2f}"
            f"\n   ▸ ChunkCE    : {self.chunk_ce_weight:.2f}")
        print(f"\n📉 Loss Components (PHASE {phase}):")
        print(f"   ▸ CE         : {ce_loss.item():.4f} × {self.ce_weight:.2f}"
            f" = {ce_loss_scaled.item():.4f}")
        print(f"   ▸ Contrastive: "
            f"{(contrastive_term or torch.tensor(0)).item():.4f} × "
            f"{self.contrastive_weight:.2f} = "
            f"{contrastive_term_scaled.item():.4f}")
        print(f"   ▸ Chunk CE   : {chunk_ce_raw.item():.4f} × "
            f"{self.chunk_ce_weight:.2f} = "
            f"{chunk_ce_loss_scaled.item():.4f}")
        entropy_gated_mean = torch.stack(entropy_gated_list).mean()
        if phase < 2:
            entropy_attn_mean = torch.stack(entropy_attn_list).mean()
            print(f"   ▸ Entropy    : Scorer = {entropy_attn_mean:.4f}, "
                f"Gated = {entropy_gated_mean:.4f}")
        else:
            entropy_attn_mean = torch.tensor(0.0, device=self.device)
            print(f"   ▸ Entropy    : Scorer = None(Phase 2), "
                f"Gated = {entropy_gated_mean:.4f}")
        print(f"   ▸ Sparsity   : {sparsity_loss.item():.4f}")
        print(f"   ▶ Total Loss : {loss_total.item():.4f}")
        print("─" * 60)

        # ───── prediction list (unchanged) ─────────────────
        probs = torch.softmax(ce_logits, 1)
        for idx in range(len(label_vec)):
            prob_np = probs[idx].detach().cpu().numpy()
            pred    = prob_np.argmax()
            print(f"   ▸ Session {batch_sids[idx]} | GT: {label_vec[idx].item()} "
                f"| Pred: {pred} | Conf: {prob_np[pred]:.4f} "
                f"| Prob: {np.round(prob_np,4)}")
        print("═" * 60)

        # ───── return (original order) ─────────────────────
        return (
            loss_total,
            ce_loss,
            contrastive_term_scaled,
            sparsity_loss,
            chunk_ce_loss_scaled,
            torch.argmax(ce_logits, 1),
            label_vec,
            chunk_ce_raw.item(),
            entropy_attn_mean,
            entropy_gated_mean
        )


    def train(self, data_loader):
        """
        Training loop with full phase-aware scheduling:
        - Adaptive loss weights
        - Adaptive temperature & threshold based on entropy
        - Phase-dependent gradient flow and Top-K usage
        - Debug info for loss weights, gradients, and prediction breakdown
        
        Main training loop for session-level emotion classification using:
        🔁 Pipeline:
        - TemporalBranch extracts chunk-wise rPPG embeddings
        - AttnScorer assigns attention scores (softmax → entmax → raw score)
        - Top-K + threshold selection for contrastive learning (SupConLossTopK)
        - GatedPooling aggregates session embeddings (used for CE)
        - Classifier makes session-level predictions

        🎯 Multi-phase Loss Scheduling:
        - SupConLossTopK: early exploration → Top-K-based → frozen
        - CE Loss: ramp-up during phase 2 → dominant in later epochs
        - SparsityLoss: attention entropy regularization
        - ChunkAuxClassifier: weak CE on Top-K chunks (epoch 20–34)
        """
        torch.autograd.set_detect_anomaly(True)
        
        # Initialize per-epoch loss trackers
        self.loss_ce_per_epoch = []
        self.loss_contrastive_per_epoch = []
        self.loss_sparsity_per_epoch = []
        self.loss_chunk_ce_per_epoch = []

        for epoch in range(self.max_epoch):
            self.train_mode()

            # === Placeholder for entropy tracking ===
            entropy_attn_all = []
            entropy_gated_all = []

            # === Temporary call for logging current scheduling phase only ===
            phase = self.update_training_state(epoch)

            # === Epoch Settings Print ===
            print("\n" + "=" * 80)
            print(f"🧠 [Epoch {epoch}/{self.max_epoch}] - Training Phase {phase}")
            print("-" * 80)
            print("🔧 Loss Scheduling:")
            print(f"   ▸ CE         : {'Inactive' if self.ce_weight == 0.0 else f'Phase 2 Ramping Up({self.ce_weight:.2f})' if self.ce_weight < 1.0 else 'Fully Active'}")
            print(f"   ▸ Contrastive: {'Off' if self.contrastive_weight == 0.0 else f'Decaying ({self.contrastive_weight:.2f})' if self.contrastive_weight < 1.0 else 'Full'}")
            print(f"   ▸ Chunk CE   : {'On ({:.2f})'.format(self.chunk_ce_weight) if self.chunk_ce_weight > 0 else '[OFF]'}")
            if phase < 1:
                print("🎯 SupCon Sampling Strategy:")
                print(f"   ▸ Max Pos            : {self.contrastive_loss_fn.max_pos}")
                print(f"   ▸ Max Neg            : {self.contrastive_loss_fn.max_neg}")
                print(f"   ▸ Pos/Neg Ratio      : {self.contrastive_loss_fn.pos_neg_ratio:.2f}")
                print(f"   ▸ SupConTemperature  : {self.contrastive_loss_fn.temperature:.2f}")
                print(f"   ▸ SoftmaxTemperature : {self.temperature:.2f}")
            print("-" * 80)

            tbar = tqdm(data_loader["train"], desc=f"Epoch {epoch}", ncols=80)
            running_loss = 0.0
            all_preds, all_labels = [], []
            
            ce_total = 0.0
            contrastive_total = 0.0
            sparsity_total = 0.0
            chunk_ce_total = 0.0
            num_batches = len(tbar)
            
            # Clear t-SNE logging buffers (visualizing chunk/session embeddings per epoch)
            self.chunk_embeddings_for_tsne.clear()
            self.chunk_labels_for_tsne.clear()
            self.session_embeddings_for_tsne.clear()
            self.session_labels_for_tsne.clear()

            for idx, batch in enumerate(tbar):
                out = self.forward_batch(batch, epoch, phase)
                if out is None:
                    print(f"[SKIP] Batch {idx} skipped due to lack of contrastive labels")
                    continue
                (
                    loss_total, 
                    ce_loss, contrastive_loss, sparsity_loss, chunk_ce_loss, 
                    preds, label_tensor, train_chunk_ce, entropy_attn, entropy_gated  
                ) = out

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss_total.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_tensor.cpu().numpy())

                entropy_attn_all.append(entropy_attn.item())
                entropy_gated_all.append(entropy_gated.item())
                
                ce_total += ce_loss.item()
                contrastive_total += contrastive_loss.item()
                sparsity_total += sparsity_loss.item()
                chunk_ce_total += train_chunk_ce

                # === GRADIENT CHECK ===
                if idx % 20 == 0:
                    modules = [
                        # ("Encoder", self.encoder),
                        ("Temporal", self.temporal_branch),
                        ("AttentionScorer", self.attn_scorer),
                        ("Projection", self.chunk_projection),
                        ("ChunkAuxClassifier", self.chunk_aux_classifier),
                        ("GatedPooling", self.pooling),
                        ("Classifier", self.classifier),
                    ]
                    for name, module in modules:
                        for pname, param in module.named_parameters():
                            req_grad = param.requires_grad
                            grad_norm = param.grad.norm().item() if param.grad is not None else None
                            if grad_norm is not None:
                                print(f"[GradCheck][{name}] {pname} | requires_grad={req_grad} | grad norm: {grad_norm:.6f}")
                            else:
                                print(f"[GradCheck][{name}] {pname} | requires_grad={req_grad} | has no grad")

            # === Entropy-aware loss scheduling (actual update) ===
            avg_entropy_attn = sum(entropy_attn_all) / len(entropy_attn_all)
            avg_entropy_gated = sum(entropy_gated_all) / len(entropy_gated_all)
            self.avg_entropy_attn = avg_entropy_attn
            
            # === Training summary ===
            avg_loss = running_loss / num_batches
            acc = accuracy_score(all_labels, all_preds)
            self.train_losses.append(avg_loss)
            self.loss_ce_per_epoch.append(ce_total / num_batches)
            self.loss_contrastive_per_epoch.append(contrastive_total / num_batches)
            self.loss_sparsity_per_epoch.append(sparsity_total / num_batches)
            self.loss_chunk_ce_per_epoch.append(chunk_ce_total / num_batches)

            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Acc = {acc:.4f}, Entropy = {avg_entropy_attn:.4f}")
            
            self.run_tsne_and_plot(level="chunk", epoch=epoch, phase="train")
            self.run_tsne_and_plot(level="session", epoch=epoch, phase="train")

            # === Best Model Check by Multiple Metrics ===
            val_loss, metrics = self.valid(data_loader, epoch=epoch)
            self.val_losses.append(val_loss)

            # Save by highest accuracy, F1, sum of metrics (ACC + F1) per phase
            phase_tag = f"phase{phase}"
            
            # Save by highest accuracy per phase
            if metrics['accuracy'] > self.best_val_acc_phase[phase]:
                self.best_val_acc_phase[phase] = metrics['accuracy']
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag=f"best_acc_{phase_tag}")
                print(f"[SAVE] Best ACC model updated at epoch {epoch} ({phase_tag})")

            # Save by highest F1 per phase
            if metrics['f1'] > self.best_val_f1_phase[phase]:
                self.best_val_f1_phase[phase] = metrics['f1']
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag=f"best_f1_{phase_tag}")
                print(f"[SAVE] Best F1 model updated at epoch {epoch} ({phase_tag})")

            # Save by best sum of metrics (Acc + F1) per phase
            if metrics['f1'] + metrics['accuracy'] > self.best_val_metric_phase[phase]:
                self.best_val_metric_phase[phase] = metrics['f1'] + metrics['accuracy']
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag=f"best_metric_{phase_tag}")
                print(f"[SAVE] Best Metric model updated at epoch {epoch} ({phase_tag})")
                            
            # Save by lowest validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag="best_loss")
                print(f"[SAVE] Best Validation loss model updated at epoch {epoch}")
                
            if ce_total / num_batches < self.best_ce_loss:
                self.best_ce_loss = ce_total / num_batches
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag="best_ce_loss")
                print(f"[SAVE] Best Train Loss model updated at epoch {epoch}")
            if (chunk_ce_total / num_batches) > 0 and (chunk_ce_total / num_batches) < self.best_chunk_ce_loss:
                self.best_chunk_ce_loss = chunk_ce_total / num_batches
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag="best_chunkce")
                print(f"[SAVE] Best Chunk CE loss model updated at epoch {epoch}")

            # Save last epoch
            if epoch + 1 == self.max_epoch:
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag="last_epoch")
                print(f"[SAVE] Final model saved at last epoch {epoch}")
            
            wandb.log({
                "loss/train_total": avg_loss,
                "loss/train_ce": self.loss_ce_per_epoch[-1],
                "loss/train_contrastive": self.loss_contrastive_per_epoch[-1],
                "loss/train_sparsity": self.loss_sparsity_per_epoch[-1],
                "loss/train_chunk_ce": self.loss_chunk_ce_per_epoch[-1],
                "acc/train": acc,
                "loss/valid_total": val_loss,
                "acc/valid": metrics["accuracy"],
                "f1/valid": metrics["f1"],
                "entropy/attn": avg_entropy_attn,
                "entropy/gated": avg_entropy_gated,
                "lr": self.scheduler.get_last_lr()[0],
                "epoch": epoch
            })

        # Final visualization
        self.plot_losses_and_lrs()

    
    
    def valid(self, data_loader, epoch=0):
        """Evaluation on session-level embeddings (no chunk-level contrastive loss applied)"""

        self.eval_mode()
        all_preds, all_labels, losses, entropies = [], [], [], []

        print("\n==== Validating ====")
        vbar = tqdm(data_loader["valid"], desc="Valid", ncols=80)

        with torch.no_grad():
            for idx, batch in enumerate(vbar):
                session_emb_dict, session_label_dict, session_entropies = self.reconstruct_sessions(
                    batch=batch, idx=idx, epoch=epoch, phase='valid'
                )

                for sid, emb in session_emb_dict.items():
                    lbl = session_label_dict[sid]
                    lbl_tensor = torch.tensor([lbl], dtype=torch.long, device=self.device)

                    outputs = self.classifier(emb)
                    loss = self.criterion(outputs, lbl_tensor)
                    losses.append(loss.item())

                    # Retrieve entropy from reconstruct_sessions
                    entropy_val = session_entropies[sid].item()
                    entropies.append(entropy_val)

                    prob = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
                    pred = np.argmax(prob)

                    print(f"[DEBUG][Valid] Session {sid} | GT: {lbl} | Pred: {pred} | Prob: {np.round(prob, 4)} | Entropy: {entropy_val:.4f}")

                    all_preds.append(pred)
                    all_labels.append(lbl)

                vbar.set_postfix(
                    loss=f"{np.mean(losses):.3f}",
                    entropy=f"{np.mean(entropies):.3f}"
                )

        # Visualize t-SNE for chunk/session level
        self.run_tsne_and_plot(level="chunk", epoch=epoch, phase="valid")
        self.run_tsne_and_plot(level="session", epoch=epoch, phase="valid")

        avg_loss = np.mean(losses)
        print(f"[VALID] Avg Loss    : {avg_loss:.4f}")
        print(f"[VALID] Avg Entropy : {np.mean(entropies):.4f}")

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        conf_mat = confusion_matrix(all_labels, all_preds)

        print(f"[VALID] Accuracy    : {acc:.4f}")
        print(f"[VALID] F1 Score    : {f1:.4f}")
        print(f"[VALID] Confusion Matrix:\n{conf_mat}")

        return avg_loss, {"accuracy": acc, "f1": f1, "confusion_matrix": conf_mat}

    
    def test(self, data_loader):
        """
        Load and evaluate all saved best models (acc, f1, metric, loss based).
        """
        self.eval_mode()

        base_name = self.config.TRAIN.MODEL_FILE_NAME
        results = {}
        
        # Best ACC/F1/Metric models per phase 0,1,2
        phase_tags = []
        for phase in range(3):
            phase_tags += [
                f"best_acc_phase{phase}",
                f"best_f1_phase{phase}",
                f"best_metric_phase{phase}"
            ]
    
        # common models (best valid/ce/chunk_ce loss, total best, last epoch)
        common_tags = [
            "best_loss", "best_ce_loss", "best_chunkce", "last_epoch"
        ]
    
        tags = phase_tags + common_tags

        for tag in tags:
            model_path = os.path.join("./saved_models", f"{base_name}_{tag}.pth")
            if not os.path.exists(model_path):
                print(f"[SKIP] {tag} model not found at {model_path}")
                continue

            print(f"\n==== Testing [{tag}] Model ====")
            self.load_best_model(model_path)
            self.eval_mode()

            all_preds, all_labels = [], []
            losses, entropies = [], []

            with torch.no_grad():
                tbar = tqdm(data_loader["test"], desc=f"Test-{tag}", ncols=80)
                for idx, batch in enumerate(tbar):
                    session_emb_dict, session_label_dict, session_entropies = self.reconstruct_sessions(
                        batch=batch, idx=idx, epoch=self.max_epoch, phase='test'
                    )

                    for sid, emb in session_emb_dict.items():
                        lbl = session_label_dict[sid]
                        lbl_tensor = torch.tensor([lbl], dtype=torch.long, device=self.device)
                        outputs = self.classifier(emb)

                        # Compute CE loss
                        loss = self.criterion(outputs, lbl_tensor)
                        losses.append(loss.item())

                        # Entropy from session (from attention)
                        entropy_val = session_entropies.get(sid, torch.tensor(0.0)).item()
                        entropies.append(entropy_val)

                        # Prediction and logging
                        prob = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
                        pred = np.argmax(prob)

                        print(f"[DEBUG] Session {sid} | GT: {lbl} | Pred: {pred} | Prob: {np.round(prob, 4)} | Entropy: {entropy_val:.4f}")

                        all_preds.append(pred)
                        all_labels.append(lbl)
                    tbar.set_postfix(
                        loss=f"{np.mean(losses):.3f}",
                        entropy=f"{np.mean(entropies):.3f}"
                    )
            acc = accuracy_score(all_labels, all_preds)
            f1_ = f1_score(all_labels, all_preds)
            conf_mat = confusion_matrix(all_labels, all_preds)
            
            # wandb logging
            wandb.log({
                f"test/acc_{tag}": acc,
                f"test/f1_{tag}": f1_,
                f"test/loss_{tag}": np.mean(losses),
                f"test/entropy_{tag}": np.mean(entropies),
            })
            
            results[tag] = {
                "accuracy": acc,
                "f1": f1_,
                "confusion_matrix": conf_mat.tolist()
            }
            
            print(f"[{tag}] Acc: {acc:.4f}, F1: {f1_:.4f}")
            print(f"[{tag}] Confusion Matrix:\n{conf_mat}")
            print("-" * 50)

        return results


    def save_model(self, epoch, train_loss=None, val_loss=None, val_metrics=None, val_loss_only=False, tag="best"):
        """
        Save the best model and its associated config with evaluation results.

        Args:
            epoch (int): Current epoch number when the model was saved.
            train_loss (float, optional): Training loss at this epoch.
            val_loss (float, optional): Validation loss at this epoch.
            val_metrics (dict, optional): Validation metrics such as accuracy, F1, etc.
            tag (str): Tag for saved model, e.g., best_acc, best_f1, etc.
        """
        model_save_dir = "./saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        base_name = self.config.TRAIN.MODEL_FILE_NAME

        # === Save all modules ===
        model_dict = {
            "temporal_branch": self.temporal_branch.state_dict(),
            "attn_scorer": self.attn_scorer.state_dict(),
            "projection_head": self.chunk_projection.state_dict(),
            "chunk_aux_classifier": self.chunk_aux_classifier.state_dict(),
            "gated_pooling": self.pooling.state_dict(),
            "classifier": self.classifier.state_dict(),
        }
        model_path = os.path.join(model_save_dir, f"{base_name}_{tag}.pth")
        torch.save(model_dict, model_path)
        print(f"[SAVE] Model saved at: {model_path}")

        # === Save config + training info ===
        config_path = os.path.join(model_save_dir, f"{base_name}_{tag}_config.yaml")
        config_dict = self.config.clone().dump()
        config_dict = yaml.safe_load(config_dict)

        config_dict["RESULTS"] = {
            "tag": tag,
            "best_epoch": epoch,
            "train_loss": float(train_loss) if train_loss is not None else None,
            "valid_loss": float(val_loss) if val_loss is not None else None,
        }

        if val_metrics is not None:
            config_dict["RESULTS"].update({
                "accuracy": float(val_metrics.get("accuracy", 0.0)),
                "f1_score": float(val_metrics.get("f1", 0.0)),
                "confusion_matrix": val_metrics.get("confusion_matrix", []),
            })

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        print(f"[SAVE] Config + training info saved at: {config_path}")

    def load_best_model(self, path):
        """
        Load the best model weights from a saved checkpoint.
        This includes TemporalBranch, AttentionScorer, ChunkProjection, 
        ChunkAuxClassifier, GatedPooling, and Classifier.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.temporal_branch.load_state_dict(checkpoint["temporal_branch"])
        self.attn_scorer.load_state_dict(checkpoint["attn_scorer"])
        self.chunk_projection.load_state_dict(checkpoint["projection_head"])
        self.chunk_aux_classifier.load_state_dict(checkpoint["chunk_aux_classifier"])
        self.pooling.load_state_dict(checkpoint["gated_pooling"])
        self.classifier.load_state_dict(checkpoint["classifier"])
        
        print(f"[LOAD] Best model loaded from: {path}")

    def plot_losses_and_lrs(self):
        """Plots and saves each individual loss component across epochs."""
        output_dir = os.path.join(self.config.LOG.PATH, self.config.TRAIN.DATA.EXP_DATA_NAME, 'plots')
        os.makedirs(output_dir, exist_ok=True)
        epochs = range(len(self.train_losses))

        loss_dict = {
            "train_total": self.train_losses,
            "val_total": self.val_losses,
            "ce": self.loss_ce_per_epoch,
            "sparsity": self.loss_sparsity_per_epoch,
            "chunk_ce": self.loss_chunk_ce_per_epoch,
            "contrastive": self.loss_contrastive_per_epoch,
        }

        for name, values in loss_dict.items():
            plt.figure(figsize=(8,5))
            plt.plot(epochs, values, label=name)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{name} Loss over Epochs")
            plt.legend()
            path = os.path.join(output_dir, f"{name}_loss.pdf")
            plt.savefig(path, dpi=300)
            plt.close()
            print(f"[PLOT] Saved: {path}")

    def train_mode(self):
        # self.encoder.train()
        self.temporal_branch.train()
        self.attn_scorer.train()
        self.chunk_projection.train()
        self.chunk_aux_classifier.train()
        self.pooling.train()
        self.classifier.train()

    def eval_mode(self):
        # self.encoder.eval()
        self.temporal_branch.eval()
        self.attn_scorer.eval()
        self.chunk_projection.eval()
        self.chunk_aux_classifier.eval()
        self.pooling.eval()
        self.classifier.eval()
