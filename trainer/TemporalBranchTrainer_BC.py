"""
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
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from trainer.BaseTrainer import BaseTrainer
from neural_encoders.model.PhysMamba import PhysMamba

from decoders.TemporalBranch import TemporalBranch

from modules.AttnScorer import AttnScorer
from modules.ProjectionHead import ProjectionHead
from modules.ChunkForward import ChunkForwardModule
from modules.ClassificationHead import ClassificationHead
from modules.ChunkAuxClassifier import ChunkAuxClassifier
from modules.TopKPooling import TopKSoftPooling
from modules.GatedAttentionPooling import GatedAttentionPooling

from tools.utils import SupConLossTopK, AlignCosineLoss
from tools.utils import reconstruct_sessions, run_tsne_and_plot

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
    + KLAlignLoss(logits_topk, logits_gated) × align_weight
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
        self.decay = 1e-4  # weight decay factor
        
        wandb.init(
            project="TemporalReMOTION",
            name=f"Exp_{self.config.TRAIN.MODEL_FILE_NAME}_CosSim_NoEntimax",
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
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
            temperature=1.0
        ).to(self.device)
        # self.alpha = 0.5 # diversity-attention balance: 0.0 ~ 1.0
        
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
        
        # ------------------------------ Top-K Soft Pooling ------------------------------
        """Top-K chunk selection based on attention scores for KL alignment"""
        self.topk_pooler = TopKSoftPooling(
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
            temperature=1.0
        ).to(self.device)
        
        # ------------------------------ Gated Attention Pooling ------------------------------
        """Aggregates chunk embeddings to a session-level embedding using Gated self-attention
           Includes entropy regularization (used in sparsity loss)"""
        self.pooling = GatedAttentionPooling(
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
            temperature=0.7
        ).to(self.device)
        self.sparsity_weight = 0.3

        # ----------------------------- Classification Head -----------------------------
        """Simple MLP head for predicting emotion labels from pooled session embedding"""
        self.classifier = ClassificationHead(
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
            num_classes=config.TRAIN.NUM_CLASSES
        ).to(self.device)
        for p in self.classifier.parameters():
            p.requires_grad = False

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
        self.contrastive_weight = 1.0
        self.top_k_ratio = 0.5
        
        """CrossEntropyLoss: standard cross-entropy loss for classification"""
        self.criterion = nn.CrossEntropyLoss() # <================================ HOW ABOUT FOCAL LOSS???
        self.ce_weight = 0.0
        
        """AlignCosineLoss: cosine distance between Top-K pooled vs Gated pooled embeddings.
        Encourages representational alignment across attention streams."""
        self.cosinse_criterion = AlignCosineLoss()

        # ------------------------------- Optimizer Setup -------------------------------
        param_groups = {'decay': [], 'no_decay': []}
        for module in [
            self.temporal_branch, self.attn_scorer, self.pooling, self.chunk_projection, 
            self.chunk_aux_classifier, self.classifier
        ]:
            for name, param in module.named_parameters():
                wd = 0.0 if ("bias" in name or "LayerNorm" in name) else self.decay
                param_groups['no_decay' if wd == 0.0 else 'decay'].append({"params": param, "weight_decay": wd})

        self.optimizer = optim.AdamW(
            param_groups['decay'] + param_groups['no_decay'],
            lr=config.TRAIN.LR
        )

        # ---------------------------- Learning Rate Scheduler ----------------------------
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        """
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epoch,  # 20
            eta_min=1e-6
        )
        """
        
        # ---------------------------- Session Reconstruction ----------------------------
        self.reconstruct_sessions = partial(reconstruct_sessions, self)
        
        # ------------------------------ t-SNE visualization ------------------------------
        self.run_tsne_and_plot = partial(run_tsne_and_plot, self)
        
        # -------------------------------- History Tracking --------------------------------
        self.best_train_loss = float('inf')
        self.best_cos_loss = float('inf')
        self.best_chunk_ce_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_val_metric = 0.0
        
        self.train_losses = []
        self.val_losses = []
        self.loss_ce_per_epoch = []
        self.loss_contrastive_per_epoch = []
        self.loss_sparsity_per_epoch = []
        self.loss_cos_per_epoch = []
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
    
    def update_loss_weights_by_epoch(self, epoch, avg_entropy_attn=None, avg_entropy_gated=None, apply_entropy_update=True):
        """
        Update dynamic loss weights, gradient flow, top-k ratio,
        and temperature/thresholds based on current epoch and entropy.
        
        Phase-aware dynamic scheduling of:
        - Loss weights (SupCon, CE, KL, ChunkAux, Sparsity)
        - Gradient control for key modules (AttnScorer, Projection, Classifier, etc.)
        - Top-K sampling ratio and temperature decay
        
        Grad flow control:
         - Classifier receives CE input from epoch 20 onward
         - But gradient is enabled only from epoch 30 for stable CE ramp-up
        """

        warmup_epoch = self.warm_up              # 0 ~ 19
        ce_ramp_epochs = 25                      # 20 ~ 44
        ce_ramp_end = warmup_epoch + ce_ramp_epochs
        supcon_cutoff = 35                       # after this, SupCon off
        aux_ce_cutoff = 34

        # ===== SupCon weight schedule =====
        # SupCon Weight (from 1.0 → 0.2 → 0.05 → 0)
        if epoch < warmup_epoch:
            self.contrastive_weight = 1.0
        elif epoch < 25:
            self.contrastive_weight = max(0.2, 1.0 - ((epoch - warmup_epoch) / 5) * 0.8)  # → 0.2
        elif epoch < supcon_cutoff:
            self.contrastive_weight = max(0.05, 0.2 - ((epoch - 25) / (supcon_cutoff - 25)) * 0.15)
        else:
            self.contrastive_weight = 0.0

        # ===== CE loss ramp-up schedule =====
        if epoch < warmup_epoch:
            self.ce_weight = 0.0
        elif epoch < 30:
            progress = (epoch - warmup_epoch) / (30 - warmup_epoch)
            self.ce_weight = 1.0 / (1.0 + math.exp(-10 * (progress - 0.5))) * 0.7
        else:
            self.ce_weight = min(1.0, 0.7 + (epoch - 30) / (ce_ramp_end - 30) * 0.3)
        """
        if epoch < warmup_epoch:
            self.ce_weight = 0.0
        elif epoch < 30:
            # ramp up faster to prepare classifier input
            self.ce_weight = min(1.0, (epoch - warmup_epoch) / (30 - warmup_epoch) * 0.7)
        else:
            # ramp up smoothly to full
            self.ce_weight = min(1.0, 0.7 + (epoch - 30) / (ce_ramp_end - 30) * 0.3)
        """

        # ===== Sparsity loss decay =====
        if epoch < warmup_epoch:
            self.sparsity_weight = 0.3
        elif epoch < ce_ramp_end:
            self.sparsity_weight = 0.3 * (1 - (epoch - warmup_epoch) / ce_ramp_epochs)
        else:
            self.sparsity_weight = 0.05

        # ===== Chunk-level CE (Top-K + session label) =====
        self.chunk_ce_weight = 0.0 if epoch < 20 else (0.2 if epoch <= aux_ce_cutoff else 0.0)

        # ===== KL AlignLoss weight =====
        self.align_weight = 0.3 if 25 <= epoch <= 35 else 0.0

        # ===== Top-K ratio schedule =====
        self.top_k_ratio = max(0.2, 0.5 - 0.4 * (epoch / self.max_epoch))

        # ===== Contrastive temperature decay =====
        self.contrastive_loss_fn.temperature = 0.1 # (fixed low)
        # self.contrastive_loss_fn.temperature = max(0.07, 0.3 - 0.23 * (epoch / self.max_epoch))

        # ===== Gradient control: AttnScorer, ProjectionHead, TopKPooling, Classifier, GatedPooling, AuxCE =====
        for p in self.attn_scorer.parameters():
            p.requires_grad = epoch < supcon_cutoff

        for p in self.chunk_projection.parameters():
            p.requires_grad = epoch < supcon_cutoff

        for p in self.pooling.parameters():
            p.requires_grad = epoch >= 15
        
        for p in self.topk_pooler.parameters():
            p.requires_grad = (epoch >= 15 and epoch <= 34)

        for p in self.classifier.parameters():
            p.requires_grad = epoch >= 30

        for p in self.chunk_aux_classifier.parameters():
            p.requires_grad = 20 <= epoch <= aux_ce_cutoff
        
        # ===== Adaptive temperature/threshold scheduling =====
        if epoch < 20 and apply_entropy_update:
            if avg_entropy_attn is not None:
                self.attn_scorer.temperature = max(0.7, 1.2 - avg_entropy_attn)
            """
            if avg_entropy_gated is not None:
                self.pooling.temperature = max(0.3, 1.2 - avg_entropy_gated)
            """
            # GatedPooling: fixed
            self.pooling.temperature = 0.7  # fixed, stable CE learning
            
        # ===== SupConLoss sampling scheduling (e.g., threshold, top-k mask, etc.) =====
        self.contrastive_loss_fn.schedule_params(epoch, self.max_epoch)

    def forward_batch(self, batch, epoch):
        """
            Forward function for one training batch (B sessions) during epoch `epoch`.

            Handles:
            - TemporalBranch → chunk-wise rPPG embeddings
            - AttnScorer attention: softmax (0–19), raw scores (20+)
            - SupConLossTopK with Top-K + threshold selection from epoch ≥ 20
            - GatedPooling for CE loss
            - TopKSoftPooling for KLAlignLoss (epoch 25–35)
            - ChunkAuxClassifier (epoch 20–34 only) for chunk-level CE supervision
            - SparsityLoss based on entropy: AttnScorer (<35) vs Gated (≥35)
            - t-SNE logging for chunk/session visualization
            - Full debug printouts for embeddings, attention, entropy, losses
        """

        chunk_tensors, batch_labels, batch_session_ids, _ = batch
        B = len(batch_labels)

        self.chunk_embeddings_for_tsne.clear()
        self.chunk_labels_for_tsne.clear()
        self.session_embeddings_for_tsne.clear()
        self.session_labels_for_tsne.clear()

        # === Step 0: Initialization ===
        session_embeds = []
        target_labels = []
        entropy_list = []
        all_proj_embeddings, all_labels_tensor, all_attn_weights = [], [], []
        all_aux_logits, all_aux_labels = [], []
        all_topk_embeds = []
        chunk_ce_losses = []  # Per-session chunk-level CE loss
        per_session_topk_embeds = []  # For KL alignment
        entropy_attn_list = []
        entropy_gated_list = []

        # === Step 1: Iterate each session in batch ===
        for i in range(B):
            chunks = chunk_tensors[i]  # (T, 1, 128)
            label = batch_labels[i]
            sess_id = batch_session_ids[i]
            num_chunks = chunks.shape[0]
            chunk_embeds, chunk_proj_embeds = [], []

            print(f"[DEBUG] Session {sess_id} -> #chunks = {num_chunks}")

            # Step 1.1: Extract embeddings for each chunk
            for j in range(num_chunks):
                chunk_data = chunks[j].unsqueeze(0).float().to(self.device)
                emb = self.forward_single_chunk_checkpoint(chunk_data)  # (1, D)
                proj = self.chunk_projection(emb)  # (1, D)
                chunk_embeds.append(emb)
                chunk_proj_embeds.append(proj)

            chunk_embeds = torch.cat(chunk_embeds, dim=0).unsqueeze(0)  # (1, T, D)
            chunk_proj_embeds = torch.cat(chunk_proj_embeds, dim=0)    # (T, D)
            chunk_labels_tensor = torch.tensor([label] * num_chunks, dtype=torch.long, device=self.device)

            # Step 1.2: t-SNE logging for visualization
            for i, cemb in enumerate(chunk_embeds[0]):
                print(f"  Chunk {i}: mean={cemb.mean().item():.4f}, std={cemb.std().item():.4f}")
                self.chunk_embeddings_for_tsne.append(cemb.detach().cpu().numpy())
                self.chunk_labels_for_tsne.append(label)

            # Step 1.3: Compute attention weights
            attn_scored, entropy_attn = self.attn_scorer(chunk_embeds, return_entropy=True)
            if epoch < 20:
                attn_weights = F.softmax(attn_scored / self.attn_scorer.temperature, dim=1)
            else:
                attn_weights = attn_scored  # Top-K selection applied later

            # Step 1.4: Gated pooling for session-level CE
            pooled, attn_gated, entropy_gated = self.pooling(chunk_embeds, return_weights=True, return_entropy=True)
            session_embeds.append(pooled.squeeze(0))
            target_labels.append(label)
            self.session_embeddings_for_tsne.append(pooled.squeeze(0).detach().cpu().numpy())
            self.session_labels_for_tsne.append(label)

            # Step 1.5: Sparsity regularization loss
            if epoch < 35:
                entropy = self.sparsity_weight * entropy_attn
                entropy_attn_list.append(entropy_attn.detach())
            else:
                entropy = self.sparsity_weight * entropy_gated
                entropy_gated_list.append(entropy_gated.detach())
            entropy_list.append(entropy)
            
            # --- DEBUG: Attention Information ---
            attn_np = attn_scored.detach().cpu().squeeze().numpy()
            print(f"[DEBUG][Attn Weights] {attn_np.tolist()}")
            print(f"[DEBUG][Attn Sparsity] mean={attn_np.mean():.4f}, std={attn_np.std():.4f}, entropy={entropy_attn.item():.4f}")
            if epoch >= self.warm_up:
                gated_np = attn_gated.detach().cpu().squeeze().numpy()
                print(f"[DEBUG][Gated Attn] {gated_np.tolist()}")
                print(f"[DEBUG][Gated Sparsity] mean={gated_np.mean():.4f}, std={gated_np.std():.4f}, entropy={entropy_gated.item():.4f}")
            
            # --- For session-level t-SNE plot --- 
            sess_emb = pooled.squeeze(0)  # (D,)
            self.session_embeddings_for_tsne.append(sess_emb.detach().cpu().numpy())
            self.session_labels_for_tsne.append(label)

            # Step 1.6: Store contrastive info (projection + attention)
            if epoch < 20:
                weighted_proj = attn_weights.squeeze(0) * chunk_proj_embeds
                all_proj_embeddings.append(weighted_proj)
                all_labels_tensor.append(chunk_labels_tensor)
                all_attn_weights.append(attn_scored.view(-1))
            else:
                all_proj_embeddings.append(chunk_proj_embeds)
                all_labels_tensor.append(chunk_labels_tensor)
                all_attn_weights.append(attn_scored.squeeze(0))

            # Step 1.7: Apply ChunkAuxClassifier on Top-K for CE
            if 20 <= epoch <= 34 and self.chunk_ce_weight > 0:
                k = min(num_chunks, max(2, int(num_chunks * self.top_k_ratio)))
                attn = attn_scored.view(-1)
                above_thd = (attn > 0.01).nonzero(as_tuple=True)[0]
                topk_idx = above_thd[torch.topk(attn[above_thd], k=k).indices] if len(above_thd) >= k else torch.topk(attn, k=k).indices
                topk_embeds = chunk_embeds[0][topk_idx]  # (K, D)
                per_session_topk_embeds.append(topk_embeds)
                topk_logits = self.chunk_aux_classifier(topk_embeds)
                topk_labels = torch.tensor([label] * len(topk_logits), dtype=torch.long, device=self.device)
                loss_i = self.criterion(topk_logits, topk_labels)
                chunk_ce_losses.append(loss_i)
            else:
                per_session_topk_embeds.append(None)

        # === Step 2: SupCon Loss ===
        if epoch < 20:
            proj_concat = torch.cat(all_proj_embeddings, dim=0)
            label_concat = torch.cat(all_labels_tensor, dim=0)
            if label_concat.unique().numel() >= 2:
                proj_norm = F.normalize(proj_concat, dim=-1)
                print(f"[DEBUG] norm mean={proj_norm.mean().item():.4f}, std={proj_norm.std().item():.4f}")
                contrastive_term = self.contrastive_loss_fn(proj_norm, label_concat)
            else:
                print("[Skip SupCon] Only one unique label in batch.")
                contrastive_term = torch.tensor(0.0, device=self.device)
        else:
            selected_proj, selected_labels = [], []
            for proj, lbl, attn in zip(all_proj_embeddings, all_labels_tensor, all_attn_weights):
                if len(attn) < 1:
                            continue
                k = min(len(attn), int(len(attn) * self.top_k_ratio))
                attn = attn.view(-1)
                threshold = 0.01
                above_thd = (attn > threshold).nonzero(as_tuple=True)[0]

                # Thresholding
                if len(above_thd) >= k:
                    topk_indices = torch.topk(attn[above_thd], k=k).indices
                    selected_idx = above_thd[topk_indices]
                else:
                    selected_idx = torch.topk(attn, k=k).indices
                selected_proj.append(proj[selected_idx])
                selected_labels.append(lbl[selected_idx])
            if len(selected_proj) > 0:
                proj_concat = torch.cat(selected_proj, dim=0)
                label_concat = torch.cat(selected_labels, dim=0)
                if label_concat.unique().numel() >= 2:
                    proj_norm = F.normalize(proj_concat, dim=-1)
                    print(f"[DEBUG] norm mean={proj_norm.mean().item():.4f}, std={proj_norm.std().item():.4f}")
                    contrastive_term = self.contrastive_loss_fn(proj_norm, label_concat)
                else:
                    print("[Skip SupCon] Only one unique label in batch.")
                    contrastive_term = torch.tensor(0.0, device=self.device)
            else:
                contrastive_term = torch.tensor(0.0, device=self.device)

        # === Step 3: Session-level CE Loss ===
        batch_embeds = torch.stack(session_embeds, dim=0)
        label_tensor = torch.tensor(target_labels, dtype=torch.long, device=self.device)
        outputs = self.classifier(batch_embeds)
        ce_loss = self.criterion(outputs, label_tensor)
        ce_loss_scaled = self.ce_weight * ce_loss

        # === Step 4: Sparsity Loss ===
        sparsity_loss = torch.stack(entropy_list).mean()
        sparsity_loss_scaled = self.sparsity_weight * sparsity_loss

        # === Step 5: KL AlignLoss ===
        cosine_losses = []
        for i in range(B):
            if self.align_weight > 0 and per_session_topk_embeds[i] is not None:
                pooled_topk, _ = self.topk_pooler(per_session_topk_embeds[i]) # (D, )
                pooled_gated = session_embeds[i] 
                cosine_sim = self.cosinse_criterion(
                    pooled_gated.unsqueeze(0), # (1, D)
                    pooled_topk.unsqueeze(0)   # (1, D)
                )
                cosine_losses.append(cosine_sim)
        cosine_loss = torch.stack(cosine_losses).mean() if len(cosine_losses) > 0 else torch.tensor(0.0, device=self.device)
        cosine_loss_scaled = self.align_weight * cosine_loss

        # === Step 6: Chunk-level CE Loss ===
        chunk_ce_loss = torch.stack(chunk_ce_losses).mean() if len(chunk_ce_losses) > 0 else torch.tensor(0.0, device=self.device)
        chunk_ce_loss_scaled = self.chunk_ce_weight * chunk_ce_loss

        # === Step 7: Total Loss Composition ===
        contrastive_term_scaled = self.contrastive_weight * contrastive_term
        loss_total = contrastive_term_scaled + ce_loss_scaled + sparsity_loss_scaled + cosine_loss_scaled + chunk_ce_loss_scaled

        preds = torch.argmax(outputs, dim=1)
        
        # --- for adaptive temperature/threshold scheduling
        entropy_attn_mean = torch.stack(entropy_attn_list).mean() if len(entropy_attn_list) > 0 else torch.tensor(0.0, device=self.device)
        entropy_gated_mean = torch.stack(entropy_gated_list).mean() if len(entropy_gated_list) > 0 else torch.tensor(0.0, device=self.device)

        print("\n📊 [Loss Breakdown & Prediction]")
        print("─" * 60)
        print(f"🔧 Loss Weights:\n   ▸ CE         : {self.ce_weight:.2f}\n   ▸ Contrastive: {self.contrastive_weight:.2f}\n   ▸ Sparsity   : {self.sparsity_weight:.2f}\n   ▸ KLAlign    : {self.align_weight:.2f}\n   ▸ ChunkCE    : {self.chunk_ce_weight:.2f}")
        print("\n📉 Loss Components:")
        print(f"   ▸ CE         : {ce_loss.item():.4f} × {self.ce_weight:.2f} = {ce_loss_scaled.item():.4f}")
        print(f"   ▸ Contrastive: {contrastive_term.item():.4f} × {self.contrastive_weight:.2f} = {contrastive_term_scaled.item():.4f}")
        print(f"   ▸ Sparsity   : {sparsity_loss.item():.4f} x {self.sparsity_weight:.2f} = {sparsity_loss_scaled.item():.4f}")
        print(f"   ▸ Cosine Align   : {cosine_loss.item():.4f} x {self.align_weight:.2f} = {cosine_loss_scaled.item():.4f}")
        print(f"   ▸ Chunk CE   : {chunk_ce_loss.item():.4f} x {self.ce_weight:.2f} = {ce_loss_scaled.item():.4f}")
        print(f"   ▸ Entropy    : Scorer = {entropy_attn_mean:.4f}, Gated = {entropy_gated_mean:.4f}")
        print(f"   ▶ Total Loss : {loss_total.item():.4f}")
        print("─" * 60)

        # --- Optional Prediction Logging ---
        if self.ce_weight > 0:
            probs = torch.softmax(outputs, dim=1)
            print("\n🔎 [Batch Predictions]")
            for i in range(len(label_tensor)):
                gt = label_tensor[i].item()
                prob = probs[i].detach().cpu().numpy()
                pred = np.argmax(prob)
                confidence = prob[pred]
                print(f"   ▸ Session {batch_session_ids[i]} | GT: {gt} | Pred: {pred} | Conf: {confidence:.4f} | Prob: {np.round(prob, 4)}")
            print("═" * 60)
        else:
            print("🔎 [Batch Predictions] Skipped (CE not active)")

        return (
            loss_total, ce_loss, # Full loss for backward
            contrastive_term_scaled, sparsity_loss_scaled, cosine_loss_scaled, chunk_ce_loss_scaled, # For raw monitoring (Already scaled)
            sparsity_loss.detach(), preds, label_tensor,
            chunk_ce_loss.item(), cosine_loss.item(), # For best model check
            entropy_attn_mean, entropy_gated_mean
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
        - GatedAttentionPooling aggregates session embeddings (used for CE)
        - Classifier makes session-level predictions

        🎯 Multi-phase Loss Scheduling:
        - SupConLossTopK: early exploration → Top-K-based → frozen
        - CE Loss: ramp-up from epoch 20 → dominant in later epochs
        - SparsityLoss: attention entropy regularization
        - KLAlignLoss: TopKSoftPooling vs GatedPooling logits (epoch 30–44)
        - ChunkAuxClassifier: weak CE on Top-K chunks (epoch 20–34)
        """
        torch.autograd.set_detect_anomaly(True)
        
        # Initialize per-epoch loss trackers
        self.loss_ce_per_epoch = []
        self.loss_contrastive_per_epoch = []
        self.loss_sparsity_per_epoch = []
        self.loss_cos_per_epoch = []
        self.loss_chunk_ce_per_epoch = []

        for epoch in range(self.max_epoch):
            self.train_mode()

            # === Placeholder for entropy tracking ===
            entropy_attn_all = []
            entropy_gated_all = []

            # === Temporary call for logging current scheduling phase only ===
            self.update_loss_weights_by_epoch(epoch, avg_entropy_attn=0.5, avg_entropy_gated=0.5, apply_entropy_update=False)

            # === Epoch Settings Print ===
            print("\n" + "=" * 80)
            print(f"🧠 [Epoch {epoch}/{self.max_epoch}] - Training Phase")
            print("-" * 80)
            print("🔧 Loss Scheduling:")
            print(f"   ▸ CE         : {'Inactive (Warm-up)' if self.ce_weight == 0.0 else f'Ramping-up ({self.ce_weight:.2f})' if self.ce_weight < 1.0 else 'Fully Active'}")
            print(f"   ▸ Contrastive: {'Off' if self.contrastive_weight == 0.0 else f'Decaying ({self.contrastive_weight:.2f})' if self.contrastive_weight < 1.0 else 'Full'}")
            print(f"   ▸ KL Align   : {'On ({:.2f})'.format(self.align_weight) if self.align_weight > 0 else '[OFF]'}")
            print(f"   ▸ Chunk CE   : {'On ({:.2f})'.format(self.chunk_ce_weight) if self.chunk_ce_weight > 0 else '[OFF]'}")
            print(f"   ▸ Sparsity   : {self.sparsity_weight:.2f}" if self.sparsity_weight > 0 else "   ▸ Sparsity   : [OFF]")
            print("🎯 SupCon Sampling Strategy:")
            print(f"   ▸ Max Pos     : {self.contrastive_loss_fn.max_pos}")
            print(f"   ▸ Max Neg     : {self.contrastive_loss_fn.max_neg}")
            print(f"   ▸ Pos/Neg Ratio: {self.contrastive_loss_fn.pos_neg_ratio:.2f}")
            print(f"   ▸ Temperature : {self.contrastive_loss_fn.temperature:.2f}")
            print(f"   ▸ AttnTemp    : {self.attn_scorer.temperature:.2f} | GatedTemp: {self.pooling.temperature:.2f}")
            print("-" * 80)

            tbar = tqdm(data_loader["train"], desc=f"Epoch {epoch}", ncols=80)
            running_loss = 0.0
            all_preds, all_labels = [], []
            
            ce_total = 0.0
            contrastive_total = 0.0
            sparsity_total = 0.0
            cos_total = 0.0
            chunk_ce_total = 0.0
            num_batches = len(tbar)

            for idx, batch in enumerate(tbar):
                (
                    loss_total, ce_loss, contrastive_loss, sparsity_loss,
                    cos_loss, chunk_ce_loss, _, preds, label_tensor,
                    train_cos, train_chunk_ce,
                    entropy_attn, entropy_gated  
                ) = self.forward_batch(batch, epoch)

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
                cos_total += train_cos
                chunk_ce_total += train_chunk_ce

                # === GRADIENT CHECK ===
                if idx % 30 == 0:
                    modules = [
                        # ("Encoder", self.encoder),
                        ("Temporal", self.temporal_branch),
                        ("AttentionScorer", self.attn_scorer),
                        ("Projection", self.chunk_projection),
                        ("GatedAttentionPooling", self.pooling),
                        ("Classifier", self.classifier),
                        ("ChunkAuxClassifier", self.chunk_aux_classifier),
                        ("TopKPooler", self.topk_pooler),
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
            self.update_loss_weights_by_epoch(epoch, avg_entropy_attn, avg_entropy_gated, apply_entropy_update=True)

            # === Training summary ===
            avg_loss = running_loss / num_batches
            acc = accuracy_score(all_labels, all_preds)
            self.train_losses.append(avg_loss)
            self.loss_ce_per_epoch.append(ce_total / num_batches)
            self.loss_contrastive_per_epoch.append(contrastive_total / num_batches)
            self.loss_sparsity_per_epoch.append(sparsity_total / num_batches)
            self.loss_cos_per_epoch.append(cos_total / num_batches)
            self.loss_chunk_ce_per_epoch.append(chunk_ce_total / num_batches)

            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Acc = {acc:.4f}")
            
            self.run_tsne_and_plot(level="chunk", epoch=epoch, phase="train")
            self.run_tsne_and_plot(level="session", epoch=epoch, phase="train")

            # === Best Model Check by Multiple Metrics ===
            val_loss, metrics = self.valid(data_loader, epoch=epoch)
            self.val_losses.append(val_loss)

            # Save by highest accuracy
            if metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = metrics['accuracy']
                self.best_epoch = epoch
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag="best_acc")
                print(f"[SAVE] Best ACC model updated at epoch {epoch}")

            # Save by highest F1
            if metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = metrics['f1']
                self.best_epoch = epoch
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag="best_f1")
                print(f"[SAVE] Best F1 model updated at epoch {epoch}")

            # Save by best sum of metrics (Acc + F1)
            if metrics['f1'] + metrics['accuracy'] > self.best_val_metric:
                self.best_val_metric = metrics['f1'] + metrics['accuracy']
                self.best_epoch = epoch
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag="best_metric")
                print(f"[SAVE] Best Metric model updated at epoch {epoch}")

            # Save by lowest validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag="best_loss")
                print(f"[SAVE] Best Validation loss model updated at epoch {epoch}")
                
            if avg_loss < self.best_train_loss:
                self.best_train_loss = avg_loss
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag="best_train_loss")
                print(f"[SAVE] Best Train Loss model updated at epoch {epoch}")
            if cos_loss > 0 and train_cos < self.best_cos_loss:
                self.best_cos_loss = train_cos
                self.save_model(epoch=epoch, val_loss=val_loss, train_loss=avg_loss, val_metrics=metrics, tag="best_cos")
                print(f"[SAVE] Best KL loss model updated at epoch {epoch}")
            if chunk_ce_loss > 0 and train_chunk_ce < self.best_chunk_ce_loss:
                self.best_chunk_ce_loss = train_chunk_ce
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
                "loss/train_cos": self.loss_cos_per_epoch[-1],
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
                    entropy_val = session_entropies.get(sid, torch.tensor(0.0)).item()
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
        tags = ["best_acc", "best_f1", "best_metric", "best_loss", 
                "best_train_loss", "best_cos", "best_chunkce", "last_epoch"]
        results = {}

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
                        batch=batch, idx=idx, phase='test'
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
            tag (str): best_acc, best_f1, best_loss, best_metric
        """
        model_save_dir = "./saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        base_name = self.config.TRAIN.MODEL_FILE_NAME

        # === Save all modules ===
        model_dict = {
            "temporal_branch": self.temporal_branch.state_dict(),
            "attention_pooling": self.pooling.state_dict(),
            "classifier": self.classifier.state_dict(),
            "projection_head": self.chunk_projection.state_dict(),
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
        This includes TemporalBranch, AttentionPooling, Classifier, and ProjectionHead.

        Args:
            path (str): Path to the .pth file containing the saved model weights.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.temporal_branch.load_state_dict(checkpoint["temporal_branch"])
        self.pooling.load_state_dict(checkpoint["attention_pooling"])
        self.classifier.load_state_dict(checkpoint["classifier"])
        self.chunk_projection.load_state_dict(checkpoint["projection_head"])
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
            "contrastive": self.loss_contrastive_per_epoch,
            "sparsity": self.loss_sparsity_per_epoch,
            "cos": self.loss_cos_per_epoch,
            "chunk_ce": self.loss_chunk_ce_per_epoch,
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
        self.topk_pooler.train()
        self.chunk_aux_classifier.train()
        self.pooling.train()
        self.classifier.train()

    def eval_mode(self):
        # self.encoder.eval()
        self.temporal_branch.eval()
        self.attn_scorer.eval()
        self.chunk_projection.eval()
        self.topk_pooler.eval()
        self.chunk_aux_classifier.eval()
        self.pooling.eval()
        self.classifier.eval()

    """
    def train(self, data_loader):
        '''
        Main training loop for session-level emotion classification using:

        - TemporalBranch to extract chunk-wise rPPG embeddings
        - AttnScorer to assign attention scores to each chunk
        - Hard Top-K + threshold chunk selection for contrastive learning
        - Projection of Hard-selected Top-k embeddings into contrastive space
        - SupConLoss applied to normalized projected Top-K embeddings
        - Gated attention pooling for session-level embedding
        - CE loss on session-level predictions

        Total Loss = 
            SupConLossTopK(chunk_proj[topk], label) × contrastive_weight
            + CE(classifier(session_repr), label) × ce_weight
        '''
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.max_epoch):
            self.train_mode()
                
            # Dynamic loss weights per epoch
            '''
            Schedule CE and contrastive weights by epoch phase
            Phase-specific weight scheduling
            CE weight and top_k_ratio dynamically adjusted based on epoch to balance learning focus
            '''
            self.update_loss_weights_by_epoch(epoch)
            
            # Print epoch header and loss scheduling summary
            print("\n" + "=" * 80)
            print(f"🧠 [Epoch {epoch}/{self.max_epoch}] - Training Phase")
            print("-" * 80)
            print("🔧 Loss Scheduling:")

            # Show Cross-Entropy weight status
            if self.ce_weight == 0.0:
                print("   ▸ CE         : Inactive (Warm-up)")
            elif self.ce_weight < 1.0:
                print(f"   ▸ CE         : Ramping-up ({self.ce_weight:.2f})")
            else:
                print("   ▸ CE         : Fully Active")

            # Show Contrastive weight status
            if self.contrastive_weight == 0.0:
                print("   ▸ Contrastive: Off")
            elif self.contrastive_weight < 1.0:
                print(f"   ▸ Contrastive: Decaying ({self.contrastive_weight:.2f})")
            else:
                print("   ▸ Contrastive: Full")

            # Show current Top-K ratio for SupConLoss
            # print(f"   ▸ Top-K Ratio: {self.contrastive_loss_fn.top_k_ratio:.2f}")

            # Show whether Sparsity Loss is active
            if self.sparsity_weight > 0:
                print(f"   ▸ Sparsity   : {self.sparsity_weight:.2f}")
            else:
                print("   ▸ Sparsity   : [OFF]")

            print("🎯 SupCon Sampling Strategy:")
            print(f"   ▸ Max Pos     : {self.contrastive_loss_fn.max_pos}")
            print(f"   ▸ Max Neg     : {self.contrastive_loss_fn.max_neg}")
            print(f"   ▸ Pos/Neg Ratio: {self.contrastive_loss_fn.pos_neg_ratio:.2f}")
            print(f"   ▸ Temperature : {self.contrastive_loss_fn.temperature:.2f}")
            print("-" * 80)

            tbar = tqdm(data_loader["train"], desc=f"Epoch {epoch}", ncols=80)

            running_loss = 0.0
            entropy_terms = []
            all_preds, all_labels = [], []

            # Clear t-SNE logging buffers (visualizing chunk/session embeddings per epoch)
            self.chunk_embeddings_for_tsne.clear()
            self.chunk_labels_for_tsne.clear()
            self.session_embeddings_for_tsne.clear()
            self.session_labels_for_tsne.clear()

            for idx, batch in enumerate(tbar):
                chunk_tensors, batch_labels, batch_session_ids, _ = batch
                B = len(batch_labels)

                session_embeds = []
                target_labels = []
                entropy_list = []
                all_proj_embeddings, all_labels_tensor, all_attn_weights = [], [], []
                
                # Each chunk is passed through PhysMamba → TemporalBranch → ProjectionHead
                # Embedding statistics and visualizations (mean/std/t-SNE) are tracked per epoch
                for i in range(B):
                    chunks = chunk_tensors[i]
                    label = batch_labels[i]
                    sess_id = batch_session_ids[i]
                    num_chunks = chunks.shape[0]
                    chunk_embeds, chunk_proj_embeds = [], []
                    print(f"[DEBUG] Session {sess_id} -> #chunks = {num_chunks}")
                    
                    # 1. Chunk-wise encoding → Projection
                    # Extract chunk embeddings via PhysMamba + TemporalBranch
                    for j in range(num_chunks):
                        chunk_data = chunks[j].unsqueeze(0).float().to(self.device) # (1, 1, 128)
                        # chunk_data.requires_grad_()
                        emb = self.forward_single_chunk_checkpoint(chunk_data) # TemporalBranch (1, D)
                        proj = self.chunk_projection(emb)  # ProjectionHead (1, D)
                        chunk_embeds.append(emb)
                        chunk_proj_embeds.append(proj)

                    # Shape: (1, T, D)
                    chunk_embeds = torch.cat(chunk_embeds, dim=0).unsqueeze(0) # (1, T, D)
                    chunk_proj_embeds = torch.cat(chunk_proj_embeds, dim=0)     # (T, D)
                    chunk_labels_tensor = torch.tensor([label] * num_chunks, dtype=torch.long, device=self.device)
                    
                    # --- For chunk-level t-SNE plot ---
                    for i, cemb in enumerate(chunk_embeds[0]):
                        print(f"  Chunk {i}: mean={cemb.mean().item():.4f}, std={cemb.std().item():.4f}")
                        cemb_np = cemb.detach().cpu().numpy() 
                        self.chunk_embeddings_for_tsne.append(cemb_np)
                        self.chunk_labels_for_tsne.append(label)
                        
                    # 2. Attention weights (softmax)
                    # --- Attention Score for SupCon (always from AttnScorer) ---
                    attn_scored, entropy_attn = self.attn_scorer(chunk_embeds, return_entropy=True)  # (B, T, 1)
                    attn_weights = F.softmax(attn_scored / self.attn_scorer.temperature, dim=1)      # (1, T, 1)
                    
                    # --- Gated Attention Pooling for CE ---
                    pooled, attn_gated, entropy_gated = self.pooling(chunk_embeds, return_weights=True, return_entropy=True)
                    session_embeds.append(pooled.squeeze(0))
                    target_labels.append(label)
                    
                    # for sparsity loss
                    entropy = self.contrastive_weight * entropy_attn + (1 - self.contrastive_weight) * entropy_gated
                    entropy_list.append(entropy)
                    
                    # 3. Weighted projection for SupConLossTopK later
                    if epoch < 10:
                        weighted_proj = attn_weights.squeeze(0) * chunk_proj_embeds  # (T, D)
                        all_proj_embeddings.append(weighted_proj)
                        all_labels_tensor.append(chunk_labels_tensor)
                        all_attn_weights.append(attn_scored.view(-1))
                    else:
                        all_proj_embeddings.append(chunk_proj_embeds)
                        all_labels_tensor.append(chunk_labels_tensor)
                        all_attn_weights.append(attn_scored.squeeze(0))
                        
                    # Attention weight debugging
                    attn_np = attn_scored.detach().cpu().squeeze().numpy()
                    print(f"[DEBUG][Attention Weights] {attn_np.tolist()}")
                    print(f"[DEBUG][Attn Sparsity] mean={attn_np.mean():.4f}, std={attn_np.std():.4f}, entropy={entropy_attn.item():.4f}")
                    if epoch >= self.warm_up:
                        attn_np = attn_gated.detach().cpu().squeeze().numpy()
                        print(f"[DEBUG][Gated Attention Weights] {attn_np.tolist()}")
                        print(f"[DEBUG][Gated Attn Sparsity] mean={attn_np.mean():.4f}, std={attn_np.std():.4f}, entropy={entropy_gated.item():.4f}")
                    
                    # --- For session-level t-SNE plot --- 
                    sess_emb = pooled.squeeze(0)  # (D,)
                    self.session_embeddings_for_tsne.append(sess_emb.detach().cpu().numpy())
                    self.session_labels_for_tsne.append(label)
                        
                if epoch < 10:
                    # === Concatenate all samples across batch ===
                    proj_concat = torch.cat(all_proj_embeddings, dim=0)  # (N, D)
                    label_concat = torch.cat(all_labels_tensor, dim=0)   # (N,)
                    # attn_concat = torch.cat(all_attn_weights, dim=0)     # (N,)
                    
                    # === SupConLoss (Soft Attn Weighted, No Top-K) ===
                    if label_concat.unique().numel() >= 2:
                        proj_norm = F.normalize(proj_concat, dim=-1)
                        print(f"[DEBUG] norm mean={proj_norm.mean().item():.4f}, std={proj_norm.std().item():.4f}")
                        contrastive_term = self.contrastive_loss_fn(proj_norm, label_concat)
                    else:
                        print("[Skip SupCon] Only one unique label in batch.")
                        contrastive_term = torch.tensor(0.0, device=self.device)
                else:
                    selected_proj, selected_labels = [], []
                    for proj, lbl, attn in zip(all_proj_embeddings, all_labels_tensor, all_attn_weights):
                        if len(attn) < 1:
                            continue
                        k = min(len(attn), int(len(attn) * self.top_k_ratio))
                        attn = attn.view(-1)
                        threshold = 0.01
                        above_thd = (attn > threshold).nonzero(as_tuple=True)[0]

                        # Thresholding
                        if len(above_thd) >= k:
                            topk_indices = torch.topk(attn[above_thd], k=k).indices
                            selected_idx = above_thd[topk_indices]
                        else:
                            selected_idx = torch.topk(attn, k=k).indices
                        selected_proj.append(proj[selected_idx])
                        selected_labels.append(lbl[selected_idx])
                    if len(selected_proj) > 0:
                        proj_concat = torch.cat(selected_proj, dim=0)
                        label_concat = torch.cat(selected_labels, dim=0)
                        norms = F.normalize(proj_concat, dim=-1)
                        print(f"[DEBUG] norm mean={norms.mean().item():.4f}, std={norms.std().item():.4f}")
                        if len(label_concat.unique()) >= 2:                   
                            contrastive_term = self.contrastive_loss_fn(norms, label_concat)
                        else:
                            print("[Skip SupCon] All labels in batch are identical, skipping contrastive loss.")
                            contrastive_term = torch.tensor(0.0, device=self.device)
                    else:
                        contrastive_term = torch.tensor(0.0, device=self.device)
                
                '''
                # SupConLossTopK: attention weight-based Top-K selection and thresholding
                contrastive_term = self.contrastive_loss_fn(
                    features=proj_concat,         # (K, D)
                    labels=label_concat,        # (K,)
                    attn_weights=attn_concat  # (K,) attention weight
                )
                '''

                
                '''
                    # --- SupConLossTopK: Diversity-aware Top-K + threshold w/ gradient to scorer ---
                    attn_scored = attn_scored.view(-1)  # ensure shape is (T,)
                    threshold = 0.01
                    above_thd = (attn_scored > threshold).nonzero(as_tuple=True)[0]
                    k = self.contrastive_loss_fn.max_pos
                    k = min(len(attn_scored), k)

                    # Prepare candidate indices and embeddings
                    if len(above_thd) >= k:
                        candidate_idx = above_thd
                    else:
                        candidate_idx = torch.arange(len(attn_scored), device=attn_scored.device)

                    # Diversity-aware selection
                    candidate_scores = attn_scored[candidate_idx]                     # (C,) -> Attn
                    candidate_embeds = chunk_embeds.squeeze(0)[candidate_idx]         # (C, D) -> Embeds
                    
                    # Normalize embeddings for cosine similarity
                    normed = F.normalize(candidate_embeds, dim=-1)                    # (C, D) -> Norm
                    sim_matrix = torch.matmul(normed, normed.T)                       # (C, C) -> Cos sim
                    div_matrix = 1.0 - sim_matrix                                     # (C, C) -> Dissimilarity
                    div_score = div_matrix.sum(dim=1)                                 # (C,)   -> Diversity Score

                    # Combine attention and diversity scores (weighted sum)
                    combined_score = self.alpha * candidate_scores + (1 - self.alpha) * div_score  # (C,)

                    # Select top-K diverse + attended embeddings
                    topk_indices = torch.topk(combined_score, k=k).indices
                    selected_idx = candidate_idx[topk_indices]

                    # Project only the Top-K selected weighted embeddings
                    topk_chunk_embeds = chunk_embeds.squeeze(0)[selected_idx]
                    selected_embeds.append(topk_chunk_embeds)
                    selected_proj.append(self.chunk_projection(topk_chunk_embeds))
                    selected_labels.append(chunk_labels_tensor[selected_idx])

                    
                    
                    # --- SupConLossTopK: Hard Top-K + threshold w/ gradient to scorer ---
                    attn_scored = attn_scored.view(-1)   # ensure shape is (T,)

                    # Threshold filtering
                    threshold = 0.01
                    above_thd = (attn_scored > threshold).nonzero(as_tuple=True)[0]
                    k = self.contrastive_loss_fn.max_pos
                    k = min(len(attn_scored), k)

                    if len(above_thd) >= k:
                        topk_indices = torch.topk(attn_scored[above_thd], k=k).indices
                        selected_idx = above_thd[topk_indices]
                    else:
                        selected_idx = torch.topk(attn_scored, k=k).indices
                    
                    # Project only the Top-K selected weighted embeddings
                    topk_chunk_embeds = chunk_embeds.squeeze(0)[selected_idx]
                    selected_embeds.append(topk_chunk_embeds)
                    selected_proj.append(self.chunk_projection(topk_chunk_embeds))
                    selected_labels.append(chunk_labels_tensor[selected_idx])

                if len(selected_proj) > 0:
                    proj_concat = torch.cat(selected_proj, dim=0)
                    label_concat = torch.cat(selected_labels, dim=0)
                    norms = F.normalize(proj_concat, dim=-1) * 10.0 # scaling
                    print(f"[DEBUG] norm mean={norms.mean().item():.4f}, std={norms.std().item():.4f}")
                    if len(label_concat.unique()) >= 2:                   
                        contrastive_term = self.contrastive_loss_fn(norms, label_concat)
                    else:
                        print("[Skip SupCon] All labels in batch are identical, skipping contrastive loss.")
                        contrastive_term = torch.tensor(0.0, device=self.device)
                else:
                    contrastive_term = torch.tensor(0.0, device=self.device)
                '''
                    
                batch_embeds = torch.stack(session_embeds, dim=0)
                label_tensor = torch.tensor(target_labels, dtype=torch.long, device=self.device)
                entropies = torch.stack(entropy_list)

                # Session-level embedding is computed by AttentionPooling and classified by ClassificationHead
                outputs = self.classifier(batch_embeds)
                ce_loss = self.criterion(outputs, label_tensor)
                sparsity_loss = self.sparsity_weight * entropies.mean()

                # Weighted loss calculation
                ce_loss_scaled = self.ce_weight * ce_loss
                contrastive_term_scaled = self.contrastive_weight * contrastive_term

                # Final combined loss
                '''
                    - CE loss (progressively weighted)
                    - Contrastive loss (initially dominant, decays during epoch 20)
                    - Sparsity loss
                '''
                loss_total = ce_loss_scaled + contrastive_term_scaled + sparsity_loss
                
                # Debug logging
                print("\n📊 [Loss Breakdown & Prediction]")
                print("─" * 60)
                print(f"🔧 Loss Weights:")
                print(f"   ▸ CE         : {self.ce_weight:.2f}")
                print(f"   ▸ Contrastive: {self.contrastive_weight:.2f}")
                print(f"   ▸ Sparsity   : {self.sparsity_weight:.2f}")
                
                print("\n📉 Loss Components:")

                if self.ce_weight > 0:
                    print(f"   ▸ CE         : {ce_loss.item():.4f} × {self.ce_weight:.2f} = {ce_loss_scaled.item():.4f}")
                else:
                    print("   ▸ CE         : [SKIPPED] (epoch < warmup)")

                if self.contrastive_weight > 0:
                    print(f"   ▸ Contrastive: {contrastive_term.item():.4f} × {self.contrastive_weight:.2f} = {contrastive_term_scaled.item():.4f}")
                else:
                    print("   ▸ Contrastive: [SKIPPED] (weight = 0)")

                if self.sparsity_weight > 0:
                    print(f"   ▸ Sparsity   : {sparsity_loss.item():.4f}")
                    print(f"   ▸ Entropy   : {entropies.mean():.4f}")
                else:
                    print("   ▸ Sparsity   : [SKIPPED]")
                    print(f"   ▸ Entropy   : {entropies.mean():.4f}")

                print(f"   ▶ Total Loss : {loss_total.item():.4f}")
                print("─" * 60)

                # Prediction only if classifier is on
                if self.ce_weight > 0:
                    probs = torch.softmax(outputs, dim=1)  # (B, num_classes)
                    print("\n🔎 [Batch Predictions]")
                    for i in range(len(label_tensor)):
                        gt = label_tensor[i].item()
                        prob = probs[i].detach().cpu().numpy()
                        pred = np.argmax(prob)
                        confidence = prob[pred]
                        print(f"   ▸ Session {batch_session_ids[i]} | GT: {gt} | Pred: {pred} | Conf: {confidence:.4f} | Prob: {np.round(prob, 4)}")
                    print("═" * 60)
                else:
                    print("🔎 [Batch Predictions] Skipped (CE not active)")

                # Backward
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Log and evaluate
                running_loss += loss_total.item()
                entropy_terms.append(entropies.mean().item())
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_tensor.cpu().numpy())

                # Optionally perform gradient checking every (100 * accumulation_steps) iterations
                if idx % 30 == 0:
                    # print(f"[DEBUG] topk_proj shape: {topk_proj.shape}, loss requires_grad: {loss.requires_grad}")
                    # print(f"[DEBUG] contrastive_term.requires_grad: {contrastive_term.requires_grad}")

                    # === GRADIENT CHECK ===
                    for name, param in self.encoder.named_parameters():
                        if param.requires_grad:
                            if param.grad is not None:
                                print(f"[GradCheck][Encoder] {name} grad norm: {param.grad.norm().item():.6f}")
                            else:
                                print(f"[GradCheck][Encoder] {name} has no grad")
                    for name, param in self.temporal_branch.named_parameters():
                        # print(name, param.requires_grad)
                        if param.grad is not None:
                            print(f"[GradCheck][Temporal] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Temporal] {name} has no grad")
                    for name, param in self.attn_scorer.named_parameters():
                        # print(name, param.requires_grad)
                        if param.grad is not None:
                            print(f"[GradCheck][AttentionScorer] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][AttentionScorer] {name} has no grad")
                    for name, param in self.chunk_projection.named_parameters():
                        # print(name, param.requires_grad)
                        if param.grad is not None:
                            print(f"[GradCheck][Projection] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Projection] {name} has no grad")
                    for name, param in self.pooling.named_parameters():
                        # print(name, param.requires_grad)
                        if param.grad is not None:
                            print(f"[GradCheck][GatedAttentionPooling] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][GatedAttentionPooling] {name} has no grad")
                    for name, param in self.classifier.named_parameters():
                        # print(name, param.requires_grad)
                        if param.grad is not None:
                            print(f"[GradCheck][Classifier] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Classifier] {name} has no grad")
                            
                tbar.set_postfix(loss=f"{running_loss / (idx + 1):.3f}", entropy=f"{np.mean(entropy_terms):.4f}")
            # Epoch summary
            avg_loss = running_loss / len(tbar)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Acc = {accuracy_score(all_labels, all_preds):.4f}")

            self.run_tsne_and_plot(level="chunk", epoch=epoch, phase="train")
            self.run_tsne_and_plot(level="session", epoch=epoch, phase="train")

            val_loss, metrics = self.valid(data_loader, epoch=epoch)
            self.val_losses.append(val_loss)

            if metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = metrics['accuracy']
                self.best_epoch = epoch
                self.save_model(epoch=epoch, val_metrics=metrics, tag="best_acc")
                print(f"[SAVE] Best ACC model updated at epoch {epoch}")
            if metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = metrics['f1']
                self.best_epoch = epoch
                self.save_model(epoch=epoch, val_metrics=metrics, tag="best_f1")
                print(f"[SAVE] Best F1 model updated at epoch {epoch}")
            if metrics['f1'] + metrics['accuracy'] > self.best_val_metric:
                self.best_val_metric = metrics['f1'] + metrics['accuracy']
                self.best_epoch = epoch
                self.save_model(epoch=epoch, val_metrics=metrics, tag="best_metric")
                print(f"[SAVE] Best Metric model updated at epoch {epoch}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model(epoch=epoch, val_loss=val_loss, tag="best_loss")
                print(f"[SAVE] Best Validation loss model updated at epoch {epoch}")
            if epoch + 1 == self.max_epoch:
                self.save_model(epoch=epoch, val_loss=val_loss, tag="last_epoch")
                print(f"[SAVE] Best Validation loss model updated at epoch {epoch}")
        
        self.plot_losses_and_lrs()
    """