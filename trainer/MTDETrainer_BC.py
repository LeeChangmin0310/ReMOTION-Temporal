"""
rPPG  →  MTDE (encoder)  →  chunk-embeddings
                         ↓
                    AttnScorer (raw scores)
                         ↓
        ┌────────────┬────────────┐
        │            │            │
   SupConLoss    Chunk-CE*    GatedPooling
        ↓            │              ↓
   (Phase-0)     (Phase-1)    session-embedding
                                        ↓
                               ClassificationHead
                                        ↓
                           CrossEntropyLoss + (optional) SparsityLoss

*Chunk-CE = Focal-CE on Top-K chunks (Phase-1 only)

========================= 50-Epoch Curriculum =========================

PH-0  (Epoch 0–14)  –  Representation Exploration
----------------------------------------------------------------------
• SupConLossTopK over *all* chunks (α-softmax, no Top-K mask)
• SparsityLoss(Entropy) warm-start λ = 0.05   → adaptive (≤ 0.20)
• AttnScorer + ProjectionHead trainable
• GatedPooling active (but CE weight = 0)
• Classifier / ChunkAuxClassifier frozen

PH-1  (Epoch 15–29) –  Chunk Discriminability
----------------------------------------------------------------------
• Top-K ratio 0.70 → 0.40  (linear)
• Focal Chunk-CE weight 0.30 → 1.00  (first 4 epochs)
• Session-level CE “cushion” weight = 0.20 (constant)
• SupConLoss weight = 0.10  (still trains ProjectionHead)
• GatedPooling / Classifier now trainable

PH-2  (Epoch 30–49) –  Session-Level Fine-tune
----------------------------------------------------------------------
• SupConLoss OFF, Chunk-CE OFF
• AttnScorer + ProjectionHead **frozen**
• Classifier LR × 2 for first 6 epochs of PH-2
• CE weight = 1.0 (only loss term)
=======================================================================
"""


import os
import yaml
import wandb
import random

import math
import entmax
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from tqdm import tqdm
from copy import deepcopy
from functools import partial
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from trainer.BaseTrainer import BaseTrainer
from neural_extractors.model.PhysMamba import PhysMamba

from REs.MTDE import MTDE

from modules.AttnScorer import AttnScorer           # Returns pre-softmax raw attn score
from modules.GatedPooling import GatedPooling       # No attn inside, takes extrernal attn raw score from AttnScorer
from modules.ProjectionHead import ProjectionHead
from modules.ChunkForward import ChunkForwardModule
from modules.ClassificationHead import ClassificationHead
from modules.ChunkAuxClassifier import ChunkAuxClassifier

from tools.utils import SupConLossTopK, FocalLoss
from tools.utils import reconstruct_sessions, run_tsne_and_plot, straight_through_topk

# ───────────────────────────────────────────────
# TOP-LEVEL CURRICULUM  (50 epochs total)
# Phase-0 :  0-14   ▶ SupCon (Softmax α=1.0)
# Phase-1 : 15-29   ▶ Top-K + Chunk-CE (Focal)
# Phase-2 : 30-49   ▶ Session-CE Fine-tune
# ───────────────────────────────────────────────
PHASE_BOUND = [15, 30]          # do not change elsewhere

class MTDETrainer_BC(BaseTrainer):
    """
    Session-Level Emotion Recognition with 50-Epoch Curriculum
    ==========================================================
    Phase 0 (0-14):  Representation Exploration
        - SupConLossTopK + entropy regularisation
        - Classifier / ChunkAux frozen
    Phase 1 (15-29): Chunk-level Refinement
        - Top-K + Focal Chunk-CE  (+ small session-CE cushion)
    Phase 2 (30-49): Session Fine-tune
        - Only CE loss, MTDE+Pooling+Classifier train
        - AttnScorer/Projection frozen, Classifier LR warm-boost
    Loss = SupCon × w₁  + Focal Chunk-CE × w₂ + Session CE × w₃ + Entropy × λ
    (w₁,w₂,w₃,λ are phase-dependent; see `update_training_state`)
    """
    def __init__(self, config, data_loader):
        super(MTDETrainer_BC, self).__init__()
        self.device = torch.device(config.DEVICE)
        self.config = config
        self.max_epoch = config.TRAIN.EPOCHS
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.warm_up = config.TRAIN.WARMUP_EPOCHS 
        self.optimizer_config = {}
        self.temperature = 1.0
        
        wandb.init(
            project="ReMOTION_Temporal",
            name=f"Exp_Arsl_50epochs",
            # config=cfg_dict,
            dir="./wandb_logs"
        )

        # --------------------------- Encoder Initialization ---------------------------
        """Used to extract physiological signals from chunked video input (frozen during training)"""
        self.extractor = PhysMamba().to(self.device)
        self.extractor = torch.nn.DataParallel(self.extractor)
        pretrained_path = os.path.join("./pretrained_extractors", "UBFC-rPPG_PhysMamba_DiffNormalized.pth")
        self.extractor.load_state_dict(torch.load(pretrained_path, map_location=self.device))

        # Freeze extractor (by default, no fine-tuning)
        for name, param in self.extractor.named_parameters():
            param.requires_grad = False
        """
        # Fine-tuning
        modules_to_finetune = [
            'Block3',           # Slow stream's low-level feature extraction block
            'Block6',           # Fast stream's high-level feature extraction block
            'ConvBlock6',       # Fast stream's last conv (selective)
            'ConvBlock3'        # stem's last conv (selective)
        ]
        for name, param in self.extractor.module.named_parameters():
            if any(key in name for key in modules_to_finetune):
                param.requires_grad = True
            else:
                param.requires_grad = False
        """

        # ---------------- MTDE (Multi-scale Temporal Dynamics Encoder) ----------------
        """Multi-scale 1D CNN + SE blocks for extracting temporal features from rPPG"""
        self.mtde = MTDE(
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
        Trained with session-level GT label per chunk and same structure with ClassificationHead"""
        self.chunk_aux_classifier = ChunkAuxClassifier(
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
            num_classes=config.TRAIN.NUM_CLASSES
        ).to(self.device)
        
        # ------------------------------ Gated Attention Pooling ------------------------------
        """Aggregates chunk embeddings to a session-level embedding using Gated self-attention
           Includes entropy regularization (used in sparsity loss)"""
        self.pooling = GatedPooling(
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
            p_drop=0.1,
            fixed_alpha=1.5      # ← fixed α 
        ).to(self.device)

        # ----------------------------- Classification Head -----------------------------
        """Simple MLP head for predicting emotion labels from pooled session embedding"""
        self.classifier = ClassificationHead(
            input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
            num_classes=config.TRAIN.NUM_CLASSES
        ).to(self.device)
        self.logit_scale = nn.Parameter(torch.tensor(1, device=self.device) * 1.5)

        # ----------------------------- Chunk Forward Module ----------------------------
        self.chunk_forward_module = ChunkForwardModule(
            extractor=self.extractor.module,
            encoder=self.mtde,
            use_checkpoint=False,
            freeze_extractor=True,
        ).to(self.device)

        # -------------------------------- Loss Functions -------------------------------
        """SupConLossTopK: supervised contrastive loss using top-K attended chunks per session"""
        self.contrastive_loss_fn = SupConLossTopK(temperature=0.1)
        
        """FocalLoss: Standard α-balanced focal loss for chunk-level classification (phase1)"""
        self.aux_criterion = FocalLoss(alpha=0.5, gamma=2.0, reduction="none")
        
        """CrossEntropyLoss: standard cross-entropy loss for session-level classification (phase2)"""
        self.criterion = nn.CrossEntropyLoss()


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
    
    def forward_chunk(self, chunk, gate_on=True):
        """
        Forward pass for a single chunk.
        Args:
            chunk (Tensor): Input chunk tensor of shape (1, C, T, H, W)
            gate_on : Phase0(True) / Phase1-2(False)
        Returns:
            emb (Tensor): Output embedding tensor of shape (1, embed_dim)
        """
        return self.chunk_forward_module(chunk, gate_on=gate_on)
    
    def forward_single_chunk_checkpoint(self, chunk, gate_on=True):
        """
        Forward pass a single chunk using the internal setting of use_checkpoint.
        """
        if self.chunk_forward_module.use_checkpoint:
            return checkpoint.checkpoint(self.forward_chunk, chunk, gate_on=gate_on)
        else:
            return self.forward_chunk(chunk, gate_on=gate_on)
        
    # --------------------- Curriculum Phase Switch ---------------------
    def update_training_state(self, epoch: int):
        """
        Switches all hyper-parameters for each curriculum phase.
        Phases: 0-14 | 15-29 | 30-49  (total 50 epochs)
        
        -- Grad-gate per curriculum phase --------------------
        • phase 0  : MTDE + Attn + Projection   (session modules OFF)
        • phase 1a : ChunkAux only              (Top-K supervision)
        • phase 1b : ChunkAux + Pool + Classifier (alignment stage)
        • phase 2  : Pool + Classifier          (final CE fine-tune)
        """
        # -------- Phase-0: SupCon exploration ---------------------------------
        if epoch < PHASE_BOUND[0]:
            phase, base_lr, wd, t_max = 0, 3e-4, 1e-4, 15
            self.contrastive_weight = 0.50
            self.chunk_ce_weight    = 0.0
            self.ce_weight          = 0.0
            self.top_k_ratio        = 0.0

            # ▸ Phase-0 starts with *mild* entropy regularisation (0.05)
            #   and will be updated online after each epoch.
            self.lambda_ent         = 0.05

        # -------- Phase-1: Top-K + Chunk-CE -----------------------------------
        elif epoch < PHASE_BOUND[1]:
            phase = 1
            sub   = epoch - PHASE_BOUND[0]            # 0‥14
            base_lr, wd, t_max = 2e-4, 5e-4, 15

            self.top_k_ratio        = 0.70 - 0.30 * (sub / 14)
            self.chunk_ce_weight    = 0.30 if sub < 4 else 1.0
            self.ce_weight          = 0.20            # constant cushion
            self.contrastive_weight = 0.10
            self.lambda_ent         = 0.0

        # -------- Phase-2: Session-CE fine-tune -------------------------------
        else:
            phase, base_lr, wd, t_max = 2, 1e-4, 1e-4, 20
            self.top_k_ratio = self.contrastive_weight = self.chunk_ce_weight = 0.0
            self.ce_weight   = 1.0
            self.lambda_ent  = 0.0

        # ────────────────── requires_grad scheduler ──────────────────

    
        sub  = epoch - PHASE_BOUND[0]            # 0‥14 in Phase-1
        p1b  = (phase == 1 and sub >= 10)        # alignment window

        def set_grad(module, flag: bool):
            for p in module.parameters():
                p.requires_grad = flag

        # ── MTDE encoder (always trainable) ──────────────────────────
        set_grad(self.mtde, True)

        # ── Attention-Scorer ─────────────────────────────────────────
        set_grad(self.attn_scorer, phase < 2)         # freeze at Phase-2

        # ── Projection head (SupCon only) ───────────────────────────
        set_grad(self.chunk_projection, phase == 0)

        # ── Chunk-level auxiliary classifier ────────────────────────
        set_grad(self.chunk_aux_classifier, phase == 1)

        # ── Gated-Pooling + Session-Classifier ──────────────────────
        set_grad(self.pooling,     p1b or phase == 2)
        set_grad(self.classifier,  p1b or phase == 2)


        # ------------- rebuild optimizer when phase changes -------------------
        if getattr(self, 'cur_phase', -1) != phase:
            self.cur_phase = phase
            frozen = self.configure_optimizer_scheduler(base_lr, wd, t_max, phase, epoch)
        else:
            frozen = []

        # SupCon internal scheduling (pos / neg budget)
        self.contrastive_loss_fn.schedule_params(epoch, self.max_epoch)

        return phase, frozen


    # ------------------ Optimizer / Scheduler builder ------------------------
    def configure_optimizer_scheduler(self, base_lr, weight_decay, t_max,
                                    phase: int, epoch: int):
        """
        Creates AdamW param-groups with per-module LR.
        Classifier receives ×2 LR for the first six epochs of Phase-2.
        """
        # -------- classifier LR warm-up (Phase-2)
        cls_boost = 2.0 if (phase == 2 and (epoch - PHASE_BOUND[1]) < 6) else 1.0
        lr_cls    = base_lr * cls_boost

        pg_decay, pg_nd, frozen = [], [], []

        def add_param(p, name, lr):
            group = pg_nd if 'bias' in name else pg_decay
            group.append({'params': p, 'lr': lr,
                        'weight_decay': 0.0 if 'bias' in name else weight_decay})

        modules = [
            ("MTDE",                 self.mtde,                 base_lr),
            ("AttnScorer",           self.attn_scorer,          base_lr),
            ("Pooling",              self.pooling,              base_lr),
            ("ChunkProj",            self.chunk_projection,     base_lr),
            ("ChunkAux",             self.chunk_aux_classifier, base_lr),
            ("Classifier",           self.classifier,           lr_cls),
        ]
        for tag, module, lr_mod in modules:        # tag = str prefix
            for pname, p in module.named_parameters():
                full_name = f"{tag}.{pname}"        # ← prefix 
                if not p.requires_grad:
                    frozen.append(full_name) 
                    if p.grad is not None:
                        p.grad = None
                    continue
                add_param(p, full_name, lr_mod)
        
        self.optimizer = optim.AdamW(pg_decay + pg_nd, betas=(0.9, 0.95))
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=t_max, T_mult=2, eta_min=5e-6)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=1e-6)
        
        return frozen
    
    # ----------------------------------------------------------
    # helper (already imported in your module header once)
    # from utils.attention_utils import straight_through_topk
    # ----------------------------------------------------------

    def forward_batch(self, batch, epoch, phase):
        """
        Mini-batch forward pass.

        Phase 0 : SupConLossTopK + SparsityLoss
        Phase 1 : Top-K → Focal Chunk-CE  (+ small session-CE cushion)
        Phase 2 : GatedPooling → Session-CE (Classifier warm-up scale)

        Core steps
        ----------
        1) rPPG → MTDE → chunk-embeddings
        2) AttnScorer → raw scores (+ α-entmax soft weights)
        3) Phase-specific losses:
        • SupCon (P0) • Chunk-CE (P1) • Session-CE (P1 cushion & P2 full)
        4) GatedPooling always produces session embedding
        5) Extensive debug print / t-SNE logging
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
            chunks = chunk_tensors[b].float().to(self.device)                                           # (T,1,128)
            T      = chunks.size(0)
            label  = batch_labels[b]
            sid    = batch_sids[b]

            print(f"[DEBUG] Session {sid}  #chunks={T}")

            # 1) Extract rPPG and its embeddings of all chunks at once (T,1,128) ➜ (1,T,D)
            gate_flag = (phase == 0)
            chunk_emb = self.forward_single_chunk_checkpoint(chunks, gate_on=gate_flag)                                    # (T,D)
            chunk_emb = chunk_emb.unsqueeze(0)                                                          # (1,T,D)
            
            # ──────── per-chunk debug & t-SNE ───────────────
            for idx, cemb in enumerate(chunk_emb[0]):                                                   # iterate T chunks
                print(f"  Chunk {idx}: mean={cemb.mean():.4f}, std={cemb.std():.4f}")
                self.chunk_embeddings_for_tsne.append(cemb.detach().cpu().numpy())
                self.chunk_labels_for_tsne.append(label)
            # ────────────────────────────────────────────────

            # 2) attention scorer 
            attn_soft, raw_scaled, st_alpha = self.attn_scorer(
                chunk_emb,                                                                              # (1,T,1)
                temperature=self.temperature,                                                           # τ  (phase-adaptive)
                epoch=epoch,                                                                            # epoch index
            )                    
            print(f"[DEBUG] Session {sid}  #chunks={T}, σ_running={self.attn_scorer.running_score_std.item():.3f}")
            
            # ─── Phase 0: SupCon + Sparsity ───────────────────────
            if phase == 0:      
                # build α (softmax in Phase-0) and entropy
                alpha = attn_soft
                entropy_attn_soft = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=1).mean()
                
                # apply α before projection
                gated_emb   = chunk_emb * alpha                                                         # (1,T,D)
                proj_embeds = self.chunk_projection(gated_emb)                                          # (1,T,128)
                
                # SupCon collection
                proj_for_supcon.append(proj_embeds.squeeze(0))
                labels_for_supcon.append(torch.full((T,), label, dtype=torch.long, device=self.device))
            
            # ─── Phase 1: Chunk-CE with ST-TopK ─────────────────
            # phase1-a,b,c
            elif phase == 1:
                sub = epoch - PHASE_BOUND[0]                                              
                # just in case
                if len(raw_scaled.squeeze(0).squeeze(-1)) < 1:
                    print("Empty attention scores – skip this session")
                    continue
                if st_alpha is None:
                    raise RuntimeError(f"st_alpha=None in Phase1!!! epoch={epoch}")
                
                # Hard-Soft with selected Top-K mask (forward hard, backward α-entmax)
                alpha_mask = straight_through_topk(
                    raw_scaled, k=min(T, max(6, int(T * self.top_k_ratio)))
                    , soft_branch="entmax_alpha", st_alpha=st_alpha
                ) # (1, T)
                
                # for Top-K monitoring
                mask_bool  = alpha_mask.squeeze(0).bool()                                               # (T,)
                print(f"[DEBUG] Top-K selected = {mask_bool.sum().item()} / {T}")
                
                # build α (α-Entmax in Phase-1) and entropy
                alpha = attn_soft                                                                       # (B,T)
                entropy_attn_soft = -(attn_soft * attn_soft.clamp_min(1e-8).log()).sum(dim=1).mean()
                
                # ChunkAuxClassifier & Focal CE ―― only this block tracked
                # projection -> no_grad in phase1-a,b,c
                with torch.no_grad():
                    _ = self.chunk_projection(chunk_emb)
                
                # gated embedding (forward hard, backward soft)                                            
                gated_emb_hard = chunk_emb * alpha_mask.unsqueeze(-1)                                   # (1,T,D)
                gated_emb_soft = chunk_emb * attn_soft                                                  # (1,T,D)
                gated_emb = gated_emb_hard + (gated_emb_soft - gated_emb_soft.detach())                 # (1,T,D)
                
                # Top-K chunk-level logits & per-chunk loss
                logits_all   = self.chunk_aux_classifier(gated_emb.squeeze(0))                          # (T,C)
                lbl_all      = torch.full((T,), label, device=self.device)                              # (T)
                ce_per_chunk = self.aux_criterion(logits_all, lbl_all)                                  # scalar
                
                # weighted sum → scalar chunk CE (backward soft)
                alpha_prob = attn_soft.squeeze(0)                                                      # (T,)
                weighted_ce  = (alpha_prob * ce_per_chunk).sum() / (alpha_prob.sum() + 1e-8)            # ← eps
                chunk_ce_losses.append(weighted_ce)
            
            # ─── Phase 2: GatedPooling + Session-CE ───────────────
            else:
                alpha = torch.ones_like(raw_scaled)                                                     # (1,T,1)
                entropy_attn_soft = 0
                
                with torch.no_grad():
                    # supcon, chunkCE related modules no_grad
                    _ = self.chunk_projection(chunk_emb)
                    _ = self.chunk_aux_classifier(chunk_emb.squeeze(0))

            # 3) session-level pooling
            pooled, attn_gated, ent_gated = self.pooling(
                chunk_emb, raw_scaled, epoch, return_weights=True, return_entropy=True)
            
            # ★──────── session-level t-SNE ───────────────────
            sess_emb = pooled.squeeze(0)                              # (D,)
            self.session_embeddings_for_tsne.append(sess_emb.detach().cpu().numpy())
            self.session_labels_for_tsne.append(label)
            # ────────────────────────────────────────────────

            session_embeds.append(pooled.squeeze(0))
            target_labels.append(label)

            # 4) entropy bookkeeping
            entropy_attn_list.append(entropy_attn_soft)
            entropy_gated_list.append(ent_gated)
            entropy_list.append(entropy_attn_soft if phase < 2 else ent_gated)

            # 5) OPTIONAL chunk-level debug
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

        # Sparsity
        sparsity_loss = torch.stack(entropy_list).mean()

        # Chunk CE (Phase-1)
        chunk_ce_raw = (torch.stack(chunk_ce_losses).mean()
                        if chunk_ce_losses else torch.tensor(0.0, device=self.device))

        # Session CE (always computed)
        sess_mat   = torch.stack(session_embeds, 0)
        label_vec  = torch.tensor(target_labels, dtype=torch.long, device=self.device)
        logits_raw  = self.classifier(sess_mat)
        if phase == 2:
            # gently raise scale within Phase-2
            scale_factor = 1.5 + 0.5 * min((epoch - PHASE_BOUND[1]) / 19, 1.0)
            ce_logits = self.logit_scale * scale_factor * logits_raw
        else:                       
            ce_logits = logits_raw.detach() + logits_raw - logits_raw.detach()
        ce_loss    = self.criterion(ce_logits, label_vec)

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
            sub = epoch - PHASE_BOUND[0]
            if sub < 10:                                            # Phase1-a,b
                loss_total = chunk_ce_loss_scaled
            else:                                                   # Phase1-c (Cushion)
                loss_total = chunk_ce_loss_scaled + ce_loss_scaled 
        else:                             
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
        Curriculum-aware training loop
        --------------------------------
        • Updates loss weights, Top-K ratio, sparsity λ each epoch
        • Dynamically freezes / unfreezes modules per phase
        • Classifier LR × 2 warm-boost during first 6 epochs of Phase-2
        • Logs gradients, entropies, t-SNE plots, and saves best checkpoints
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
            phase, frozen = self.update_training_state(epoch)

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
            print(f"❄️ Frozen Modules:")
            if frozen:
                for module_name in frozen:
                    print(f"   ▸ {module_name}")
            else:
                print("   ▸ [None]")
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
                        # ("Extractor", self.extractor),
                        ("MTDE", self.mtde),
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
            
            # ───────────────────────────────────────────────────
            # Adaptive sparsity weight (Phase-0 only)
            if phase == 0:
                target_H  = 1.10                      # desired entropy (bits)
                gap       = max(0.0, target_H - avg_entropy_attn)
                self.lambda_ent = 0.05 + 0.15 * gap   # 0.05 – 0.20
            # ───────────────────────────────────────────────────
            
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
                "loss/train_ce": self.loss_ce_per_epoch[-1],
                "loss/train_total": avg_loss,
                "loss/valid_total": val_loss,
                
                "loss/train_contrastive": self.loss_contrastive_per_epoch[-1],
                "loss/train_sparsity": self.loss_sparsity_per_epoch[-1],
                "loss/train_chunk_ce": self.loss_chunk_ce_per_epoch[-1],
                
                "acc/valid": metrics["accuracy"],
                "f1/valid": metrics["f1"],
                "acc/train": acc,
                
                "entropy/gated": avg_entropy_gated,
                "entropy/attn": avg_entropy_attn,
                "lambda_ent": self.lambda_ent,
                
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
            #"""
            model_path = os.path.join("./saved_models", f"{base_name}_{tag}.pth")
            if not os.path.exists(model_path):
                print(f"[SKIP] {tag} model not found at {model_path}")
                continue
            #"""
            # model_path = "./saved_models/Arsl_BC_PhysMamba_Normal_TemporalBranch_best_loss.pth"
            print(f"\n==== Testing [{tag}] Model ====")
            self.load_best_model(model_path)
            self.eval_mode()

            all_preds, all_labels = [], []
            losses, entropies = [], []
            
            with torch.no_grad():
                tbar = tqdm(data_loader["test"], desc=f"Test-{tag}", ncols=80)
                """
                if self.config.TEST.DATA.LABEL_COLUMN == 'BC_Vlnc':
                    skip_ids = set()
                    label_one_ids = []
                    for idx, batch in enumerate(data_loader["test"]):
                        session_emb_dict, session_label_dict, _ = self.reconstruct_sessions(
                            batch=batch, idx=idx, epoch=self.max_epoch, phase='test'
                        )
                        for sid, lbl in session_label_dict.items():
                            if lbl == 1:
                                label_one_ids.append(sid)
                    if len(label_one_ids) >= 7:
                        random.seed(42)
                        skip_ids = set(random.sample(label_one_ids, 7))
                    """
                for idx, batch in enumerate(tbar):
                    session_emb_dict, session_label_dict, session_entropies = self.reconstruct_sessions(
                        batch=batch, idx=idx, epoch=self.max_epoch, phase='test'
                    )
                    for sid, emb in session_emb_dict.items():
                        """
                        if self.config.TEST.DATA.LABEL_COLUMN == 'BC_Vlnc':
                            if sid in skip_ids:
                                print(f"[SKIP] Session {sid} with label=1 is excluded from inference.")
                                continue
                        """

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
            # break
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
            "mtde"                  : self.mtde.state_dict(),
            "attn_scorer"           : self.attn_scorer.state_dict(),
            "projection_head"       : self.chunk_projection.state_dict(),
            "chunk_aux_classifier"  : self.chunk_aux_classifier.state_dict(),
            "gated_pooling"         : self.pooling.state_dict(),
            "classifier"            : self.classifier.state_dict(),
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
        This includes MTDE(encoder), AttentionScorer, ChunkProjection, 
        ChunkAuxClassifier, GatedPooling, and Classifier.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.mtde.load_state_dict(checkpoint["mtde"], strict=False)
        self.attn_scorer.load_state_dict(checkpoint["attn_scorer"], strict=False)
        self.chunk_projection.load_state_dict(checkpoint["projection_head"], strict=False)
        self.chunk_aux_classifier.load_state_dict(checkpoint["chunk_aux_classifier"], strict=False)
        self.pooling.load_state_dict(checkpoint["gated_pooling"], strict=False)
        self.classifier.load_state_dict(checkpoint["classifier"], strict=False)
        
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
        # self.extractor.train()
        self.mtde.train()
        self.attn_scorer.train()
        self.chunk_projection.train()
        self.chunk_aux_classifier.train()
        self.pooling.train()
        self.classifier.train()

    def eval_mode(self):
        # self.extractor.eval()
        self.mtde.eval()
        self.attn_scorer.eval()
        self.chunk_projection.eval()
        self.chunk_aux_classifier.eval()
        self.pooling.eval()
        self.classifier.eval()