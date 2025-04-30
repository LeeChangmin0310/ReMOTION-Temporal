import os
import time
import entmax
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from torch.nn.functional import cosine_similarity
from sklearn.manifold import TSNE  # For t-SNE (Feature collapse checking and classification vis)

# ------------------ Utility functions for session reconstruction ------------------
def reconstruct_sessions(self, batch=None, idx=None, epoch=None, phase="train"):
    """
    Processes all chunks per session in a batch and applies attention pooling.
    (Reconstruct session-level embeddings from a batch, chunk by chunk with checkpointing.)
    Args:
        batch (tuple): includes chunk tensors, labels, session ids
    Returns:
        session_embeddings (dict), session_labels (dict)
    """
    chunk_tensors, batch_labels, batch_session_ids, _ = batch
    session_embeddings, session_labels, session_entropies = {}, {}, {}

    for chunks, label, sess_id in zip(chunk_tensors, batch_labels, batch_session_ids):
        # chunks shape: (num_chunks, C, T, H, W)
        num_chunks = chunks.shape[0]
        print(f"[DEBUG] Session {sess_id} -> #chunks = {num_chunks}")
        
        chunk_embeds = []
        for i in range(num_chunks):
            chunk_data = chunks[i].unsqueeze(0).float().to(self.device, non_blocking=True)
            emb = self.forward_single_chunk_checkpoint(chunk_data) # shape (1, embed_dim)
            chunk_embeds.append(emb)

        # cat embeddings
        chunk_embeds = torch.cat(chunk_embeds, dim=0)  # (num_chunks, embed_dim) <- (T, D)
        chunk_embeds = chunk_embeds.unsqueeze(0)       # (1, T, D)
        
        # --- Logging ---
        for i, cemb in enumerate(chunk_embeds[0]):
            print(f"  Chunk {i}: mean={cemb.mean().item():.4f}, std={cemb.std().item():.4f}")
            cemb_np = cemb.detach().cpu().numpy() 
            self.chunk_embeddings_for_tsne.append(cemb_np)
            self.chunk_labels_for_tsne.append(label)

        # === Attention ===
        attn_weights, raw_scores, _ = self.attn_scorer(chunk_embeds, temperature=self.temperature, epoch=epoch)  # (1, T, 1)
        pooled, attn_gated, entropy_gated = self.pooling(chunk_embeds, raw_scores, return_weights=True, return_entropy=True)
        if attn_weights is not None:
            entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=1).mean()
        else:
            attn_weights = attn_gated
            entropy = entropy_gated
            
        attn_np = attn_weights.detach().cpu().squeeze().numpy()
        if phase in ['train', 'valid']:
            if epoch <= 30: # valid, phase0 and phase1
                print(f"[DEBUG][AttnScorer Weights] {attn_np.tolist()}")
                print(f"[DEBUG][AttnScorer Sparsity] mean={attn_np.mean():.4f}, std={attn_np.std():.4f}, entropy={entropy.item():.4f}")
        else: # test and phase 2
            print(f"[DEBUG][GatedPooling Weights] {attn_np.tolist()}")
            print(f"[DEBUG][GatedPooling Sparsity] mean={attn_np.mean():.4f}, std={attn_np.std():.4f}, entropy={entropy.item():.4f}")

        # for t-SNE
        self.session_embeddings_for_tsne.append(pooled.detach().cpu().numpy())
        self.session_labels_for_tsne.append(label)

        session_embeddings[sess_id] = pooled
        session_entropies[sess_id] = entropy
        session_labels[sess_id] = label
    
    return session_embeddings, session_labels, session_entropies


# ------------------ Utility functions for t-SNE visualization ------------------
def run_tsne_and_plot(self, level="chunk", epoch=0, phase="train"):
    """
    Runs t-SNE on either chunk-level or session-level embeddings 
    and saves the plot to a PNG file.
    Args:
        level: "chunk" or "session"
        epoch: current epoch number (for naming and title)
        phase: "train", "valid", or "test"
    """
    if level == "chunk":
        emb_list = self.chunk_embeddings_for_tsne
        lbl_list = self.chunk_labels_for_tsne
        out_dir = f"./chunk_TSNE_{phase}"
        out_name = f"tsne_chunk_level_{phase}_epoch{epoch}.png"
        title_str = f"{phase.capitalize()} Chunk-level t-SNE (Epoch {epoch})"
    else:
        emb_list = self.session_embeddings_for_tsne
        lbl_list = self.session_labels_for_tsne
        out_dir = f"./session_TSNE_{phase}"
        out_name = f"tsne_session_level_{phase}_epoch{epoch}.png"
        title_str = f"{phase.capitalize()} Session-level t-SNE (Epoch {epoch})"

    if len(emb_list) == 0:
        print(f"[TSNE] No embeddings found for {level} level. Skip.")
        return

    os.makedirs(out_dir, exist_ok=True)
    emb_array = np.vstack(emb_list)
    label_array = np.array(lbl_list)

    tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30)
    tsne_results = tsne.fit_transform(emb_array)

    plt.figure(figsize=(6, 6))
    unique_labels = np.unique(label_array)
    for ul in unique_labels:
        idx = np.where(label_array == ul)
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=f"Label {ul}", alpha=0.7)
    plt.title(title_str)
    plt.legend()
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[TSNE] Saved {level}-level t-SNE to {out_path}")

    # Clear after each run
    if level == "chunk":
        self.chunk_embeddings_for_tsne.clear()
        self.chunk_labels_for_tsne.clear()
    else:
        self.session_embeddings_for_tsne.clear()
        self.session_labels_for_tsne.clear()


# --- Utility function for Focal Loss ---
class FocalLoss(nn.Module):
    """
    Class-balanced focal loss.
    Args
    ----
    alpha : weight for the rare class (scalar or 1-D tensor)
    gamma : focusing parameter
    reduction : "none" | "mean" | "sum"
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "none") -> None:
        super().__init__()
        self.alpha     = alpha          
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # CE per-sample
        ce_loss = F.cross_entropy(logits, targets,
                                  reduction="none", label_smoothing=0.0)
        # Convert to probability of correct class
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1. - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:                               # "none"
            return focal

# --- Utility function for straight-through Top-K attention ---
def straight_through_topk(raw_scores: torch.Tensor, k: int, 
                          soft_branch: Literal["softmax", "entmax15", "entmax_alpha"] = "entmax_alpha", 
                          st_alpha: float = 1.9):
    """
    Forward : hard 1/0 Top-K mask
    Backward: soft probs (Entmax or Softmax) for gradient
    Args:
        raw_scores : (B,T,1)  –  logit or pre-score
        k          : Top-K
        soft_fn    : softmax  or  partial(entmax15)
        st_alpha   : straight-through alpha for a-entmax (default: 1.5)
    """
    B, T, _ = raw_scores.shape
    logits  = raw_scores.view(B, T)

    # ---- soft branch (∂L/∂logits) -----------------------------------------
    if soft_branch == "softmax":
        probs = torch.softmax(logits, dim=1)
    elif soft_branch == "entmax15":
        probs = entmax.entmax15(logits, dim=1)
    else:                                   # α-entmax
        probs = entmax.entmax_bisect(logits, alpha=st_alpha, dim=1)


    # ---- hard branch (forward only) ---------------------------------------
    _, top_idx = torch.topk(logits, k=min(k, T), dim=1)
    hard = torch.zeros_like(probs).scatter_(1, top_idx, 1.0)

    # ---- straight-through --------------------------------------------------
    # forward : hard , backward : probs
    return hard + (probs - probs.detach())      # ≡ probs.grad , hard.forward  <- (B,T)


# --- Utility functions for supervised contrastive loss with top-k attention-based chunk selection ---
class SupConLossTopK(nn.Module):
    """
    Supervised Contrastive Loss with Top-K Hard Positive & Negative Sampling

    Highlights:
    - Enforces a fixed max number of positive/negative samples per anchor
    - Dynamically adapts selection based on similarity matrix
    - Ensures a balanced and informative set of pairs for learning

    Parameters:
    - temperature: Scaling factor for similarity logits
    - max_pos: Max number of positive samples per anchor
    - max_neg: Max number of negative samples per anchor
    - pos_neg_ratio: Controls relative number of positives based on available negatives
    """
    def __init__(self, temperature=0.1, max_pos=6, max_neg=30, pos_neg_ratio=1.0):
        super(SupConLossTopK, self).__init__()
        self.temperature = temperature
        self.max_pos = max_pos
        self.max_neg = max_neg
        self.pos_neg_ratio = pos_neg_ratio

    def forward(self, features, labels):
        """
        Args:
            features (Tensor): shape (K, D) - normalized chunk embeddings
            labels (Tensor):   shape (K,)   - corresponding session labels

        Returns:
            Tensor: scalar contrastive loss averaged over anchors
        """
        device = features.device
        if features.size(0) <= 1:
            return torch.tensor(0.0, device=device)

        # Step 1: Normalize and compute similarity matrix
        features = F.normalize(features, dim=1)
        print(f"[DEBUG] norm mean={features.mean().item():.4f}, std={features.std().item():.4f}")
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()  # log-sum-exp stability
        exp_sim = torch.exp(sim_matrix)

        # Step 2: Build raw positive and negative masks (exclude self-pairs)
        mask_pos = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)
        mask_pos.fill_diagonal_(0)  # Avoid self-contrast
        mask_neg = 1.0 - mask_pos - torch.eye(len(labels), device=device)

        # Step 3: Refine masks by top-k sampling for each anchor
        new_mask_pos = torch.zeros_like(mask_pos)
        new_mask_neg = torch.zeros_like(mask_neg)

        for i in range(len(labels)):
            pos_idx = (mask_pos[i] > 0).nonzero(as_tuple=True)[0]  # same class
            neg_idx = (mask_neg[i] > 0).nonzero(as_tuple=True)[0]  # different class
            
            # === Skip anchors with no positives or no negatives ===
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                print(f"[Skip SupCon] anchor {i}: no positive available.")
                continue

            # Adaptive limit: ensure balanced pairs and avoid overload
            max_pos_i = max_pos_i = min(
                len(pos_idx),
                max(1, min(self.max_pos, int(len(neg_idx) * self.pos_neg_ratio)))
            )
            max_neg_i = min(len(neg_idx), max(6, self.max_neg))

            if max_pos_i > 0:
                top_pos = torch.topk(sim_matrix[i][pos_idx], max_pos_i).indices
                new_mask_pos[i][pos_idx[top_pos]] = 1.0

            if max_neg_i > 0:
                # top_neg = torch.topk(sim_matrix[i][neg_idx], max_neg_i, largest=False).indices
                # new_mask_neg[i][neg_idx[top_neg]] = 1.0
                k_neg = min(max_neg_i, len(neg_idx))
                rand_pos = torch.randperm(len(neg_idx), device=device)[:k_neg]  # positions
                new_mask_neg[i][neg_idx[rand_pos]] = 1.0                       # absolute idx

        # Step 4: Compute loss from selected masks
        numerator = (exp_sim * new_mask_pos).sum(dim=1)
        denominator = (exp_sim * (new_mask_pos + new_mask_neg)).sum(dim=1)
        loss = -torch.log((numerator / denominator).clamp(min=1e-8))

        # Debug: Show selected positives/negatives
        print(f"[DEBUG][SupCon] pos per sample: {new_mask_pos.sum(dim=1).tolist()}")
        print(f"[DEBUG][SupCon] neg per sample: {new_mask_neg.sum(dim=1).tolist()}")

        return loss.mean()

    def schedule_params(self, epoch: int, max_epoch: int):
        """
        Curriculum-based positive / negative sampling schedule for SupConLossTopK.
        """
        # ---------- Phase-1  (0-19) : warm-up ----------
        if epoch < 20:                      # 0-19
            self.max_pos        = 6
            self.max_neg        = 24
            self.pos_neg_ratio  = 1.00      # full contrastive

        # ---------- Phase-2  (20-34) : ramp-up ----------
        elif epoch < 35:                    # 20-34
            if   epoch < 25:                # 20-24
                self.max_pos       = 6
                self.max_neg       = 30
                self.pos_neg_ratio = 0.80
            elif epoch < 30:                # 25-29
                self.max_pos       = 8
                self.max_neg       = 40
                self.pos_neg_ratio = 0.65 - 0.025 * (epoch - 25)  # 0.65→0.55
            else:                           # 30-34
                self.max_pos       = 8
                self.max_neg       = 48
                self.pos_neg_ratio = 0.50 - 0.02  * (epoch - 30)  # 0.50→0.40

        # ---------- Phase-3  (≥35) : SupCon off ----------
        else:
            self.max_pos        = 0
            self.max_neg        = 0
            self.pos_neg_ratio  = 0.0       # disable SupCon

# --- Utility function for Consine Similarity loss between two session-level embeddings ---
class AlignCosineLoss(nn.Module):
    """"
    Cosine Similarity Loss for session-level embeddings.
    This loss function computes the cosine similarity between two session-level embeddings
    and returns the negative mean similarity as the loss.
    Args:
        emb1 (Tensor): session-level embedding from gated_attn_pooled
        emb2 (Tensor): session-level embedding from topk_pooled
    Returns:
        consine_sim (Tensor): scalar cosine sim
    """
    def __init__(self):
        super().__init__()

    def forward(self, emb1, emb2):
        # emb1: (B, D), emb2: (B, D)
        emb1 = F.normalize(emb1, dim=-1) # gated is the anchor
        emb2 = F.normalize(emb2.detach(), dim=-1)  # topk is the target
        cos_sim = F.cosine_similarity(emb1, emb2, dim=-1)  # (B,)
        return 1.0 - cos_sim.mean()

# --- Utility function for KL Divergence loss between two distributions of logits---
class AlignKLLoss(nn.Module):
    def __init__(self, temperature=1.0, reduction='batchmean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, logits1, logits2):
        """
        Compute KL divergence between logits1 (trainable) and logits2 (fixed target).
        Args:
            logits1 (Tensor): classifier output from topk_pooled, shape (B, C)
            logits2 (Tensor): classifier output from gated_pooled, shape (B, C)
            temperature (float): temperature scaling
        Returns:
            kl_loss (Tensor): scalar KL loss
        """
        T = self.temperature
        log_p1 = F.log_softmax(logits1 / T, dim=-1)
        p2 = F.softmax(logits2.detach() / T, dim=-1)
        kl = F.kl_div(log_p1, p2, reduction=self.reduction) * (T ** 2)
        return kl



'''
# ----- Utility functions for supervised contrastive loss -----
# This loss is used to enhance the chunk-level embeddings' diversity
# by maximizing the similarity of embeddings from the same session.
# It is designed for weak session-level labels, where all chunks in a session
# are assumed to belong to the same class.
# The loss is computed as the negative log of the ratio of the sum of
# positive pair scores to the sum of all pair scores (excluding self-similarity).
# This is a supervised version of the contrastive loss, where the positive pairs
# are defined by the session-level labels.
class SupConLossTopK_v2(nn.Module):
    """
    Supervised Contrastive Loss with attention-based Top-K and threshold filtering.

    This loss function performs:
    1. Chunk selection based on attention weights (Top-K and thresholded).
    2. Similarity-based positive and negative pair construction.
    3. Safe gradient flow through attention, projection, and encoder modules.

    Args:
        temperature (float): Scaling factor for similarity scores.
        max_total_k (int): Max number of chunks selected per batch (after thresholding).
        threshold (float): Minimum attention score to consider a chunk as valid.
        max_pos (int): Maximum number of positive pairs per anchor.
        max_neg (int): Maximum number of negative pairs per anchor.
        pos_neg_ratio (float): Controls relative number of positives based on available negatives.
    """
    def __init__(self, temperature=0.1, max_total_k=30, threshold=0.01,
                 max_pos=6, max_neg=30, pos_neg_ratio=1.0, max_total_k_per_label=6, min_total_k_per_label=6):
        
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
    
        self.max_total_k = max_total_k
        self.max_total_k_per_label = max_total_k_per_label
        self.min_total_k_per_label = min_total_k_per_label
        
        self.max_pos = max_pos
        self.max_neg = max_neg
        self.pos_neg_ratio = pos_neg_ratio

    def forward(self, features, labels, attn_weights):
        """
        Supervised Contrastive Loss with Label-Aware Attention-Based Top-K Selection and Threshold Fallback.

        This loss function performs contrastive learning by:
        1. Selecting discriminative chunk embeddings based on attention scores.
        2. Ensuring both positive and negative samples are present via label-aware selection.
        3. Constructing similarity matrix and contrastive pairs.
        4. Computing supervised contrastive loss on selected embeddings.

        Args:
            features (Tensor): Shape (N, D), normalized or project-then-normalized chunk embeddings.
            labels (Tensor): Shape (N,), session-level labels per chunk.
            attn_weights (Tensor): Shape (N,), attention scores for each chunk.

        Returns:
            Tensor: Scalar supervised contrastive loss.
        """
        device = features.device
        N = features.size(0)

        if N <= 1:
            print("[Skip SupCon] Not enough chunks in batch.")
            return torch.tensor(0.0, device=device)

        selected_idx_list = []

        # === Step 1: Label-aware Top-K + Threshold Selection ===
        unique_labels = labels.unique()
        for label in unique_labels:
            idx_label = (labels == label).nonzero(as_tuple=True)[0]
            if len(idx_label) <= 1:
                continue  # Need at least 2 for positive pair

            attn_label = attn_weights[idx_label]
            k = min(self.max_total_k_per_label, len(idx_label))
            topk_vals, topk_indices_local = torch.topk(attn_label, k=k)
            
            """
            min_k_local = max(2, min(6, k))
            # git commit -m "AttnScorerGrad with no threshoding"
            # Threshold filtering (soft fallback to Top-K if too strict)
            above_thd = (topk_vals > self.threshold).nonzero(as_tuple=True)[0]
            if len(above_thd)  >= min_k_local:
                final_indices = idx_label[topk_indices_local[above_thd]]
            else:
                final_indices = idx_label[topk_indices_local]

            if final_indices.numel() > 0:
                selected_idx_list.append(final_indices)
            """
        if len(selected_idx_list) == 0:
            print("[Skip SupCon] No chunks passed label-aware Top-K selection.")
            return torch.tensor(0.0, device=device)

        selected_idx = torch.cat(selected_idx_list, dim=0)
        
        selected_labels = labels[selected_idx]
        if selected_labels.unique().numel() <= 1:
            print("[Skip SupCon] Only one unique label in selected samples.")
            return torch.tensor(0.0, device=device)

        # === Step 2: Slice features and labels ===
        selected_features = features[selected_idx]
        selected_labels = labels[selected_idx]

        if selected_features.size(0) <= 1:
            print("[Skip SupCon] Too few features selected.")
            return torch.tensor(0.0, device=device)

        # === Debug: Inspect distribution of selected features ===
        with torch.no_grad():
            selected_norms = F.normalize(selected_features, dim=-1) * 10.0
            print(f"[DEBUG] Selected Norms → mean={selected_norms.mean():.4f}, std={selected_norms.std():.4f}, K={selected_norms.size(0)}")

        # === Step 3: Compute pairwise similarity ===
        feats = F.normalize(selected_features, dim=1)
        sim_matrix = torch.matmul(feats, feats.T) / self.temperature
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()  # for numerical stability
        exp_sim = torch.exp(sim_matrix)

        # === Step 4: Build Positive/Negative Pair Masks ===
        K = selected_labels.size(0)
        label_matrix = selected_labels.unsqueeze(0) == selected_labels.unsqueeze(1)  # (K, K)
        mask_pos = label_matrix.float() - torch.eye(K, device=device)  # remove self-comparisons
        mask_neg = 1.0 - label_matrix.float()

        # Initialize Top-K pairwise masks
        new_mask_pos = torch.zeros_like(mask_pos)
        new_mask_neg = torch.zeros_like(mask_neg)

        for i in range(K):
            pos_idx = (mask_pos[i] > 0).nonzero(as_tuple=True)[0]
            neg_idx = (mask_neg[i] > 0).nonzero(as_tuple=True)[0]

            max_pos_i = min(self.max_pos, int(len(neg_idx) * self.pos_neg_ratio), len(pos_idx))
            max_neg_i = min(self.max_neg, len(neg_idx))

            if max_pos_i > 0:
                top_pos = torch.topk(sim_matrix[i][pos_idx], max_pos_i).indices
                new_mask_pos[i][pos_idx[top_pos]] = 1.0

            if max_neg_i > 0:
                top_neg = torch.topk(sim_matrix[i][neg_idx], max_neg_i, largest=False).indices
                new_mask_neg[i][neg_idx[top_neg]] = 1.0

        # === Step 5: Compute Supervised Contrastive Loss ===
        numerator = (exp_sim * new_mask_pos).sum(dim=1)
        denominator = (exp_sim * (new_mask_pos + new_mask_neg)).sum(dim=1)
        loss = -torch.log((numerator / denominator).clamp(min=1e-8))

        # === Debug: Pair statistics ===
        print(f"[DEBUG][SupCon] Positive count per anchor: {new_mask_pos.sum(dim=1).tolist()}")
        print(f"[DEBUG][SupCon] Negative count per anchor: {new_mask_neg.sum(dim=1).tolist()}")

        return loss.mean()


    def schedule_params(self, epoch, max_epoch):
        """
        Optionally schedule positive/negative limits and ratios dynamically during training.

        Args:
            epoch (int): current epoch
            max_epoch (int): total number of training epochs
        """
        self.max_pos = min(8, 4 + epoch // 10)
        self.max_neg = min(40, 24 + epoch // 5)
        self.pos_neg_ratio = max(0.3, 1.0 - (epoch / max_epoch) * 0.7)
        # self.threshold = max(0.005, min(0.05, epoch / max_epoch * 0.05))
        self.max_total_k_per_label = max(6, min(20, int(10 + (10 * self.pos_neg_ratio))))
class SupConLoss_v0(nn.Module):
    """
    Supervised Contrastive Loss (for weak session-level labels).
    Designed for chunk-level embeddings grouped by session.
    """

    def __init__(self, temperature=0.1):
        """
        Args:
            temperature (float): Scaling factor for cosine similarity.
        """
        super(SupConLoss_v0, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (N, D) float Tensor - chunk-level embeddings (normalized).
            labels: (N,) long Tensor - session-level labels (same for all chunks in a session).
        
        Returns:
            Scalar loss (float)
        """
        device = embeddings.device
        N = embeddings.size(0)

        # Normalize to unit hypersphere (cosine similarity)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # (N, D)

        # Compute pairwise similarity (cosine)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature  # (N, N)
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()  # for numerical stability
        exp_sim = torch.exp(sim_matrix)

        # Mask: same class -> 1, else 0
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (N, N)

        # Remove self-contrast (diagonal)
        self_mask = torch.eye(N, device=device)
        mask = mask - self_mask

        # Numerator: sum over positives
        numerator = (exp_sim * mask).sum(dim=1)

        # Denominator: sum over all except self
        denominator = exp_sim.sum(dim=1) - torch.exp(torch.diag(sim_matrix))

        # Compute log-ratio loss
        loss = -torch.log((numerator / denominator).clamp(min=1e-8))
        return loss.mean()

# ----- Utility functions for Contrastive Loss that enhence chunk-level diversity -----
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        """
        embeddings: (N, D), where each row is an embedding
        Computes contrastive loss among all pairs (instance-level)
        """
        device = embeddings.device
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature # (N, D) → (N, N) similarity matrix
        sim = sim - torch.max(sim, dim=1, keepdim=True)[0] # numerical stability
        exp_sim = torch.exp(sim) # exponentiate
        mask = ~torch.eye(embeddings.size(0), device=device).bool()         # remove diagonal (self-similarity)


        # positive pair scores (assumes all off-diagonal pairs are positives in current setup)
        pos = exp_sim[mask].view(embeddings.size(0), -1).sum(dim=1)
        denom = exp_sim.sum(dim=1) - torch.exp(torch.diag(sim)) # total similarity minus self

        # avoid division by zero and log(0)
        frac = (pos / denom).clamp(min=1e-8)
        loss = -torch.log(frac).mean() # final loss

        return loss
    



# ----- Utility functions for normalizing rPPG signals from pretrained model -----
# These functions are used to normalize the rPPG signals output by the pretrained model.
# The normalization is done per chunk, i.e., the mean and variance are computed per chunk.
# This is necessary to ensure that the signals are comparable across different sessions.

def normalize_rppg(rppg):
    """
    Normalizes the recovered rPPG signal per chunk to zero mean and unit variance.
    Args:
      rppg: Tensor of shape (batch, signal_length)
    Returns:
      normalized_rppg: Tensor of the same shape
    """
    mean = rppg.mean(dim=1, keepdim=True)
    std = rppg.std(dim=1, keepdim=True)
    zscore = (rppg - mean) / (std + 1e-6)
    return zscore


# ---------------- Utility function for session-level aggregation ----------------
# This function is used to aggregate the chunk-level embeddings to session-level embeddings.
# The chunk-level embeddings are first grouped by subject and ordered by chunk id.
# Then, the embeddings are aggregated via mean pooling to produce a session-level embedding.

def aggregate_session_embeddings(fused_embeddings, subject_ids, chunk_ids, pooling_module, return_weights=False):
    """
    Aggregate fused embeddings (after AttentionFusion) to session-level using AttentionPooling.

    Args:
        fused_embeddings: Tensor of shape (N_chunks, fusion_dim)
        subject_ids: List of session ids
        chunk_ids: List of chunk ids (used for sorting)
        pooling_module: AttentionPooling module

    Returns:
        Dict mapping session_id to session-level embedding (after pooling)
    """
    aggregated = {}
    for emb, subj, cid in zip(fused_embeddings, subject_ids, chunk_ids):
        if subj not in aggregated:
            aggregated[subj] = []
        aggregated[subj].append((int(cid), emb))

    session_embeddings = {}
    session_attn = {} if return_weights else None

    for subj, chunks in aggregated.items():
        # print(f"[DEBUG] Session {subj} -> unsorted_ids={[c[0] for c in chunks]}, #chunks={len(chunks)}")
        # Sort by chunk id
        sorted_chunks = sorted(chunks, key=lambda x: x[0])
        # print(f"[DEBUG] Session {subj} -> sorted_ids={[c[0] for c in sorted_chunks]}, #chunks={len(sorted_chunks)}")

        emb_list = [emb for _, emb in sorted_chunks]

        if not emb_list:
            print(f"[DEBUG] No chunks for subject {subj}")
            continue
        try:
            chunk_tensor = torch.stack(emb_list, dim=0).to(next(pooling_module.parameters()).device)  # (N_chunks, fusion_dim)
            if return_weights:
                pooled, attn_weights = pooling_module(chunk_tensor, return_weights=True)  # → (fusion_dim,)
                session_attn[subj] = attn_weights.detach().cpu()
            else:
                pooled = pooling_module(chunk_tensor, return_weights=False)

            session_embeddings[subj] = pooled
        except Exception as e:
            print(f"[DEBUG] Error aggregating subject {subj}: {e}")
    if return_weights:
        return session_embeddings, session_attn
    else:
        return session_embeddings



'''
def aggregate_session_embeddings(chunk_embeddings, subject_ids, chunk_ids):
    """
    Groups chunk embeddings by subject and orders them by chunk id,
    then aggregates them via mean pooling to produce a session-level embedding.
    """
    aggregated = {}
    for emb, subj, cid in zip(chunk_embeddings, subject_ids, chunk_ids):
        if subj not in aggregated:
            aggregated[subj] = {}
        aggregated[subj][int(cid)] = emb
    aggregated_embeddings = {}
    for subj, chunks in aggregated.items():
        sorted_keys = sorted(chunks.keys())
        session_tensor = torch.stack([chunks[k] for k in sorted_keys], dim=0)  # (num_chunks, embedding_dim)
        aggregated_embeddings[subj] = session_tensor.mean(dim=0)  # Mean pooling per session
    return aggregated_embeddings
'''

# ------------------- Utility functions for signal processing -------------------
                # Non-differentiable signal processing functions

def bandpass_filter(signal, fs=30, lowcut=0.04, highcut=2.5, order=2):
    """
    Apply a Butterworth bandpass filter to the 1D numpy signal.
    Args:
        signal (np.ndarray): 1D signal array
        fs (float): sampling frequency (default 30Hz for rPPG)
        lowcut (float): low cutoff frequency
        highcut (float): high cutoff frequency
        order (int): order of the Butterworth filter
    Returns:
        filtered_signal (np.ndarray): bandpass filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def exponential_moving_standardization(signal, alpha=0.99):
    """
    Exponential Moving Standardization (EMS).
    Args:
        signal (np.ndarray): 1D signal array
        alpha (float): smoothing factor (0 < alpha < 1)
    Returns:
        standardized_signal (np.ndarray): EMS-applied signal
    """
    # Initialize mean and var
    mean_ema = 0.0
    var_ema = 0.0
    eps = 1e-8

    out = np.zeros_like(signal)
    for i, x in enumerate(signal):
        # Update exponential moving mean
        mean_ema = alpha * mean_ema + (1 - alpha) * x
        # Update exponential moving variance
        diff = (x - mean_ema) ** 2
        var_ema = alpha * var_ema + (1 - alpha) * diff
        # Standardize
        out[i] = (x - mean_ema) / (np.sqrt(var_ema) + eps)
    return out
# --------------------------------------------------------------------------------

# -------------------- PyTorch version of the above functions --------------------
                    # Differentiable signal processing functions

class DifferentiableBandpassFilter(nn.Module):
    def __init__(self, fs=30, lowcut=0.04, highcut=2.5, kernel_size=51):
        super(DifferentiableBandpassFilter, self).__init__()
        from scipy.signal import firwin
        nyq = 0.5 * fs
        taps = firwin(kernel_size, [lowcut / nyq, highcut / nyq], pass_zero=False)
        self.register_buffer('kernel', torch.tensor(taps, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.padding = kernel_size // 2

    def forward(self, signal):
        if signal.dim() == 2:  # [B, T]
            signal = signal.unsqueeze(1)
        filtered = nn.functional.conv1d(signal, self.kernel, padding=self.padding)
        return filtered.squeeze(1)

def differentiable_ems(signal, alpha=0.99, eps=1e-8):
    # signal: [B, T]
    B, T = signal.shape
    mean_ema = torch.zeros(B, device=signal.device)
    var_ema = torch.zeros(B, device=signal.device)
    outputs = []
    for t in range(T):
        x = signal[:, t]
        mean_ema = alpha * mean_ema + (1 - alpha) * x
        diff = (x - mean_ema) ** 2
        var_ema = alpha * var_ema + (1 - alpha) * diff
        outputs.append((x - mean_ema) / (torch.sqrt(var_ema) + eps))
    return torch.stack(outputs, dim=1)
# ---------------------------------------------------------------------------------

# ------- Utility functions for visualization and debugging of rPPG signals -------
def visualize_rPPG(rPPG_tensor, sample_index=0, save_dir="./PPGviz", filename_prefix="rPPG_debug"):
    """
    Visualize rPPG signal from PhysMamba output, shape: [B, T], and save to a file.
    
    Parameters:
        rPPG_tensor (Tensor): rPPG output tensor of shape [B, T].
        sample_index (int): which sample to visualize.
        save_dir (str): directory to save the plot.
        filename_prefix (str): prefix for the saved filename.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    rPPG_sample = rPPG_tensor[sample_index].detach().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.plot(rPPG_sample)
    plt.title("Debug: rPPG Signal from PhysMamba")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    
    # Generate a unique filename (e.g., using current time stamp)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(save_dir, f"{filename_prefix}_{timestr}.png")
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved rPPG visualization at: {save_path}")

i=0
def visualize_rPPG_frequency(rppg_signal, fs=30, session_id="unknown", save_dir="./PPGFreqviz_392 and 524", filename_prefix="rPPG_freq"):
    """
    Visualize and save the FFT spectrum and spectrogram from rPPG signal(1D numpy array).
    Parameters:
        rppg_signal (np.ndarray): aggregate rPPG signal (1D array).
        fs (int): sampling frequency of the signal.
        session_id (str): session identifier for the visualization.
        save_dir (str): directory to save the plots.
        filename_prefix (str): prefix for the saved filenames.
    """
    global i
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # FFT calculation
    fft_vals = np.fft.rfft(rppg_signal)
    fft_freq = np.fft.rfftfreq(len(rppg_signal), d=1/fs)
    amplitude = np.abs(fft_vals)
    
    plt.figure(figsize=(10, 4))
    plt.plot(fft_freq, amplitude)
    plt.title(f"Session {session_id} rPPG FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    fft_save_path = os.path.join(save_dir, f"{filename_prefix}_session{session_id}_fft_{i}.png")
    plt.savefig(fft_save_path, dpi=300)
    plt.close()
    print(f"Saved FFT spectrum for session {session_id} at {fft_save_path}")

    # Spectrogram calculation
    plt.figure(figsize=(10, 4))
    # specgram returns: spectrum, frequencies, times, and im
    spectrum, freqs, times, im = plt.specgram(rppg_signal, Fs=fs, NFFT=128, noverlap=64, cmap='jet')
    plt.title(f"Session {session_id} rPPG Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label='Intensity (dB)')
    spec_save_path = os.path.join(save_dir, f"{filename_prefix}_session{session_id}_spec_{i}.png")
    plt.savefig(spec_save_path, dpi=300)
    plt.close()
    print(f"Saved spectrogram for session {session_id} at {spec_save_path}")
    i+=1
'''