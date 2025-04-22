import os
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
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.manifold import TSNE  # For t-SNE (Feature collapse checking and classification vis)

from trainer.BaseTrainer import BaseTrainer
from neural_encoders.model.PhysMamba import PhysMamba

from decoders.TemporalBranch import TemporalBranch # 1DCNN Only ver.
from modules.ChunkForward import ChunkForwardModule
from modules.AttentionPooling import AttentionPooling
from modules.ClassificationHead import ClassificationHead
# from modules.simple_classifier import SimpleClassifier

from tools.utils import normalize_rppg  # , aggregate_session_embeddings, ContrastiveLoss
        
class TemporalBranchTrainer_BC(BaseTrainer):
    """
    Trainer that:
    1) Processes each batch containing multiple sessions
    2) For each session in the batch, we retrieve its chunks 
       and pass them chunk-by-chunk through PhysMamba + TemporalBranch, 
       using gradient checkpointing for each chunk forward pass.
    3) Then we apply AttentionPooling on the chunk embeddings -> session embedding
    4) Classifier -> Loss -> Backward
    5) By doing so, we handle memory more effectively (though compute is heavier),
       and ensure gradient flows from PhysMamba to Classifier properly.
    """
    # def __init__(self, encoder, temporal_branch, pooling, classifier, normalize_fn, config):
    def __init__(self, config, data_loader):
        super(TemporalBranchTrainer_BC, self).__init__()
        self.device = torch.device(config.DEVICE)
        self.config = config
        self.max_epoch = config.TRAIN.EPOCHS
        self.batch_size = config.TRAIN.BATCH_SIZE
        
        # ---------------------- Encoder Initialization ----------------------
        self.encoder = PhysMamba().to(self.device)
        self.encoder = torch.nn.DataParallel(self.encoder)
        pretrained_path = os.path.join("./pretrained_encoders", "UBFC-rPPG_PhysMamba_DiffNormalized.pth")
        self.encoder.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        # Frozen Encoder
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        '''
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
        '''
                
        # ---------------------- Temporal-branch Decoder ----------------------
        self.temporal_branch = TemporalBranch(embedding_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM).to(self.device)
        
        # ---------------------- Z-Normalize for recoverd rPPG ----------------------
        self.normalize_fn = normalize_rppg
        
        # ---------------------- Attention-based Pooling ----------------------
        self.pooling = AttentionPooling(input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
                                        projected_dim=256).to(self.device)
        self.sparsity_weight = 5e-3
        
        # ------------------------- Classification Head -------------------------
        self.classifier = ClassificationHead(input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM, 
                                             num_classes=config.TRAIN.NUM_CLASSES).to(self.device)        

        # --------------- Chunk forward module for checkpoint usage ---------------
        self.chunk_forward_module = ChunkForwardModule(encoder=self.encoder.module,
                                                        temporal_branch=self.temporal_branch,
                                                        use_checkpoint=False,         # checkpoint flag
                                                        freeze_encoder=True          # encoder freeze flag
                                                    ).to(self.device)

        # -------------------------- Loss Function --------------------------
        self.criterion = nn.CrossEntropyLoss()
        
        # ---------------------------- Optimizer ----------------------------
        '''
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.temporal_branch.parameters()) +
            list(self.pooling.parameters()) +
            list(self.classifier.parameters()),
            lr=self.config.TRAIN.LR
        )
        '''
        self.decay = 5e-4
        param_groups = {'decay': [], 'no_decay': []}
        for module in [self.encoder.module, self.temporal_branch, self.pooling, self.classifier]:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                wd = 0.0 if ("bias" in name or "LayerNorm" in name) else self.decay
                param_groups['no_decay' if wd == 0.0 else 'decay'].append({"params": param, "weight_decay": wd})

        self.optimizer = optim.AdamW(param_groups['decay'] + param_groups['no_decay'], lr=config.TRAIN.LR)
        
        # --------------------- Scheduler Initialization ---------------------
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6
        )
        
        # --------------------- Logging and t-SNE Initialization ---------------------
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []

        self.chunk_embeddings_for_tsne = []
        self.chunk_labels_for_tsne = []
        self.session_embeddings_for_tsne = []
        self.session_labels_for_tsne = []


    def forward_chunk(self, chunk):
        return self.chunk_forward_module(chunk)
    
    def forward_single_chunk_checkpoint(self, chunk):
        """
        Forward pass a single chunk using the internal setting of use_checkpoint.
        """
        if self.chunk_forward_module.use_checkpoint:
            return checkpoint.checkpoint(self.forward_chunk, chunk)
        else:
            return self.forward_chunk(chunk)


    def reconstruct_sessions(self, batch, idx):
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
                chunk_data.requires_grad_()
                emb = self.forward_single_chunk_checkpoint(chunk_data) # shape (1, embed_dim)
                chunk_embeds.append(emb)
                # print("[CHECK] Chunk emb grad_fn:", emb.grad_fn)
                # print("[CHECK] Chunk emb requires_grad:", emb.requires_grad)

            # cat embeddings
            chunk_embeds = torch.cat(chunk_embeds, dim=0)  # (num_chunks, embed_dim) <- (T, D)
            chunk_embeds = chunk_embeds.unsqueeze(0)       # (1, T, D)

            # print("[DEBUG] chunk_embeds shape:", chunk_embeds.shape)
            # print(f"[DEBUG] requires_grad per chunk: {[e.requires_grad for e in chunk_embeds]}")
            # print("[DEBUG][Chunk Embedding] mean/std:")
            
            # optional debugging
            for i, cemb in enumerate(chunk_embeds[0]):
                print(f"  Chunk {i}: mean={cemb.mean().item():.4f}, std={cemb.std().item():.4f}")
                cemb_np = cemb.detach().cpu().numpy() 
                self.chunk_embeddings_for_tsne.append(cemb_np)
                self.chunk_labels_for_tsne.append(label)

            # attention pooling
            sess_emb, attn_weights, entropy = self.pooling(chunk_embeds, return_weights=True, return_entropy=True)
            # print(f"sess_emb.requires_grad = {sess_emb.requires_grad}")
            # print(f"sess_emb grad_fn = {sess_emb.grad_fn}")

            attn_np = attn_weights.detach().cpu().squeeze().numpy()
            print(f"[DEBUG][Attention Weights] {attn_np.tolist()}")
            print(f"[DEBUG][Attn Sparsity] mean={attn_np.mean():.4f}, std={attn_np.std():.4f}, entropy={entropy.item():.4f}")

            # for t-SNE
            self.session_embeddings_for_tsne.append(sess_emb.detach().cpu().numpy())
            self.session_labels_for_tsne.append(label)

            session_embeddings[sess_id] = sess_emb
            session_entropies[sess_id] = entropy
            session_labels[sess_id] = label
        
        return session_embeddings, session_labels, session_entropies

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
            
    def train(self, data_loader):
        """
        Training loop for session-multi-batch without gradient accumulation.
        """
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.max_epoch):
            print(f"\n==== Training Epoch: {epoch} ====")
            tbar = tqdm(data_loader["train"], desc=f"Epoch {epoch}", ncols=80)

            self.train_mode()
            running_loss = 0.0
            batch_losses = []
            entropy_terms = []

            self.optimizer.zero_grad()

            for idx, batch in enumerate(tbar):
                self.optimizer.zero_grad()
                session_emb_dict, session_label_dict, session_entropies = self.reconstruct_sessions(batch, idx)

                total_losses = []
                for sess_id, emb in session_emb_dict.items():
                    label = session_label_dict[sess_id]
                    label_tensor = torch.tensor([label], dtype=torch.long, device=self.device)
                    outputs = self.classifier(emb)
                    # print(f"[CHECK] outputs.requires_grad: {outputs.requires_grad}")
                    # print(f"[CHECK] outputs grad_fn: {outputs.grad_fn}")

                    ce_loss = self.criterion(outputs, label_tensor)
                    sparsity_loss = self.sparsity_weight * session_entropies[sess_id]
                    loss = ce_loss + sparsity_loss
                    
                    # --- Debugging: Print classifier outputs ---
                    probs = torch.softmax(outputs, dim=1)
                    print(f"[DEBUG] GT label: {label}, Probs after Softmax: {probs}")
                    print(f"[DEBUG] Session {sess_id} - CE Loss: {ce_loss.item():.4f}, Sparsity Loss: {sparsity_loss.item():.4f}")
                    # --- End Debugging ---

                    total_losses.append(loss)
                    running_loss += loss.item()
                    batch_losses.append(loss.item())
                    entropy_terms.append(session_entropies[sess_id].item())

                total_loss = torch.stack(total_losses).mean()
                total_loss.backward()
                '''
                # --- DEBUGGING ---
                for name, param in self.encoder.named_parameters():
                    if param.requires_grad:
                        print(f"{name}: grad is None? {param.grad is None}")
                for name, param in self.temporal_branch.named_parameters():
                    if param.requires_grad:
                        print(f"{name}: grad is None? {param.grad is None}")
                for name, param in self.pooling.named_parameters():
                    if param.requires_grad:
                        print(f"{name}: grad is None? {param.grad is None}")
                for name, param in self.classifier.named_parameters():
                    if param.requires_grad:
                        print(f"{name}: grad is None? {param.grad is None}")
                '''
                self.optimizer.step()
                self.scheduler.step()
                # self.optimizer.zero_grad()
                
                # Optionally perform gradient checking every (100 * accumulation_steps) iterations
                if idx % 30 == 0:
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
                    for name, param in self.pooling.named_parameters():
                        # print(name, param.requires_grad)
                        if param.grad is not None:
                            print(f"[GradCheck][AttentionPooling] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Classifier] {name} has no grad")
                    for name, param in self.classifier.named_parameters():
                        # print(name, param.requires_grad)
                        if param.grad is not None:
                            print(f"[GradCheck][Classifier] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Classifier] {name} has no grad")
                
                avg_loss = running_loss / (idx + 1)
                tbar.set_postfix(loss=f"{avg_loss:.3f}", entropy=f"{np.mean(entropy_terms):.4f}")

            torch.cuda.empty_cache()
            avg_train_loss = np.mean(batch_losses)
            self.train_losses.append(avg_train_loss)
            print(f"Epoch {epoch}: Average Train Loss = {avg_train_loss:.4f}")

            self.run_tsne_and_plot(level="chunk", epoch=epoch, phase="train")
            self.run_tsne_and_plot(level="session", epoch=epoch, phase="train")

            val_loss, metrics = self.valid(data_loader, epoch=epoch)
            self.val_losses.append(val_loss)
            print(f"Epoch {epoch} - Valid Loss: {val_loss:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            print("Confusion Matrix:\n", metrics["confusion_matrix"])

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model(epoch)
                print(f"[SAVE] Best model updated at epoch {epoch}")

        self.plot_losses_and_lrs()
    
    def valid(self, data_loader, epoch=0):
        self.eval_mode()
        all_preds, all_labels, losses, entropies = [], [], [], []
        
        print("\n==== Validating ====")
        vbar = tqdm(data_loader["valid"], desc="Valid", ncols=80)
        with torch.no_grad():
            for idx, batch in enumerate(vbar):
                session_emb_dict, session_label_dict, session_entropies = self.reconstruct_sessions(batch, idx)
                
                for sid, emb in session_emb_dict.items():
                    lbl = session_label_dict[sid]
                    lbl_tensor = torch.tensor([lbl], dtype=torch.long, device=self.device)
                    # input_vec = emb.unsqueeze(0)
                    outputs = self.classifier(emb)

                    loss = self.criterion(outputs, lbl_tensor)
                    losses.append(loss.item())
                    entropies.append(session_entropies[sid].item())

                    preds = torch.argmax(outputs, dim=1)
                    # --- Debugging: Print Validation outputs ---
                    print(f"[DEBUG] GT label: {lbl}, Softmax probabilities: {preds.cpu().numpy()}")
                    print(f"[DEBUG] Predicted label: {preds.item()}")
                    # --- End Debugging ---
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(lbl_tensor.cpu().numpy())

                avg_loss = np.mean(losses)
                vbar.set_postfix(loss=f"{avg_loss:.3f}")

        self.run_tsne_and_plot(level="chunk", epoch=epoch, phase="valid")
        self.run_tsne_and_plot(level="session", epoch=epoch, phase="valid")
        
        avg_loss = np.mean(losses)
        print(f"[VALID] Avg Entropy: {np.mean(entropies):.4f}")
        print(f"[VALID] Avg Loss: {avg_loss:.4f}")
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        conf_mat = confusion_matrix(all_labels, all_preds)

        return avg_loss, {"accuracy":acc, "f1":f1, "confusion_matrix":conf_mat}

    def test(self, data_loader, epoch=0):
        """Test loop for session-level data."""
        self.eval_mode()
        all_preds, all_labels = [], []

        print("\n==== Testing ====")
        tbar = tqdm(data_loader["test"], desc="Testing", ncols=80)
        with torch.no_grad():
            for idx, batch in enumerate(tbar):
                session_emb_dict, session_label_dict, _ = self.reconstruct_sessions(batch, idx)
                for sid, emb in session_emb_dict.items():
                    lbl = session_label_dict[sid]
                    lbl_tensor = torch.tensor([lbl], dtype=torch.long, device=self.device)

                    # input_vec = emb.unsqueeze(0)
                    outputs = self.classifier(emb)
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(lbl_tensor.cpu().numpy())
        self.run_tsne_and_plot(level="chunk", epoch=epoch, phase="test")
        self.run_tsne_and_plot(level="session", epoch=epoch, phase="test")
        
        acc = accuracy_score(all_labels, all_preds)
        f1_ = f1_score(all_labels, all_preds)
        conf_mat = confusion_matrix(all_labels, all_preds)
        
        print("[Test] Acc: {:.4f}, F1: {:.4f}".format(acc, f1_))
        print("Confusion Matrix:", conf_mat)
        return {"accuracy":acc, "f1":f1_, "confusion_matrix":conf_mat}

    def save_model(self, epoch):
        """Saves the best classifier model."""
        model_save_dir = "./saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        split_method = self.config.TRAIN.DATA.SPLIT_METHOD
        current_subject = self.config.TRAIN.DATA.get('current_subject','all')
        fold_index = self.config.TRAIN.DATA.get('FOLD_INDEX','NA')
        filename = f"{self.config.TRAIN.DATA.DATASET}_{split_method}_subj{current_subject}_fold{fold_index}_epoch{epoch}.pth"
        path = os.path.join(model_save_dir, filename)
        torch.save(self.classifier.state_dict(), path)
        print("[SAVE] Best model at:", path)

    def plot_losses_and_lrs(self):
        """Plots and saves training and validation loss curves."""
        output_dir = os.path.join(self.config.LOG.PATH, self.config.TRAIN.DATA.EXP_DATA_NAME, 'plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        epochs = range(len(self.train_losses))
        
        plt.figure(figsize=(10,6))
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        
        loss_plot_path = os.path.join(output_dir, "loss_plot.pdf")
        
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()
        
        print("[PLOT] Saved loss plot at:", loss_plot_path)

    def train_mode(self):
        # self.encoder.train() 
        self.temporal_branch.train()
        self.pooling.train()
        self.classifier.train()

    def eval_mode(self):
        # self.encoder.eval()
        self.temporal_branch.eval()
        self.pooling.eval()
        self.classifier.eval()