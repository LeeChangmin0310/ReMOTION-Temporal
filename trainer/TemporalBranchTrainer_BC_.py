import os
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.manifold import TSNE  # For t-SNE (Feature collapse checking and classification vis)

from trainer.BaseTrainer import BaseTrainer
from neural_encoders.model.PhysMamba import PhysMamba

from decoders.TemporalBranch import TemporalBranch
from modules.AttentionPooling import AttentionPooling
# from modules.simple_classifier import SimpleClassifier
from modules.ClassificationHead import ClassificationHead

from tools.utils import normalize_rppg, aggregate_session_embeddings, ContrastiveLoss

class TemporalBranchTrainer_BC(BaseTrainer):
    """
    Trainer class for the Temporal branch with Attention pooling and Classification head.
    This trainer is designed for session-level data processing.
    Initialize the Emotion Trainer.
        Input: rPPG signal chunk (B, T, 1)
        ↓
        [TemporalBranch]
        ├── 1D CNN (Conv → BN → ReLU → MaxPool ×2)
        │     ⤷ Low-level temporal features (B, C, T//4)
        ├── LSTM (Bidirectional, 2-layer)
        │     ⤷ Temporal dynamics across time (B, T//4, 2H)
        ├── Self-Attention Pooling
        │     ⤷ Focused summary of time sequence → (B, 2H)
        └── FC + ReLU + Dropout + LayerNorm
            ⤷ Final chunk-level embedding (B, D=embedding_dim)
        ↓
        [Multiple chunks → stacked into] → (1, N_chunks, D)
        ↓
        [AttentionPooling]
        └── Chunk-wise attention scores (1, N_chunks, 1)
        └── Softmax → Attention weights
        └── Weighted sum → Session embedding (1, D)
        ↓
        Classifier for emotion prediction
        
        Input: chunk-level r    PPG (B, T=128, 1)
        ↓
        [TemporalBranch]
            CNN(1D) + Dropout + ReLU + AvgPool
            LSTM (Bi) + LayerNorm
            (Optional) Self-Attention → pooled feature
            FC + ReLU + LayerNorm
        → Chunk Embedding (B, D)
        ↓
        [Session Aggregation]
            Stack N_chunks → (1, N, D)
        ↓
        [AttentionPooling]
            Attention weights → Weighted sum
        ↓
        Classifier → Emotion label


    """
    def __init__(self, config, data_loader):
        super(TemporalBranchTrainer_BC, self).__init__()
        self.device = torch.device(config.DEVICE)
        self.config = config
        self.max_epoch = config.TRAIN.EPOCHS
        self.batch_size = config.TRAIN.BATCH_SIZE

        # Initialize the Frozen rPPG Encoder (PhysMamba)
        self.encoder = PhysMamba().to(self.device)
        self.encoder = torch.nn.DataParallel(self.encoder)
        pretrained_path = os.path.join("./pretrained_encoders", "UBFC-rPPG_PhysMamba_DiffNormalized.pth")
        self.encoder.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        # self.encoder.module.forward = self.encoder.module.forward_partial
        
        target_blocks = ["Block5","ConvBlockLast"]
        for name, param in self.encoder.module.named_parameters():
            if any(k in name for k in target_blocks):
                param.requires_grad = True
            else:
                param.requires_grad = False
        '''
        for param in self.encoder.module.parameters():
            param.requires_grad = False
        '''
        
        # Initialize the Temporal branch, Attention pooling, and Classification head
        self.temporal_branch = TemporalBranch(embedding_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM).to(self.device)
        self.pooling = AttentionPooling(fusion_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM).to(self.device)
        self.classifier = ClassificationHead(input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM, 
                                             num_classes=config.TRAIN.NUM_CLASSES).to(self.device)
        # self.classifier = SimpleClassifier(input_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM, num_classes=config.TRAIN.NUM_CLASSES).to(self.device)

        self.criterion = nn.CrossEntropyLoss() #  if config.TRAIN.NUM_CLASSES > 1 else nn.BCEWithLogitsLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.alpha = 0.3

        '''
        learnable_params = list(self.temporal_branch.parameters()) + \
                           list(self.pooling.parameters()) + \
                           list(self.classifier.parameters())
        self.optimizer = optim.AdamW(learnable_params, lr=config.TRAIN.LR, weight_decay=1e-3)
        '''
        
        decay = []
        no_decay = []
        self.decay = 5e-4
        '''
        '''        
        # Encoder (lr = base_lr * 0.5)
        for name, param in self.encoder.module.named_parameters():
            if not param.requires_grad:
                continue
            if any(block in name for block in target_blocks):
                if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                    no_decay.append({"params": param, "weight_decay": 0.0, "lr": config.TRAIN.LR * 0.5})
                else:
                    decay.append({"params": param, "weight_decay": 5e-4, "lr": config.TRAIN.LR * 0.5})
        # TemporalBranch (lr = base_lr)
        for name, param in self.temporal_branch.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                no_decay.append({"params": param, "weight_decay": 0.0, "lr": config.TRAIN.LR})
            else:
                decay.append({"params": param, "weight_decay": self.decay, "lr": config.TRAIN.LR})

        # AttentionPooling (lr = base_lr * 0.5) -> (lr = base_lr)
        for name, param in self.pooling.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                no_decay.append({"params": param, "weight_decay": 0.0, "lr": config.TRAIN.LR})
            else:
                decay.append({"params": param, "weight_decay": self.decay, "lr": config.TRAIN.LR})

        # Classifier (lr = base_lr * 0.5) -> (lr = base_lr)
        for name, param in self.classifier.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                no_decay.append({"params": param, "weight_decay": 0.0, "lr": config.TRAIN.LR * 0.5})
            else:
                decay.append({"params": param, "weight_decay": self.decay, "lr": config.TRAIN.LR * 0.5})

        self.optimizer = optim.AdamW(decay + no_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=1e-5)

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=5e-5)
        '''
        # ReduceLROnPlateau scheduler: Reduce lr when validation loss isn't improve
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',              # loss 기준
            factor=0.5,              # lr을 절반으로
            patience=3,              # 3번 동안 개선 없으면 감소
            verbose=True,
            min_lr=1e-6              # 최소 학습률
        )
        '''
        '''
        self.num_train_sessions = 11420
        print(f"Computed steps_per_epoch (session-level): {self.num_train_sessions}")

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS,
            steps_per_epoch=self.num_train_sessions // config.TRAIN.BATCH_SIZE,
            pct_start=0.4, anneal_strategy='cos', final_div_factor=1000, cycle_momentum=False
        )
        '''
        # create lists to store embeddings and labels for t-SNE
        self.chunk_embeddings_for_tsne = []
        self.chunk_labels_for_tsne = []
        self.session_embeddings_for_tsne = []
        self.session_labels_for_tsne = []
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        
    def reconstruct_sessions(self, batch, idx):
        """
        Reconstruct session-level embeddings from a batch.
        If SPLIT_METHOD is empty (session-level processing), each sample in the batch
        is a complete session containing multiple chunks.
        
        Returns:
            session_embeddings: dict mapping session_id to session-level embedding tensor.
            session_labels: dict mapping session_id to label.
        """
        session_embeddings = {}
        session_labels = {}

        pooled_embeddings = []
        session_ids = []

        # Assume batch is a tuple of (chunks, label, session_id, file_list)
        # where chunks shape: (num_chunks, chunk_length, H, W, C)
        batch_chunks, batch_labels, batch_session_ids, _ = batch

        # Loop over each sample (session) in the batch
        for chunks, label, sess_id in zip(batch_chunks, batch_labels, batch_session_ids):
            # Process each chunk through the decoder
            # Here, we need to rearrange dimensions as expected by the temporal branch.
            # For example, if temporal_branch expects (B, C, T, H, W):
            chunk_embeddings = []
            chunk_pooled = []
            num_chunks = chunks.shape[0]
            print(f"[DEBUG] Session {sess_id} → Number of chunks: {num_chunks}")
            
            for i in range(num_chunks):
                # Get i-th chunk: shape (C, chunk_length, H, W)
                chunk = chunks[i]
                # PhysMamba expects (B, C, T, H, W) for 3D convolution and it has been already satisfied
                # chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(self.device)
                chunk_tensor = chunk.unsqueeze(0).float().to(self.device, non_blocking=True)

                # print("Original chunk shape:", chunk.shape)
                
                # with torch.no_grad():
                # rppg = self.encoder(chunk_tensor)
                '''
                '''
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    rppg = self.encoder(chunk_tensor)
                # print(f"[DEBUG][rPPG] mean: {rppg.mean().item():.4f}, std: {rppg.std().item():.4f}")
                rppg = rppg.float()
                rppg = normalize_rppg(rppg)            # differentiable z-score normalization
                rppg = rppg.unsqueeze(-1)              # (B, T, 1) for TemporalBranch

                # Pass through temporal branch to obtain embedding (1, embed_dim)
                emb, pooled = self.temporal_branch(rppg)
                chunk_embeddings.append(emb)
                
            # Concatenate chunk embeddings: (num_chunks, embed_dim)
            chunk_embeddings = torch.cat(chunk_embeddings, dim=0)
            print(f"[DEBUG] requires_grad per chunk: {[e.requires_grad for e in chunk_embeddings]}")
            print("[DEBUG][Chunk Embedding] mean/std:")
            for i, emb in enumerate(chunk_embeddings):
                print(f"  Chunk {i}: mean={emb.mean().item():.4f}, std={emb.std().item():.4f}")
                # Chunk-level embeddings
                self.chunk_embeddings_for_tsne.append(emb.detach().cpu().numpy()) 
                self.chunk_labels_for_tsne.append(label)  # same label for each chunk of the session

            # Fuse the chunk embeddings using the attention pooling module
            sess_embedding, attn_weights = self.pooling(chunk_embeddings, return_weights=True)  # shape: (embed_dim,)
            print("[DEBUG][Attention] weights:", attn_weights.squeeze().tolist())

            session_embeddings[sess_id] = sess_embedding
            session_labels[sess_id] = label  # label is assumed to be already a scalar or tensor
            
            # Collect session-level embedding for t-SNE
            self.session_embeddings_for_tsne.append(sess_embedding.detach().cpu().numpy())
            self.session_labels_for_tsne.append(label)
            
            '''
            # for contrastive loss
            pooled_embeddings.append(sess_embedding)  # or chunk_pooled averaged
            session_ids.append(sess_id)
            '''
        # contrastive_loss = self.contrastive_loss(torch.stack(pooled_embeddings), session_ids)
        return session_embeddings, session_labels, 0# contrastive_loss
    
    def run_tsne_and_plot(self, level="chunk", epoch=0):
        """
        Runs t-SNE on either chunk-level or session-level embeddings 
        and saves the plot to a PNG file.

        level: "chunk" or "session"
        epoch: current epoch number (for naming and title)
        """
        # Decide folder name based on level
        if level == "chunk":
            out_dir = "./chunk_TSNE"
            out_name = f"tsne_chunk_level_epoch{epoch}.png"
            emb_list = self.chunk_embeddings_for_tsne
            lbl_list = self.chunk_labels_for_tsne
            title_str = f"Chunk-level t-SNE (Epoch {epoch})"
        else:
            out_dir = "./session_TSNE"
            out_name = f"tsne_session_level_epoch{epoch}.png"
            emb_list = self.session_embeddings_for_tsne
            lbl_list = self.session_labels_for_tsne
            title_str = f"Session-level t-SNE (Epoch {epoch})"
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        if len(emb_list) == 0:
            print(f"[TSNE] No embeddings found for {level}-level. Skipping t-SNE.")
            return

        emb_array = np.vstack(emb_list)  # shape: (N, embed_dim)
        label_array = np.array(lbl_list)

        # We apply TSNE
        tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30)
        tsne_results = tsne.fit_transform(emb_array)  # (N, 2)

        # Plot
        plt.figure(figsize=(6,6))
        unique_labels = np.unique(label_array)
        for ul in unique_labels:
            idx = np.where(label_array == ul)
            plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=f"Label {ul}", alpha=0.7)
        plt.legend()
        plt.title(title_str)
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[TSNE] Saved {level}-level t-SNE plot as {out_path}")

        # Clear the lists if desired (so they don't accumulate over multiple epochs)
        # If you want to keep collecting across epochs, comment these lines out
        if level == "chunk":
            self.chunk_embeddings_for_tsne.clear()
            self.chunk_labels_for_tsne.clear()
        else:
            self.session_embeddings_for_tsne.clear()
            self.session_labels_for_tsne.clear()

    def train(self, data_loader):
        """
        Training loop with session-level aggregation, debugging prints,
        and gradient accumulation to simulate a larger effective batch size.
        """
        torch.autograd.set_detect_anomaly(True)
        
        accumulation_steps = 4  # Number of iterations to accumulate gradients
        for epoch in range(self.max_epoch):
            print(f"\n==== Training Epoch: {epoch} ====")
            tbar = tqdm(data_loader["train"], desc=f"Epoch {epoch}", ncols=80)
            
            # Set the models to training mode
            self.encoder.train()
            self.temporal_branch.train()
            self.pooling.train()
            self.classifier.train()
            
            running_loss = 0.0
            batch_losses = []
            all_logits = []
            
            # Zero gradients before starting the accumulation loop
            self.optimizer.zero_grad()
            
            for idx, batch in enumerate(tbar):
                # Reconstruct session-level embeddings and labels from the batch
                session_emb_dict, session_label_dict, contrastive_loss = self.reconstruct_sessions(batch=batch, idx=idx)
                
                # --- Debugging: Print aggregated embedding shapes per session ---
                for sess_id, emb in session_emb_dict.items():
                    print(f"[DEBUG] Session {sess_id}: Aggregated embedding shape: {emb.shape}")
                # --- End Debugging ---
                
                classification_losses = []
                for sess_id, emb in session_emb_dict.items():
                    # Get ground truth label for each session
                    label = session_label_dict[sess_id]
                    if not torch.is_tensor(label):
                        label_tensor = torch.tensor(label, dtype=torch.long).unsqueeze(0).to(self.device)
                    else:
                        label_tensor = label.to(self.device).long().unsqueeze(0)
        
                    # Forward pass: Pass the aggregated session embedding to the classifier
                    input_vec = emb.unsqueeze(0).to(self.device)
                    outputs = self.classifier(input_vec)
                    loss = self.criterion(outputs, label_tensor)
        
                    # --- Debugging: Print classifier outputs ---
                    logits = outputs
                    print(f"[DEBUG][Logits] {logits}")
                    print(f"[DEBUG][Logits] Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")
                    probs = torch.softmax(logits, dim=1)
                    print(f"[DEBUG] GT label: {label}, Probs after Softmax: {probs}")
                    print(f"[DEBUG][Softmax] Mean: {probs.mean().item():.4f}, Std: {probs.std().item():.4f}")
                    # --- End Debugging ---
                    
                    print(f"[DEBUG] Session {sess_id} - Loss: {loss.item():.4f}")
                    print(f"[DEBUG] Session {sess_id} - Embedding mean: {emb.mean().item():.4f}, std: {emb.std().item():.4f}")
                    print(f"[DEBUG] Session {sess_id} - GT label: {label}, Predicted logits: {outputs.detach().cpu().numpy()}")
                    # --- End Debugging ---
        
                    classification_losses.append(loss)
                    running_loss += loss.item()
                    batch_losses.append(loss.item())
                
                # Compute total loss for this batch (contrastive_loss is assumed to be 0 if not used)
                classification_total = torch.stack(classification_losses).mean()
                batch_loss = classification_total + self.alpha * contrastive_loss
                
                # Scale the loss for gradient accumulation
                batch_loss = batch_loss / accumulation_steps
                batch_loss.backward()
                
                # Optionally perform gradient checking every (100 * accumulation_steps) iterations
                if (idx + 1) % accumulation_steps == 0:
                    if idx % (100 * accumulation_steps) == 0:
                        for name, param in self.temporal_branch.named_parameters():
                            if param.grad is not None:
                                print(f"[GradCheck][Temporal] {name} grad norm: {param.grad.norm().item():.6f}")
                            else:
                                print(f"[GradCheck][Temporal] {name} has no grad")
                        for name, param in self.pooling.named_parameters():
                            if param.grad is not None:
                                print(f"[GradCheck][AttentionPooling] {name} grad norm: {param.grad.norm().item():.6f}")
                            else:
                                print(f"[GradCheck][AttentionPooling] {name} has no grad")
                        for name, param in self.classifier.named_parameters():
                            if param.grad is not None:
                                print(f"[GradCheck][Classifier] {name} grad norm: {param.grad.norm().item():.6f}")
                            else:
                                print(f"[GradCheck][Classifier] {name} has no grad")
        
                    # Step the optimizer and scheduler after accumulating gradients
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()  # For CosineAnnealingWarmRestarts, step per update
                
                if idx % 100 == 0:
                    # === GRADIENT CHECK ===
                    for name, param in self.encoder.named_parameters():
                        if param.grad is not None:
                            print(f"[GradCheck][Encoder] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Encoder] {name} has no grad")
                    for name, param in self.temporal_branch.named_parameters():
                        if param.grad is not None:
                            print(f"[GradCheck][Temporal] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Temporal] {name} has no grad")
                    for name, param in self.pooling.named_parameters():
                        if param.grad is not None:
                            print(f"[GradCheck][AttentionPooling] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Classifier] {name} has no grad")
                    for name, param in self.classifier.named_parameters():
                        if param.grad is not None:
                            print(f"[GradCheck][Classifier] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Classifier] {name} has no grad")
                               
                avg_loss = running_loss / (idx + 1)
                tbar.set_postfix(loss=f"{avg_loss:.3f}")
            
            torch.cuda.empty_cache()
            
            # Epoch results logging
            avg_train_loss = np.mean(batch_losses)
            self.train_losses.append(avg_train_loss)
            print(f"Epoch {epoch}: Average Training Loss: {avg_train_loss:.4f}")
        
            if all_logits:
                all_logits_tensor = torch.cat(all_logits, dim=0)
                softmax_out = torch.softmax(all_logits_tensor, dim=1)
                print(f"[DEBUG][Epoch {epoch}] Softmax output mean: {softmax_out.mean().item():.4f}, std: {softmax_out.std().item():.4f}")

            # run t-SNE
            self.run_tsne_and_plot(level="chunk", epoch=epoch)
            self.run_tsne_and_plot(level="session", epoch=epoch)
            
            # Validation step
            val_loss, metrics = self.valid(data_loader)
            self.val_losses.append(val_loss)
            print(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            print("Validation Confusion Matrix:")
            print(metrics["confusion_matrix"])
        
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model(epoch)
                print(f"Best model updated at epoch {epoch} with validation loss {val_loss:.4f}")
        
        self.plot_losses_and_lrs()

    def valid(self, data_loader):
        """Validation loop for session-level data."""
        self.encoder.eval()
        self.temporal_branch.eval()
        self.pooling.eval()
        self.classifier.eval()
        
        all_preds = []
        all_labels = []
        losses = []
        
        print("\n==== Validating ====")
        vbar = tqdm(data_loader["valid"], ncols=80, desc="Validation")
        with torch.no_grad():
            for valid_batch in vbar:
                session_emb_dict, session_label_dict, _ = self.reconstruct_sessions(batch=valid_batch, idx=None)
                for sess_id, emb in session_emb_dict.items():
                    label = session_label_dict[sess_id]
                    if not torch.is_tensor(label):
                        label_tensor = torch.tensor(label, dtype=torch.long).unsqueeze(0).to(self.device)
                    else:
                        label_tensor = label.to(self.device).long().unsqueeze(0)
        
                    # Forward pass: Pass the aggregated session embedding to the classifier
                    input_vec = emb.unsqueeze(0).to(self.device)
                    outputs = self.classifier(input_vec)
                    
                    # Compute loss
                    loss = self.criterion(outputs, label_tensor)
                    losses.append(loss.item())
                    
                    # Get softmax probabilities
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    # --- Debugging: Print Validation outputs ---
                    print(f"[DEBUG] GT label: {label}, Softmax probabilities: {probs.cpu().numpy()}")
                    print(f"[DEBUG] Predicted label: {preds.item()}")

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(label_tensor.cpu().numpy())
                    
                    vbar.set_postfix(loss=f"{loss.item():.3f}")
        torch.cuda.empty_cache()
        
        avg_loss = np.mean(losses)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        conf_mat = confusion_matrix(all_labels, all_preds)
        
        metrics = {"accuracy": accuracy, "f1": f1, "confusion_matrix": conf_mat}
        return avg_loss, metrics

    def test(self, data_loader):
        """Test loop for session-level data."""
        # self.encoder.eval()
        self.temporal_branch.eval()
        self.pooling.eval()
        self.classifier.eval()
        
        all_preds = []
        all_labels = []
        
        print("\n==== Testing ====")
        tbar = tqdm(data_loader["test"], ncols=80, desc="Testing")
        with torch.no_grad():
            for test_batch in tbar:
                session_emb_dict, session_label_dict, _ = self.reconstruct_sessions(batch=test_batch, idx=None)
                for sess_id, emb in session_emb_dict.items():
                    label = session_label_dict[sess_id]
                    if not torch.is_tensor(label):
                        label_tensor = torch.tensor(label, dtype=torch.long).unsqueeze(0).to(self.device)
                    else:
                        label_tensor = label.to(self.device).long().unsqueeze(0)
        
                    # Forward pass: Pass the aggregated session embedding to the classifier
                    input_vec = emb.unsqueeze(0).to(self.device)
                    outputs = self.classifier(input_vec)

                    preds = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(label_tensor.cpu().numpy())
                    
                    tbar.set_postfix()
        torch.cuda.empty_cache()
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        conf_mat = confusion_matrix(all_labels, all_preds)
        print("Test Accuracy: {:.4f}".format(accuracy))
        print("Test F1 Score: {:.4f}".format(f1))
        print("Confusion Matrix:")
        print(conf_mat)
        return {"accuracy": accuracy, "f1": f1, "confusion_matrix": conf_mat}

    def save_model(self, epoch):
        """Saves the best classifier model."""
        model_save_dir = "./saved_models"
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir, exist_ok=True)
        split_method = self.config.TRAIN.DATA.SPLIT_METHOD
        current_subject = self.config.TRAIN.DATA.get('current_subject', 'all')
        fold_index = self.config.TRAIN.DATA.get('FOLD_INDEX', 'NA')
        filename = f"{self.config.TRAIN.DATA.DATASET}_{split_method}_subject{current_subject}_fold{fold_index}_epoch{epoch}.pth"
        model_path = os.path.join(model_save_dir, filename)
        torch.save(self.classifier.state_dict(), model_path)
        print("Saved best model at:", model_path)

    def plot_losses_and_lrs(self):
        """Plots and saves training and validation loss curves."""
        output_dir = os.path.join(self.config.LOG.PATH, self.config.TRAIN.DATA.EXP_DATA_NAME, 'plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        epochs = range(len(self.train_losses))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        loss_plot_path = os.path.join(output_dir, "loss_plot.pdf")
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()
        print("Saved loss plot at:", loss_plot_path)
        
    '''
    def train(self, data_loader):
        """Training loop with session-level aggregation and debugging prints."""
        torch.autograd.set_detect_anomaly(True)
    
        for epoch in range(self.max_epoch):
            print(f"\n==== Training Epoch: {epoch} ====")
            tbar = tqdm(data_loader["train"], desc=f"Epoch {epoch}", ncols=80)
            
            # self.encoder.train()
            self.temporal_branch.train()
            self.pooling.train()
            self.classifier.train()
            
            running_loss = 0.0
            batch_losses = []
            all_logits = []

            for idx, batch in enumerate(tbar):
                # Reconstruct session-level embeddings and labels from the batch
                session_emb_dict, session_label_dict, contrastive_loss = self.reconstruct_sessions(batch=batch, idx=idx)
                
                # --- Debugging: Check that each session has the correct number of chunks ---
                for sess_id, emb in session_emb_dict.items():
                    # You can print additional details if needed
                    print(f"[DEBUG] Session {sess_id}: Aggregated embedding shape: {emb.shape}")
                # --- End Debugging ---
                
                batch_loss = 0.0
                classification_losses = []

                for sess_id, emb in session_emb_dict.items():
                    # For each session, get its corresponding GT label
                    label = session_label_dict[sess_id]
                    if not torch.is_tensor(label):
                        label_tensor = torch.tensor(label, dtype=torch.long).unsqueeze(0).to(self.device)
                    else:
                        label_tensor = label.to(self.device).long().unsqueeze(0)

                    # Pass the aggregated session embedding to the classifier
                    input_vec = emb.unsqueeze(0).to(self.device)
                    outputs = self.classifier(input_vec)
                    loss = self.criterion(outputs, label_tensor)

                    # --- Debugging: Print classifier logits ---
                    logits = outputs
                    print(f"[DEBUG][Logits] {logits}")
                    print(f"[DEBUG][Logits] Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")
                    probs = torch.softmax(logits, dim=1)
                    print(f"[DEBUG] GT label: {label}, Probs after Softmax: {probs}")
                    print(f"[DEBUG][Softmax] Mean: {probs.mean().item():.4f}, Std: {probs.std().item():.4f}")
                    # --- End Debugging ---
                    
                    # --- Debugging: Print details for this session ---
                    print(f"[DEBUG] Session {sess_id} - Loss: {loss.item():.4f}")
                    print(f"[DEBUG] Session {sess_id} - Embedding mean: {emb.mean().item():.4f}, std: {emb.std().item():.4f}")
                    print(f"[DEBUG] Session {sess_id} - GT label: {label}, Predicted logits: {outputs.detach().cpu().numpy()}")
                    # --- End Debugging ---

                    classification_losses.append(loss)
                    running_loss += loss.item()
                    batch_losses.append(loss.item())
                
                # === Total Loss ===
                classification_total = torch.stack(classification_losses).mean()
                batch_loss = classification_total + self.alpha * contrastive_loss
                
                # === Backward & Step ===
                self.optimizer.zero_grad()
                batch_loss.backward()
                
                if idx % 100 == 0:
                    # === GRADIENT CHECK ===
                    for name, param in self.temporal_branch.named_parameters():
                        if param.grad is not None:
                            print(f"[GradCheck][Temporal] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Temporal] {name} has no grad")
                    for name, param in self.pooling.named_parameters():
                        if param.grad is not None:
                            print(f"[GradCheck][AttentionPooling] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Classifier] {name} has no grad")
                    for name, param in self.classifier.named_parameters():
                        if param.grad is not None:
                            print(f"[GradCheck][Classifier] {name} grad norm: {param.grad.norm().item():.6f}")
                        else:
                            print(f"[GradCheck][Classifier] {name} has no grad")

                self.optimizer.step()
                self.scheduler.step()
                    
                avg_loss = running_loss / (idx + 1)
                tbar.set_postfix(loss=f"{avg_loss:.3f}")
            
            torch.cuda.empty_cache()
            
            # === Epoch Result ===
            avg_train_loss = np.mean(batch_losses)
            self.train_losses.append(avg_train_loss)
            print(f"Epoch {epoch}: Average Training Loss: {avg_train_loss:.4f}")

            # Optionally, print aggregated logits statistics for the entire epoch:
            if all_logits:
                all_logits_tensor = torch.cat(all_logits, dim=0)
                softmax_out = torch.softmax(all_logits_tensor, dim=1)
                print(f"[DEBUG][Epoch {epoch}] Softmax output mean: {softmax_out.mean().item():.4f}, std: {softmax_out.std().item():.4f}")

            # === Validation ===
            val_loss, metrics = self.valid(data_loader)
            self.val_losses.append(val_loss)
            print(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            print("Validation Confusion Matrix:")
            print(metrics["confusion_matrix"])

            # ReduceLROnPlateau
            # self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model(epoch)
                print(f"Best model updated at epoch {epoch} with validation loss {val_loss:.4f}")
            
        self.plot_losses_and_lrs()
    
    def reconstruct_sessions(self, idx=None, batch=None, return_weights=False):
        """Reconstruct session-level embeddings and labels from batch data."""
        data_batch, labels_batch, session_ids, chunk_ids = batch
        all_session_ids = [int(s) for s in session_ids]
        all_chunk_ids = [int(c) for c in chunk_ids]

        data_tensor = data_batch.to(self.device)
        rppg_signals = self.physmamba(data_tensor)
        rppg_signals.requires_grad_()

        rppg_norm = normalize_rppg(rppg_signals)
        temp_emb = self.temporal_branch(rppg_norm.unsqueeze(-1))
        
        """if idx is not None and idx % 2000 == 0:
            print(f"[DEBUG] rppg_signals.requires_grad: {rppg_signals.requires_grad}")
            print(f"[DEBUG] rppg_norm.requires_grad: {rppg_norm.requires_grad}")
            print(f"[DEBUG] Step {idx}: [rPPG] shape: {rppg_signals.shape}, mean: {rppg_signals.mean().item():.4f}, std: {rppg_signals.std().item():.4f}")
            print(f"[DEBUG] temp_emb shape: {temp_emb.shape}, mean={temp_emb.mean().item():.4f}, std={temp_emb.std().item():.4f}")"""
        if return_weights:
            session_embeddings, session_attentions = aggregate_session_embeddings(
                temp_emb, all_session_ids, all_chunk_ids, self.pooling, return_weights=True
            )
        else:
            session_embeddings = aggregate_session_embeddings(
                temp_emb, all_session_ids, all_chunk_ids, self.pooling, return_weights=False
            )
            session_attentions = None
        """
        if idx is not None and idx % 2000 == 0:
            print(f"[DEBUG] Aggregated {len(session_embeddings)} session embeddings")
            for sess_id, emb in session_embeddings.items():
                print(f"  Session {sess_id} emb: shape={emb.shape}, mean={emb.mean().item():.4f}, std={emb.std().item():.4f}")
        """
        session_labels = {}
        for s, lab in zip(all_session_ids, labels_batch):
            if s not in session_labels:
                session_labels[s] = lab
            else:
                if session_labels[s] != lab:
                    print(f"[⚠️ Label Mismatch] session {s} has inconsistent labels: {session_labels[s]} vs {lab}")

        if return_weights:
            return session_embeddings, session_labels, session_attentions
        else:
            return session_embeddings, session_labels

    def debug_gradients(self):
        print("\n[GRADIENT CHECK]")
        for name, param in self.temporal_branch.named_parameters():
            if param.requires_grad:
                grad_norm = param.grad.norm().item() if param.grad is not None else None
                print(f"{name} → grad: {'None' if grad_norm is None else f'{grad_norm:.4e}'}")

    def debug_optimizer(self):
        lr = self.optimizer.param_groups[0]['lr']
        print(f"[DEBUG][Optimizer] Current learning rate: {lr:.6f}")
    
    def debug_weights(self):
        print("\n[WEIGHT CHECK]")
        for name, param in self.classifier.named_parameters():
            if param.requires_grad:
                print(f"{name} → weight mean: {param.data.mean().item():.4f}")
          
    def visualize_attention_sample(self, train_loader, epoch):
        """ 매 epoch 끝날때 샘플 하나에 대해 Attention score를 뽑고 저장 """
        self.temporal_branch.eval()
        self.classifier.eval()
        self.pooling.eval()

        # ex) 그냥 첫 batch 하나만 가져온다고 가정
        sample_batch = next(iter(train_loader))
        
        # reconstruct_sessions에서 return_weights=True 를 통해 attn_weights도 꺼낼 수 있게 수정
        session_emb_dict, session_label_dict, session_attn_dict = \
            self.reconstruct_sessions(batch=sample_batch, return_weights=True)

        # 세션 하나만 골라서 시각화
        if len(session_attn_dict) < 1:
            print("[INFO] No session found for attention visualization.")
            return

        # 예: 첫번째 세션만
        some_sess_id = list(session_attn_dict.keys())[0]

        alpha = session_attn_dict[some_sess_id]  # shape (T,1)
        alpha = alpha.squeeze()
        if alpha.ndim == 0:
            alpha = alpha.unsqueeze(0)
        alpha = alpha.cpu().numpy()             # shape (T,)


        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.bar(range(len(alpha)), alpha, color='skyblue')
        plt.title(f"Epoch {epoch} - Session {some_sess_id} Attention")
        plt.xlabel("Chunk index")
        plt.ylabel("Weight")

        save_dir = "./attention_plots"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch{epoch}_sess{some_sess_id}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved attention plot at {save_path}")
    
    # The train(), valid(), test(), save_model(), and plot_losses_and_lrs() functions remain unchanged.
    def train(self, data_loader):
        """Training loop with session-level aggregation and visualization."""  
        torch.autograd.set_detect_anomaly(True)
      
        for epoch in range(self.max_epoch):
            print(f"\n==== Training Epoch: {epoch} ====")
            tbar = tqdm(data_loader["train"], desc=f"Epoch {epoch}", ncols=80)
            self.temporal_branch.train()
            self.classifier.train()
            running_loss = 0.0
            batch_losses = []
            all_logits = []

            # Iterate over training batches
            for idx, batch in enumerate(tbar):
                if idx == 0:
                    print("\n[DEBUG][CHECK] TemporalBranch parameter requires_grad and device")
                    for name, param in self.temporal_branch.named_parameters():
                        print(f"  {name}: requires_grad={param.requires_grad}, device={param.device}")

                    print("\n[DEBUG][CHECK] Pooling parameter requires_grad and device")
                    for name, param in self.pooling.named_parameters():
                        print(f"  {name}: requires_grad={param.requires_grad}, device={param.device}")

                    print("\n[DEBUG][CHECK] Classifier parameter requires_grad and device")
                    for name, param in self.classifier.named_parameters():
                        print(f"  {name}: requires_grad={param.requires_grad}, device={param.device}")

                # [DEBUG] Print batch length
                # print(f"[DEBUG] Epoch {epoch}, Batch {idx}, batch length: {len(batch)}")

                # Reconstruct session-level embeddings and labels
                session_emb_dict, session_label_dict = self.reconstruct_sessions(idx=idx, batch=batch)
                if not session_emb_dict:
                    print(f"[WARNING] No session embedding found at step {idx}")
                # [DEBUG] Print unique labels in the batch
                """
                if idx % 400 == 0: 
                    unique_labels = set(session_label_dict.values())
                    print(f"[DEBUG] Unique labels in batch {idx}: {unique_labels}")
                """
                
                # Accumulate loss over sessions in this batch
                batch_loss = 0.0
                num_sessions = 0
                """
                if idx % 2000 == 0:
                    print("[DEBUG] Label Counter:", Counter(session_label_dict.values()))
                """
                for sess_id, emb in session_emb_dict.items():
                    # [DEBUG] Print session embedding shape``
                    # print(f"[DEBUG] Batch {idx} has no session embeddings! Skipping...")
                    # print(f"[DEBUG] emb shape for session {sess_id}: {emb.shape}")

                    # Prepare input vector for the classifier
                    input_vec = emb.unsqueeze(0).to(self.device)  # (1, fusion_dim)
                    label = session_label_dict[sess_id]
                    # label_tensor = torch.tensor(label).float().unsqueeze(0).to(self.device)
                    """
                    if label_tensor.dim() == 1:
                        label_tensor = label_tensor.unsqueeze(1)
                    """
                    label_tensor = torch.tensor([label], dtype=torch.long).to(self.device)

                    # Forward pass through classifier
                    outputs = self.classifier(input_vec)
                    loss = self.criterion(outputs, label_tensor)

                    if idx % 200 == 0:
                        self.debug_weights()
                        self.debug_gradients()
                        self.debug_optimizer()
                        """
                        for sid, emb in session_emb_dict.items():
                            print(f"[DEBUG] session {sid} pooled embedding: shape={emb.shape}, mean={emb.mean().item():.4f}, std={emb.std().item():.4f}")
                            break
                        print(f"[DEBUG][Loss] session {sess_id} loss: {loss.item():.4f}")
                        print(f"[DEBUG] Session {sess_id} → mean: {emb.mean().item():.4f}, std: {emb.std().item():.4f}")
                        print(f"[DEBUG] classifier output (logits): {outputs.detach().cpu().numpy()}")
                        probs = torch.softmax(outputs, dim=1)
                        print(f"[DEBUG] classifier probs: {probs.detach().cpu().numpy()}")
                        all_logits = []
                        for sess_id, emb in session_emb_dict.items():
                            input_vec = emb.unsqueeze(0).to(self.device)
                            outputs = self.classifier(input_vec)
                            all_logits.append(outputs.detach().cpu())
                        if all_logits:
                            all_logits_tensor = torch.cat(all_logits, dim=0)  # shape: (num_sessions, 2)
                            softmax_out = torch.softmax(all_logits_tensor, dim=1)
                            print(f"[DEBUG] Softmax output: {softmax_out}")
                            print(f"[DEBUG] Softmax (mean): {softmax_out.mean().item():.4f}, std: {softmax_out.std().item():.4f}")
                            print(f"[DEBUG] Logits mean: {all_logits_tensor.mean().item():.4f}, std: {all_logits_tensor.std().item():.4f}")
                        if idx % 200 == 0 and all_logits:
                            all_logits_tensor = torch.cat(all_logits, dim=0)
                            preds = torch.argmax(all_logits_tensor, dim=1).to(self.device)
                            true = torch.tensor(list(session_label_dict.values()), device=self.device)
                            correct = (preds == true).sum().item()
                            print(f"[DEBUG] batch acc (approx): {correct}/{len(true)} = {correct / len(true):.2f}")
                        """
                    # Ensure label shape is correct in BCEWithLogitsLoss
                    batch_loss += loss
                    running_loss += loss.item()
                    batch_losses.append(loss.item())
                    all_logits.append(outputs.detach().cpu())
                    num_sessions += 1
                    
                # Backward once per batch
                if num_sessions > 0:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                
            avg_loss = running_loss / (idx + 1)
            tbar.set_postfix(loss=f"{avg_loss:.3f}")
            """
            # Print running loss every 100 mini-batches
            if (idx + 1) % 100 == 99:
                avg_loss = running_loss / (idx + 1)
                print(f"[Epoch {epoch}, Batch {idx+1}] Loss: {avg_loss:.3f}")
            """
            avg_train_loss = np.mean(batch_losses)
            self.train_losses.append(avg_train_loss)
            print(f"Epoch {epoch}: Average Training Loss: {avg_train_loss:.4f}")
            
            # DEBUG: Logits statistics per epoch
            if all_logits:
                all_logits_tensor = torch.cat(all_logits, dim=0)
                softmax_out = torch.softmax(all_logits_tensor, dim=1)
                print(f"[DEBUG][Epoch {epoch}] Softmax output: {softmax_out}")
                print(f"[DEBUG][Epoch {epoch}] Softmax (mean): {softmax_out.mean().item():.4f}, std: {softmax_out.std().item():.4f}")
                print(f"[DEBUG][Epoch {epoch}] Logits mean: {all_logits_tensor.mean().item():.4f}, std: {all_logits_tensor.std().item():.4f}")

            # self.visualize_attention_sample(data_loader["train"], epoch)

            # Validation
            val_loss, metrics = self.valid(data_loader)
            self.val_losses.append(val_loss)
            print(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            print("Validation Confusion Matrix:")
            print(metrics["confusion_matrix"])
            
            # Save model if validation loss improves
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model(epoch)
                print(f"Best model updated at epoch {epoch} with validation loss {val_loss:.4f}")
        
        self.plot_losses_and_lrs()

    def valid(self, data_loader):
        """Validation loop with session-level aggregation and progress visualization."""
        self.temporal_branch.eval()
        self.classifier.eval()
        all_preds = []
        all_labels = []
        losses = []
        
        print("\n==== Validating ====")
        vbar = tqdm(data_loader["valid"], ncols=80, desc="Validation")
        with torch.no_grad():
            for valid_batch in vbar:
                session_emb_dict, session_label_dict = self.reconstruct_sessions(batch=valid_batch)
                for sess_id, emb in session_emb_dict.items():
                    input_vec = emb.unsqueeze(0).to(self.device)
                    label = session_label_dict[sess_id]
                    # label_tensor = torch.tensor(label).float().unsqueeze(0).to(self.device)
                    label_tensor = torch.tensor([label], dtype=torch.long).to(self.device)  # shape: (1,)
                    outputs = self.classifier(input_vec)
                    """
                    if label_tensor.dim() == 1:
                        label_tensor = label_tensor.unsqueeze(1)
                    """
                    
                    loss = self.criterion(outputs, label_tensor)
                    losses.append(loss.item())
                    
                    # preds = torch.sigmoid(outputs)
                    # preds = (preds > 0.5).float()
                    preds = torch.argmax(outputs, dim=1)  # shape: (1,)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(label_tensor.cpu().numpy())
                    
                    vbar.set_postfix(loss=f"{loss.item():.3f}")
        avg_loss = np.mean(losses)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        conf_mat = confusion_matrix(all_labels, all_preds)
        metrics = {"accuracy": accuracy, "f1": f1, "confusion_matrix": conf_mat}
        return avg_loss, metrics


    def test(self, data_loader):
        """Test loop with session-level aggregation and progress visualization."""
        self.temporal_branch.eval()
        self.classifier.eval()
        all_preds = []
        all_labels = []
        
        print("\n==== Testing ====")
        tbar = tqdm(data_loader["test"], ncols=80, desc="Testing")
        with torch.no_grad():
            for test_batch in tbar:
                session_emb_dict, session_label_dict = self.reconstruct_sessions(batch=test_batch)
                for sess_id, emb in session_emb_dict.items():
                    input_vec = emb.unsqueeze(0).to(self.device)
                    label = session_label_dict[sess_id]
                    # label_tensor = torch.tensor(label).float().unsqueeze(0).to(self.device)
                    label_tensor = torch.tensor([label], dtype=torch.long).to(self.device)

                    outputs = self.classifier(input_vec)
                    # preds = torch.sigmoid(outputs)
                    # preds = (preds > 0.5).float()
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(label_tensor.cpu().numpy())
                    
                    tbar.set_postfix()  # Optionally update additional info here
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        conf_mat = confusion_matrix(all_labels, all_preds)
        print("Test Accuracy: {:.4f}".format(accuracy))
        print("Test F1 Score: {:.4f}".format(f1))
        print("Confusion Matrix:")
        print(conf_mat)
        return {"accuracy": accuracy, "f1": f1, "confusion_matrix": conf_mat}


    def save_model(self, epoch):
        """Saves the best classifier model."""
        model_save_dir = "./saved_models"
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir, exist_ok=True)
        split_method = self.config.TRAIN.DATA.SPLIT_METHOD
        current_subject = self.config.TRAIN.DATA.get('current_subject', 'all')
        fold_index = self.config.TRAIN.DATA.get('FOLD_INDEX', 'NA')
        filename = f"{self.config.TRAIN.DATA.DATASET}_{split_method}_subject{current_subject}_fold{fold_index}_epoch{epoch}.pth"
        model_path = os.path.join(model_save_dir, filename)
        torch.save(self.classifier.state_dict(), model_path)
        print("Saved best model at:", model_path)

    def plot_losses_and_lrs(self):
        """Plots and saves training and validation loss curves."""
        output_dir = os.path.join(self.config.LOG.PATH, self.config.TRAIN.DATA.EXP_DATA_NAME, 'plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        epochs = range(len(self.train_losses))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        loss_plot_path = os.path.join(output_dir, "loss_plot.pdf")
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()
        print("Saved loss plot at:", loss_plot_path)
    '''