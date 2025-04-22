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
from trainer.BaseTrainer import BaseTrainer  # Base trainer class

# Import modules from proper directories
from neural_encoders.model.PhysMamba import PhysMamba  # Pretrained rPPG encoder

from decoders.TemporalBranch import TemporalBranch        # Temporal decoder (CNN-LSTM)
from decoders.PRVBranch import PRVBranch                  # Differentiable PRV decoder
from decoders.FrequencyBranch import FrequencyBranch      # Frequency decoder (CWFrequency)

from modules.AttentionPooling import AttentionPooling     # Attention-based pooling
from modules.AttentionFusion import AttentionFusion        # Attention-based fusion

from modules.ClassificationHead import ClassificationHead  # Final classifier
from tools.utils import normalize_rppg, aggregate_session_embeddings, visualize_rPPG, visualize_rPPG_frequency

class PhysMambaTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        """
        Initialize the Emotion Trainer.
        [PhysMamba]
            ↓
        [normalized rPPG]
            ↓
        ┌────────────┬──────────┬────────────┐
        │ Temporal   │   PRV    │  Frequency │ ← Multi-branch Decoders
        └────────────┴──────────┴────────────┘
            ↓
        [AttentionFusion]
            ↓
        [chunk embeddings] (N_chunks, fusion_dim)
            ↓
        [AttentionPooling] ← session-level
            ↓
        [Classifier]

        - Loads the pretrained PhysMamba encoder and freezes its weights.
        - Initializes multi-branch decoders: Temporal, PRV, Frequency.
        - Initializes the AttentionFusion module for multi-branch fusion.
        - Initializes the AttentionPooling module for session-level aggregation.
        - Initializes the ClassificationHead for final fusion and emotion classification.
        - Sets up the optimizer and OneCycleLR scheduler based on session-level updates.
        """
        super(PhysMambaTrainer, self).__init__()
        self.device = torch.device(config.DEVICE)
        self.config = config
        self.max_epoch = config.TRAIN.EPOCHS
        self.batch_size = config.TRAIN.BATCH_SIZE

        # ---------------------- Encoder Initialization ----------------------
        self.physmamba = PhysMamba().to(self.device)
        pretrained_path = os.path.join("./pretrained_encoders", "UBFC-rPPG_PhysMamba_DiffNormalized.pth")
        self.physmamba = torch.nn.DataParallel(self.physmamba)
        self.physmamba.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        for param in self.physmamba.module.parameters():
            param.requires_grad = False
        '''
        # Get parameters for unfrozen high-level PhysMamba layers
        unfrozen_phys_params = []
        for block in [self.physmamba.module.Block5, self.physmamba.module.Block6,
                    self.physmamba.module.fuse_2, self.physmamba.module.upsample2]:
            for param in block.parameters():
                param.requires_grad = True
                unfrozen_phys_params.append(param)
        '''

        # ---------------------- Multi-branch Decoders ----------------------
        self.temporal_branch = TemporalBranch(embedding_dim=config.MODEL.EMOTION.TEMPORAL_EMBED_DIM).to(self.device)
        self.prv_branch = PRVBranch(embedding_dim=config.MODEL.EMOTION.PRV_EMBED_DIM).to(self.device)
        self.freq_branch = FrequencyBranch(embedding_dim=config.MODEL.EMOTION.CWFREQ_EMBED_DIM).to(self.device)

        # ---------------------- Attention-based Pooling ----------------------
        fusion_dim = config.MODEL.EMOTION.FUSION_DIM
        self.pooling = AttentionPooling(fusion_dim=fusion_dim).to(self.device)

        # --------------------------- Fusion Layer ---------------------------
        self.fusion = AttentionFusion(
                        input_dims=[
                            config.MODEL.EMOTION.TEMPORAL_EMBED_DIM,
                            config.MODEL.EMOTION.PRV_EMBED_DIM,
                            config.MODEL.EMOTION.CWFREQ_EMBED_DIM
                        ],
                        fusion_dim=config.MODEL.EMOTION.FUSION_DIM
                    ).to(self.device)
        
        # ----------------------- Classification Head  -----------------------
        self.classifier = ClassificationHead(input_dim=fusion_dim, num_classes=config.TRAIN.NUM_CLASSES).to(self.device)

        # -------------------------- Loss Function --------------------------
        if config.TRAIN.NUM_CLASSES == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # ---------------------------- Optimizer ----------------------------
        # Only the decoders and classifier are learnable
        learnable_params = list(self.temporal_branch.parameters()) + \
                           list(self.prv_branch.parameters()) + \
                           list(self.freq_branch.parameters()) + \
                           list(self.fusion.parameters()) + \
                           list(self.pooling.parameters()) + \
                           list(self.classifier.parameters())
        # self.optimizer = optim.Adam(learnable_params, lr=config.TRAIN.LR, weight_decay=0.0005)
        self.optimizer = optim.AdamW(learnable_params, lr=config.TRAIN.LR, weight_decay=1e-4)

        '''
        # Create optimizer with two parameter groups:
        # - Higher LR for classifier parameters.
        # - Lower LR for unfrozen PhysMamba layers.
        self.optimizer = optim.Adam([
                                    {'params': classifier_params, 'lr': config.TRAIN.LR, 'weight_decay': 0.01},
                                    {'params': unfrozen_phys_params, 'lr': config.TRAIN.LR * 0.1, 'weight_decay': 0.01}
                                    ])
        '''

        # --------------------- Scheduler Initialization ---------------------
        """ Scheduler: CosineAnnealingWarmRestarts """
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        """ Scheduler: CosineAnnealingLR """
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-6)
        
        """ Scheduler: OneCycleLR with computed steps_per_epoch """
        # Compute steps_per_epoch based on session-level aggregation
        # self.num_train_sessions = self.compute_steps_per_epoch(data_loader["train"])
        self.num_train_sessions = 11420 # <--- Hardcoded value for steps_per_epoch(precomputed in 8:2 split, batch_size=4)
        print(f"Computed steps_per_epoch (session-level): {self.num_train_sessions}")
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS,
            steps_per_epoch=self.num_train_sessions // config.TRAIN.BATCH_SIZE,
            pct_start=0.1, anneal_strategy='cos', final_div_factor=1000, cycle_momentum=False
        )

        # ---------------------- Metrics Initialization ----------------------
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []

    def compute_steps_per_epoch(self, train_batches):
        """
        Computes the total number of session-level updates per epoch.
        Aggregates session counts using aggregate_session_embeddings.
        """
        total_sessions = 0
        with torch.no_grad():
            for batch in tqdm(train_batches, desc="Computing steps per epoch", total=len(train_batches), leave=False):
                # Unpack the batch assuming batch = (data, labels, session_ids, chunk_ids)
                data_batch, labels_batch, session_ids, chunk_ids = batch

                all_data, all_session_ids, all_chunk_ids = [], [], []
                # Iterate over samples in the batch using zip
                for data, label, session_id, chunk_id in zip(data_batch, labels_batch, session_ids, chunk_ids):
                    # data is expected to be a torch.Tensor; session_id and chunk_id can be strings/ints
                    if isinstance(data, str):
                        raise TypeError(f"Data is string: {data}")
                    # data is already a tensor, so no need for torch.tensor(data)
                    all_data.append(data)
                    all_session_ids.append(int(session_id))  # convert session id to int if needed
                    all_chunk_ids.append(int(chunk_id))      # convert chunk id to int if needed

                # Stack the video data tensors
                data_tensor = torch.stack(all_data, dim=0).to(self.device)  # shape: (N, C, T, H, W)
                # Forward pass through the encoder
                rppg_signals = self.physmamba(data_tensor)  # (N, rPPG_SIGNAL_LENGTH)
                rppg_norm = normalize_rppg(rppg_signals)       # Normalize rPPG
                temp_emb = self.temporal_branch(rppg_norm.unsqueeze(-1))
                prv_emb = self.prv_branch(rppg_norm)
                freq_emb = self.freq_branch(rppg_norm)
                chunk_embeddings = torch.cat([temp_emb, prv_emb, freq_emb], dim=1)
                session_dict = aggregate_session_embeddings(chunk_embeddings, all_session_ids, all_chunk_ids)
                total_sessions += len(session_dict)
        return total_sessions


    def reconstruct_sessions(self, idx=None, batch=None):
        """
        Reconstructs session-level embeddings from a batch.
        Uses PhysMamba encoder, normalization, multi-branch decoders, and aggregation.
        
        Args:
            batch: Tuple of (data, labels, session_ids, chunk_ids)
        Returns:
            session_embeddings: Dict mapping session_id to aggregated embedding (1D tensor).
            session_labels: Dict mapping session_id to label.
        """
        data_batch, labels_batch, session_ids, chunk_ids = batch
        all_session_ids = [int(s) for s in session_ids]
        all_chunk_ids = [int(c) for c in chunk_ids]
        
        # data_batch is already a tensor
        data_tensor = data_batch.to(self.device)  # shape: (N, C, T, H, W)
        
        rppg_signals = self.physmamba(data_tensor)
        rppg_norm = normalize_rppg(rppg_signals)
        '''
        if idx != None and idx % 400 == 0:
            print(f"\n[DEBUG] rppg_norm mean: {rppg_norm.mean().item():.4f}, std: {rppg_norm.std().item():.4f}")
        '''
        temp_emb = self.temporal_branch(rppg_norm.unsqueeze(-1))
        prv_emb = self.prv_branch(rppg_norm)
        freq_emb = self.freq_branch(rppg_norm)
        fused_emb = self.fusion([temp_emb, prv_emb, freq_emb])
        # print(f"[DEBUG] fused_emb shape: {fused_emb.shape}")
        if idx != None and idx % 400 == 0:
            print(f"\n[DEBUG] fused_emb std: {fused_emb.std(dim=0).mean().item():.4f}")
            print(f"[DEBUG] fused_emb - min: {fused_emb.min().item()}, max: {fused_emb.max().item()}")



        # chunk_embeddings = torch.cat([temp_emb, prv_emb, freq_emb])
        session_embeddings = aggregate_session_embeddings(fused_emb, all_session_ids, all_chunk_ids, self.pooling)
        
        if len(session_embeddings) == 0:
            print(f"[DEBUG] Epoch {idx} → No session embeddings!")
            print(f"    session_ids: {all_session_ids}")
            print(f"    chunk_ids: {all_chunk_ids}")
            print(f"    chunk_embeddings shape: {fused_emb.shape}")
        
        # For session_labels, assume that the label for a session is the first label encountered
        session_labels = {}
        for s, lab in zip(all_session_ids, labels_batch):
            if s not in session_labels:
                session_labels[s] = lab
        return session_embeddings, session_labels


    def train(self, data_loader):
        """Training loop with session-level aggregation and visualization."""        
        for epoch in range(self.max_epoch):
            print(f"\n==== Training Epoch: {epoch} ====")
            tbar = tqdm(data_loader["train"], desc=f"Epoch {epoch}", ncols=80)
            self.temporal_branch.train()
            self.prv_branch.train()
            self.freq_branch.train()
            self.classifier.train()
            running_loss = 0.0
            batch_losses = []
            all_logits = []

            # Iterate over training batches
            for idx, batch in enumerate(tbar):
                # [DEBUG] Print batch length
                # print(f"[DEBUG] Epoch {epoch}, Batch {idx}, batch length: {len(batch)}")

                # Reconstruct session-level embeddings and labels
                session_emb_dict, session_label_dict = self.reconstruct_sessions(idx=idx, batch=batch)
                
                # [DEBUG] Print unique labels in the batch
                '''
                if idx % 400 == 0: 
                    unique_labels = set(session_label_dict.values())
                    print(f"[DEBUG] Unique labels in batch {idx}: {unique_labels}")
                '''
                
                # Accumulate loss over sessions in this batch
                batch_loss = 0.0
                num_sessions = 0
                for sess_id, emb in session_emb_dict.items():
                    # [DEBUG] Print session embedding shape``
                    # print(f"[DEBUG] Batch {idx} has no session embeddings! Skipping...")
                    # print(f"[DEBUG] emb shape for session {sess_id}: {emb.shape}")

                    # Prepare input vector for the classifier
                    input_vec = emb.unsqueeze(0).to(self.device)  # (1, fusion_dim)
                    label = session_label_dict[sess_id]
                    # label_tensor = torch.tensor(label).float().unsqueeze(0).to(self.device)
                    '''
                    if label_tensor.dim() == 1:
                        label_tensor = label_tensor.unsqueeze(1)
                    '''
                    label_tensor = torch.tensor([label], dtype=torch.long).to(self.device)

                    # Forward pass through classifier
                    outputs = self.classifier(input_vec)
                    loss = self.criterion(outputs, label_tensor)
                    
                    '''
                    if idx % 00 == 0:
                        print(f"[DEBUG] Session {sess_id} → mean: {emb.mean().item():.4f}, std: {emb.std().item():.4f}")
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
                    '''
                    
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
            '''
            # Print running loss every 100 mini-batches
            if (idx + 1) % 100 == 99:
                avg_loss = running_loss / (idx + 1)
                print(f"[Epoch {epoch}, Batch {idx+1}] Loss: {avg_loss:.3f}")
            '''
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
        self.prv_branch.eval()
        self.freq_branch.eval()
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
                    '''
                    if label_tensor.dim() == 1:
                        label_tensor = label_tensor.unsqueeze(1)
                    '''
                    
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
        self.prv_branch.eval()
        self.freq_branch.eval()
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
