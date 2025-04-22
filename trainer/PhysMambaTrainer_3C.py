"""PhysMamba Trainer."""
import os
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.optim as optim
import random
from evaluation.metrics import calculate_metrics
from neural_encoders.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_encoders.model.PhysMamba import PhysMamba
from trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
from scipy.signal import welch
from scipy.signal import butter, filtfilt, find_peaks


class PhysMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        self.frame_rate = config.TRAIN.DATA.FS

        self.model = PhysMamba().to(self.device)  # [3, T, 128,128]
        if self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.criterion_Pearson = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay = 0.0005)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.criterion_Pearson_test = Neg_Pearson()
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            self.model.train()
            loss_rPPG_avg = []
            running_loss = 0.0
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)

                pred_ppg = (pred_ppg-torch.mean(pred_ppg, axis=-1).view(-1, 1))/torch.std(pred_ppg, axis=-1).view(-1, 1)    # normalize
                
                labels = (labels - torch.mean(labels)) / \
                            torch.std(labels)
                loss = self.criterion_Pearson(pred_ppg, labels)

                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                self.optimizer.step()
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item())
            
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
            torch.cuda.empty_cache()
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss)) 
        
    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)
                rPPG = self.model(
                    valid_batch[0].to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)  # normalize
                loss_ecg = self.criterion_Pearson(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)
    
    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test = self.model(data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            self.save_test_outputs(predictions, labels, self.config)
    '''
    # HR MAE and RMSE Test code
    # if you want to use this code, you need need GT HR values from ECG of MANOB-HCI dataset
    def test(self, data_loader):
        """Runs the model on the test set and computes HR metrics by comparing the predicted HR (from rPPG) with the ground truth HR.
        The ground truth HR is obtained from the emotion_labels_updated.csv in MAHNOBHCILoader."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()  # Dictionary to store rPPG segments for each subject
        gt_hr_values = dict()  # Dictionary to store ground truth HR for each subject

        # Load model weights based on the toolbox mode
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))
        
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        
        with torch.no_grad():
            # Iterate over test batches
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data = test_batch[0].to(self.config.DEVICE)
                # The label from MAHNOBHCILoader is a single HR value per video
                # label = torch.tensor(test_batch[1]).to(self.config.DEVICE)
                label = test_batch[1].to(self.config.DEVICE) # Ground truth HR values from ECG
                # print(label)
                pred_ppg_test = self.model(data)  # Model outputs rPPG signals
                
                # If output saving is enabled, move tensors to CPU
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()
                
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]  # Subject identifier (e.g., "34")
                    # Store the ground truth HR (single value) for this subject if not stored yet
                    if subj_index not in gt_hr_values:
                        gt_hr_values[subj_index] = label[idx].item()
                    # Collect predicted rPPG signals from different segments for the same subject
                    if subj_index not in predictions:
                        predictions[subj_index] = []
                    predictions[subj_index].append(pred_ppg_test[idx])
        
        # Compute predicted HR for each subject by concatenating all rPPG segments and applying get_hr()
        pred_hr_dict = {}
        for subj, ppg_segments in predictions.items():
            # Concatenate segments along the time axis
            ppg_signal = torch.cat(ppg_segments, dim=0).cpu().numpy()
            # Use get_hr() to extract HR from the rPPG signal
            pred_hr = self.get_hr(ppg_signal, sr=self.config.TEST.DATA.FS)
            pred_hr_dict[subj] = pred_hr
        
        # Convert dictionaries to arrays for metric calculation
        pred_hr_arr = np.array(list(pred_hr_dict.values()))
        gt_hr_arr = np.array(list(gt_hr_values.values()))
        
        # Compute evaluation metrics (e.g., MAE, RMSE)
        mae = np.mean(np.abs(pred_hr_arr - gt_hr_arr))
        rmse = np.sqrt(np.mean((pred_hr_arr - gt_hr_arr) ** 2))
        
        print("Test HR MAE: {:.2f}".format(mae))
        print("Test HR RMSE: {:.2f}".format(rmse))
        
        # Optionally, generate Bland–Altman plots or compute additional metrics.
        # Example (uncomment and adjust if needed):
        # from evaluation.BlandAltmanPy import BlandAltman
        # ba = BlandAltman(gt_hr_arr, pred_hr_arr, self.config, averaged=True)
        # ba.scatter_plot(
        #     x_label='GT HR [bpm]',
        #     y_label='rPPG HR [bpm]',
        #     show_legend=True, figure_size=(5, 5),
        #     the_title='BlandAltman_ScatterPlot',
        #     file_name='BlandAltman_ScatterPlot.pdf')
        # ba.difference_plot(
        #     x_label='Difference [bpm]',
        #     y_label='Average [bpm]',
        #     show_legend=True, figure_size=(5, 5),
        #     the_title='BlandAltman_DifferencePlot',
        #     file_name='BlandAltman_DifferencePlot.pdf')

        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, gt_hr_values, self.config)
    
    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    # HR calculation based on ground truth label
    def bandpass_filter(self, signal, fs=30, lowcut=0.5, highcut=4, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, signal)

    def get_hr(self, rppg_signal, sr=30, min_hr=30, max_hr=180):
        """Extracts HR from rPPG signal using peak detection."""
        # 1. Remove DC component
        rppg_signal = rppg_signal - np.mean(rppg_signal)
        # 2. Bandpass filter(0.5-4Hz)
        filtered_signal = self.bandpass_filter(rppg_signal, fs=sr, lowcut=0.5, highcut=4, order=4)
        # 3. Peak detection
        peaks, _ = find_peaks(filtered_signal, distance=sr*0.6, height=np.mean(filtered_signal) + np.std(filtered_signal))
        if len(peaks) < 2:
            return np.nan
        # 4. Calculate HR
        rr_intervals = np.diff(peaks) / sr
        # HR = 60 / RR
        hr_values = 60 / rr_intervals
        # Filter HR values outside the range [min_hr, max_hr]
        hr_values = hr_values[(hr_values >= min_hr) & (hr_values <= max_hr)]
        if len(hr_values) == 0:
            return np.nan
        return np.mean(hr_values)
    '''
    # original HR calc code
    def get_hr(self, y, sr=30, min=30, max=180):
        p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
        return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
