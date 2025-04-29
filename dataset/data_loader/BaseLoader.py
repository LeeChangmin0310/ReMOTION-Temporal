"""
The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported: UBFC-rPPG, PURE, SCAMPS, BP4D+, UBFC-PHYS and MAHNOB-HCI.
"""

import csv
import glob
import os
import re
from math import ceil
from scipy import signal
from scipy import sparse
from unsupervised_extractors.methods import POS_WANG
from unsupervised_extractors import utils
import math
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from retinaface import RetinaFace   # Source code: https://github.com/serengil/retinaface


class BaseLoader(Dataset):
    """The base class for data loading based on pytorch Dataset.

    The dataloader supports both providing data for pytorch training and common data-preprocessing methods,
    including reading files, resizing each frame, chunking, and video-signal synchronization.
    """

    @staticmethod
    def add_data_loader_args(parser):
        """Adds arguments to parser for training process"""
        parser.add_argument("--cached_path", default=None, type=str)
        parser.add_argument("--preprocess", default=None, action='store_true')
        return parser

    def __init__(self, dataset_name, raw_data_path, config_data):
        """Inits dataloader with lists of files.

        Args:
            dataset_name(str): name of the dataloader.
            raw_data_path(string): path to the folder containing all data.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.inputs = list()
        self.labels = list()
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        self.config_data = config_data
        
        if hasattr(config_data, 'SPLIT_METHOD') and config_data.SPLIT_METHOD in ["LOSO", "KFold"]:
            begin, end = 0.0, 1.0
        else:
            begin, end = config_data.BEGIN, config_data.END
        
        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN >= 0)
        assert (config_data.END <= 1)
        if config_data.DO_PREPROCESS:
            self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
            self.preprocess_dataset(self.raw_data_dirs, config_data.PREPROCESS, begin, end)
        else:
            if not os.path.exists(self.cached_path):
                print('CACHED_PATH:', self.cached_path)
                raise ValueError(self.dataset_name, 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
                self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
                self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN, config_data.END)
                print('File list generated.', end='\n\n')
            self.load_preprocessed_data()
        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f"{self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, index):
        """Returns a clip of video (3, T, W, H) and its corresponding signal (T)."""
        # Debug: print the file path stored in self.inputs[index]
        '''file_path = self.inputs[index]
        print("Loading file:", file_path)'''
        data = np.load(self.inputs[index]).copy()
        
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            data = np.transpose(data, (3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)
        else:
            raise ValueError('Unsupported Data Format!')
        data = np.float32(data)

        # Extract the subject ID from the input file name (now stored as subject_id in file prefix)
        session_id = self.extract_session_id(self.inputs[index])
        if self.config_data.LABEL_SOURCE == 'CSV':
            label = self.get_label_by_session(session_id)  # Get label based on subject id
        else:
            label = np.load(self.labels[index]).copy()
            label = np.float32(label)
        
        # Get the filename and chunk id from the file path
        item_path = self.inputs[index]
        item_path_filename = os.path.basename(item_path)
        split_idx = item_path_filename.rindex('_')
        
        # Convert filename and chunk_id explicitly to string
        filename = str(item_path_filename[:split_idx])
        chunk_id = str(item_path_filename[split_idx + 6:].split('.')[0])
        return data, label, filename, chunk_id

    def extract_session_id(self, file_path):
        """
        Extract Session ID (or in our modified case, subject ID) from file path.
        e.g., "/path/to/501_input0.npy" -> "501"
        """
        file_name = os.path.basename(file_path)
        session_id = file_name.split("_")[0]
        return session_id

    def get_label_by_session(self, session_id):
        """
        Returns the desired label(s) for the given session_id based on configuration.
        If config_data.LABEL_COLUMN is a string, returns that column's value.
        If it's a list, returns a dictionary with each specified column.
        """
        if session_id in self.emotion_label_dict:
            label_info = self.emotion_label_dict[session_id]
            desired_label = self.config_data.LABEL_COLUMN  # e.g., "HR" or ["feltEmo", "feltArsl", "feltVlnc"]
            if isinstance(desired_label, list):
                return {col: label_info[col] for col in desired_label}
            else:
                return label_info[desired_label]
        else:
            raise ValueError(f"Emotion label for session {session_id} not found!")

    def get_raw_data(self, raw_data_path):
        """
        Returns raw data directories under the path.
        Args:
            raw_data_path(str): Path to raw video data.
        """
        raise Exception("'get_raw_data' Not Implemented")
    
    def get_subject_id(self, session_id):
        """
        Returns the subject ID corresponding to the given session_id.
        This method should be overridden in the derived class.
        """
        raise NotImplementedError("Please implement get_subject_id in the derived class")

    def split_raw_data(self, data_dirs, begin, end):
        """
        Returns a subset of data directories based on the cross-validation strategy.
        For subject-level CV, groups raw data by unique subject IDs.
        """
        # If using fraction-based split (i.e., no subject-level CV)
        if not hasattr(self.config_data, 'SPLIT_METHOD') or self.config_data.SPLIT_METHOD == "fraction":
            file_num = len(data_dirs)
            choose_range = range(int(begin * file_num), int(end * file_num))
            return [data_dirs[i] for i in choose_range]
        
        # For subject-level CV (LOSO or k_fold)
        subject_dict = {}
        for d in data_dirs:
            session_id = d['index']
            subject_id = self.get_subject_id(session_id)  # must be implemented in derived class
            if subject_id not in subject_dict:
                subject_dict[subject_id] = []
            subject_dict[subject_id].append(d)
        
        unique_subjects = sorted(list(subject_dict.keys()))
        cv_strategy = self.config_data.SPLIT_METHOD  # 'LOSO' or 'k_fold'
        
        if cv_strategy == "LOSO":
            test_subject = self.config_data.get('current_subject', None)
            if test_subject is None:
                raise ValueError("For LOSO, 'current_subject' must be specified in config_data.")
            if begin == 0 and end < 1:
                train_subjects = [s for s in unique_subjects if s != test_subject]
                split_data = []
                for s in train_subjects:
                    split_data.extend(subject_dict[s])
                return split_data
            else:
                return subject_dict[test_subject]
        
        elif cv_strategy == "KFold":
            n_folds = self.config_data.NUM_FOLDS
            current_fold = self.config_data.FOLD_INDEX
            folds = {i: [] for i in range(n_folds)}
            for idx, s in enumerate(unique_subjects):
                fold_idx = idx % n_folds
                folds[fold_idx].append(s)
            if begin == 0 and end < 1:
                train_subjects = [s for i, subs in folds.items() if i != current_fold for s in subs]
                split_data = []
                for s in train_subjects:
                    split_data.extend(subject_dict[s])
                return split_data
            else:
                test_subjects = folds[current_fold]
                split_data = []
                for s in test_subjects:
                    split_data.extend(subject_dict[s])
                return split_data
        else:
            raise ValueError("Unsupported SPLIT_METHOD in config_data!")

    def read_npy_video(self, video_file):
        """Reads a video file in numpy format (.npy) and returns frames (T, H, W, 3)."""
        frames = np.load(video_file[0])
        if np.issubdtype(frames.dtype, np.integer) and np.min(frames) >= 0 and np.max(frames) <= 255:
            processed_frames = [frame.astype(np.uint8)[..., :3] for frame in frames]
        elif np.issubdtype(frames.dtype, np.floating) and np.min(frames) >= 0.0 and np.max(frames) <= 1.0:
            processed_frames = [(np.round(frame * 255)).astype(np.uint8)[..., :3] for frame in frames]
        else:
            raise Exception(f'Loaded frames are of an incorrect type or range! Received type {frames.dtype} with range {np.min(frames)} to {np.max(frames)}.')
        return np.asarray(processed_frames)

    def generate_pos_psuedo_labels(self, frames, fs=30):
        """Generated POS-based PPG Psuedo Labels For Training
        Args:
            frames(List[array]): video frames.
            fs(int or float): Sampling rate of video.
        Returns:
            env_norm_bvp: Hilbert envelope normalized POS PPG signal.
        """
        WinSec = 1.6
        RGB = POS_WANG._process_video(frames)
        N = RGB.shape[0]
        H = np.zeros((1, N))
        l = math.ceil(WinSec * fs)
        for n in range(N):
            m = n - l
            if m >= 0:
                Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
                Cn = np.mat(Cn).H
                S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
                h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
                mean_h = np.mean(h)
                for temp in range(h.shape[1]):
                    h[0, temp] = h[0, temp] - mean_h
                H[0, m:n] = H[0, m:n] + (h[0])
        bvp = H
        bvp = utils.detrend(np.mat(bvp).H, 100)
        bvp = np.asarray(np.transpose(bvp))[0]
        min_freq = 0.70
        max_freq = 3
        b, a = signal.butter(2, [(min_freq) / fs * 2, (max_freq) / fs * 2], btype='bandpass')
        pos_bvp = signal.filtfilt(b, a, bvp.astype(np.double))
        analytic_signal = signal.hilbert(pos_bvp)
        amplitude_envelope = np.abs(analytic_signal)
        env_norm_bvp = pos_bvp / amplitude_envelope
        return np.array(env_norm_bvp)

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Parses and preprocesses raw data based on split.
        Args:
            data_dirs(List[str]): list of video directories.
            config_preprocess(CfgNode): preprocessing settings.
            begin(float): start fraction.
            end(float): end fraction.
        """
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)
        file_list_dict = self.multi_process_manager(data_dirs_split, config_preprocess)
        self.build_file_list(file_list_dict)
        self.load_preprocessed_data()
        print("Total Number of raw files preprocessed:", len(data_dirs_split), end='\n\n')

    def preprocess(self, frames, bvps, config_preprocess):
        """Preprocesses video and label data.
        Args:
            frames(np.array): Video frames.
            bvps(np.array): Label signal.
            config_preprocess(CfgNode): preprocessing settings.
        Returns:
            frames_clips(np.array): Preprocessed video chunks.
            bvps_clips(np.array): Preprocessed label chunks.
        """
        frames = self.crop_face_resize(
            frames,
            config_preprocess.CROP_FACE.DO_CROP_FACE,
            config_preprocess.CROP_FACE.BACKEND,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
            config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H)
        data = []
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")
        if config_preprocess.DO_CHUNK:
            frames_clips, bvps_clips = self.chunk(data, bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])
        return frames_clips, bvps_clips

    def face_detection(self, frame, backend, use_larger_box=False, larger_box_coef=1.0):
        """Performs face detection on a single frame.
        Args:
            frame(np.array): Single video frame.
            backend(str): Face detection backend.
            use_larger_box(bool): Whether to enlarge the bounding box.
            larger_box_coef(float): Enlargement coefficient.
        Returns:
            face_box_coor(List[int]): Coordinates of the face bounding box.
        """
        if backend == "HC":
            detector = cv2.CascadeClassifier('./dataset/haarcascade_frontalface_default.xml')
            face_zone = detector.detectMultiScale(frame)
            if len(face_zone) < 1:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
            elif len(face_zone) >= 2:
                max_width_index = np.argmax(face_zone[:, 2])
                face_box_coor = face_zone[max_width_index]
                print("Warning: Multiple faces detected. Using the largest one.")
            else:
                face_box_coor = face_zone[0]
        elif backend == "RF":
            res = RetinaFace.detect_faces(frame)
            if len(res) > 0:
                highest_score_face = max(res.values(), key=lambda x: x['score'])
                face_zone = highest_score_face['facial_area']
                x_min, y_min, x_max, y_max = face_zone
                x = x_min
                y = y_min
                width = x_max - x_min
                height = y_max - y_min
                center_x = x + width // 2
                center_y = y + height // 2
                square_size = max(width, height)
                new_x = center_x - (square_size // 2)
                new_y = center_y - (square_size // 2)
                face_box_coor = [new_x, new_y, square_size, square_size]
            else:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        else:
            raise ValueError("Unsupported face detection backend!")
        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        return face_box_coor

    def crop_face_resize(self, frames, use_face_detection, backend, use_larger_box, larger_box_coef, use_dynamic_detection, 
                         detection_freq, use_median_box, width, height):
        """Crops face and resizes frames.
        Args:
            frames(np.array): Video frames.
            use_dynamic_detection(bool): Whether to perform dynamic detection.
            detection_freq(int): Frequency of detection.
            width(int): Target width.
            height(int): Target height.
            use_larger_box(bool): Whether to enlarge the detected box.
            use_face_detection(bool): Whether to perform face detection.
            larger_box_coef(float): Enlargement coefficient.
        Returns:
            resized_frames(list[np.array]): Cropped and resized frames.
        """
        if use_dynamic_detection:
            num_dynamic_det = ceil(frames.shape[0] / detection_freq)
        else:
            num_dynamic_det = 1
        face_region_all = []
        for idx in range(num_dynamic_det):
            if use_face_detection:
                face_region_all.append(self.face_detection(frames[detection_freq * idx], backend, use_larger_box, larger_box_coef))
            else:
                face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
        face_region_all = np.asarray(face_region_all, dtype='int')
        if use_median_box:
            face_region_median = np.median(face_region_all, axis=0).astype('int')
        resized_frames = np.zeros((frames.shape[0], height, width, 3))
        for i in range(frames.shape[0]):
            frame = frames[i]
            reference_index = i // detection_freq if use_dynamic_detection else 0
            if use_face_detection:
                face_region = face_region_median if use_median_box else face_region_all[reference_index]
                frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                              max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
            resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return resized_frames

    def chunk(self, frames, bvps, chunk_length):
        """Chunks the data into segments.
        Args:
            frames(np.array): Video frames.
            bvps(np.array): Label signal.
            chunk_length(int): Length of each chunk.
        Returns:
            frames_clips: List of video chunks.
            bvp_clips: List of label chunks.
        """
        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)

    def save(self, frames_clips, bvps_clips, filename):
        """Saves the chunked data.
        Args:
            frames_clips(np.array): Video chunks.
            bvps_clips(np.array): Label chunks.
            filename: Base filename.
        Returns:
            count: Number of files saved.
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        for i in range(len(bvps_clips)):
            input_path_name = os.path.join(self.cached_path, f"{filename}_input{count}.npy")
            label_path_name = os.path.join(self.cached_path, f"{filename}_label{count}.npy")
            self.inputs.append(input_path_name)
            self.labels.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return count

    def save_multi_process(self, frames_clips, bvps_clips, filename):
        """Saves chunked data using multi-processing.
        Args:
            frames_clips(np.array): Video chunks.
            bvps_clips(np.array): Label chunks.
            filename: Base filename.
        Returns:
            input_path_name_list: List of saved file paths.
            label_path_name_list: List of saved label file paths.
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        for i in range(len(bvps_clips)):
            input_path_name = os.path.join(self.cached_path, f"{filename}_input{count}.npy")
            label_path_name = os.path.join(self.cached_path, f"{filename}_label{count}.npy")
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return input_path_name_list, label_path_name_list

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=8):
        """Distributes preprocessing tasks across multiple processes.
        Args:
            data_dirs(List[str]): List of video directories.
            config_preprocess(Dict): Preprocessing settings.
            multi_process_quota(Int): Maximum number of processes.
        Returns:
            file_list_dict(Dict): Dictionary of processed file paths.
        """
        print('Preprocessing dataset...')
        file_num = len(data_dirs)
        choose_range = range(file_num)
        pbar = tqdm(list(choose_range))
        manager = Manager()
        file_list_dict = manager.dict()
        p_list = []
        running_num = 0
        for i in choose_range:
            process_flag = True
            while process_flag:
                if running_num < multi_process_quota:
                    p = Process(target=self.preprocess_dataset_subprocess, 
                                args=(data_dirs, config_preprocess, i, file_list_dict))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()
        return file_list_dict

    def build_file_list(self, file_list_dict):
        """Builds and saves the file list used by the dataloader.
        Args:
            file_list_dict(Dict): Dictionary of processed file paths.
        """
        file_list = []
        for process_num, file_paths in file_list_dict.items():
            file_list += file_paths
        if not file_list:
            raise ValueError(self.dataset_name, 'No files in file list')
        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """If a file list does not exist, builds one based on preprocessed data.
        Args:
            data_dirs(List[str]): List of video directories.
            begin(float): Start fraction.
            end(float): End fraction.
        """
        # This version should be overridden for subject-level caching
        raise NotImplementedError("Please implement build_file_list_retroactive in the derived class")

    def load_preprocessed_data(self):
        """Loads the preprocessed data based on the file list."""
        file_list_df = pd.read_csv(self.file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        self.inputs = sorted(inputs)
        if self.config_data.LABEL_SOURCE == 'CSV':
            if self.config_data.get('config_mode', None) == 'unsupervised':
                labels = []
                for file_path in self.inputs:
                    subject_id = self.extract_session_id(file_path)
                    label = self.get_label_by_session(subject_id)
                    labels.append(label)
                self.labels = labels
            else:
                self.labels = [input_file.replace("input", "label") for input_file in self.inputs]
        else:
            self.labels = None
        self.preprocessed_data_len = len(inputs)

    @staticmethod
    def diff_normalize_data(data):
        """Calculates discrete difference along time and normalizes by standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j] = (data[j + 1] - data[j]) / (data[j + 1] + data[j] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data

    @staticmethod
    def diff_normalize_label(label):
        """Calculates discrete difference of label signal and normalizes by standard deviation."""
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label

    @staticmethod
    def standardized_data(data):
        """Z-score standardization for video data."""
        data = (data - np.mean(data)) / np.std(data)
        data[np.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = (label - np.mean(label)) / np.std(label)
        label[np.isnan(label)] = 0
        return label

    @staticmethod
    def resample_ppg(input_signal, target_length):
        """Resamples a PPG sequence to a target length."""
        return np.interp(
            np.linspace(1, input_signal.shape[0], target_length),
            np.linspace(1, input_signal.shape[0], input_signal.shape[0]),
            input_signal
        )