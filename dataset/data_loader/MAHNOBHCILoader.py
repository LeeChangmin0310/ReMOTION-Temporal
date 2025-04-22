"""The dataloader for MAHNOB-HCI dataset.

Details for the MAHNOB-HCI Dataset see http://mahnob-db.eu/hci-tagging/.
If you use this dataset, please cite this paper:
Soleymani, M., Caro, Y. L., Schmidt, P., Sha, C., & Pun, T. (2012). Multimodal emotion recognition using regression and fusion. IEEE transactions on affective computing, 3(4), 429-441.
"""

import os
import re
import glob
from multiprocessing import Process, Manager

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset.data_loader.BaseLoader import BaseLoader

class MAHNOBHCILoader(BaseLoader):
    """The data loader for the MAHNOB-HCI dataset with emotion recognition.
       Each subject may have multiple sessions.
    """

    def __init__(self, name, data_path, config_data, config_mode='neural_methods', device=None):
        """
        Initializes an MAHNOB-HCI dataloader.
        Args:
        Initializes an MAHNOB-HCI dataloader.
        Raw data should be stored in a folder with the following structure:
        Video format: .avi (NDHWC)
        Args:
            data_path (str): Path of the folder storing raw video and emotion data.
                             For example, data_path should be "MAHNOB_HCI_Emotion" with a structure like:
                                MAHNOB_HCI_Emotion/
                                  ├── emotion_labels.csv
                                  ├── 2(Session)/
                                  │     └── 2.avi
                                  └── 4(Session)/
                                        └── 4.avi
            name (str): Name of the dataloader.
            config_data (CfgNode): Data settings (ref: config.py), which should include:
                                   - LABEL_SOURCE (e.g., "CSV" or "PPG")
                                   - LABEL_COLUMN (e.g., "HR" or a list such as ["feltEmo", "feltArsl", "feltVlnc"])
        """
        # Read the emotion labels CSV and build a mapping from session to its info.
        emotion_label_csv = os.path.join(data_path, "emotion_labels.csv")
        self.emotion_label_dict = self.read_emotion_labels(emotion_label_csv)
        self.config_mode = config_mode
        super().__init__(name, data_path, config_data)
        self.group_sessions()

    def read_emotion_labels(self, csv_path):
        """
        Reads the emotion labels CSV.
        Creates a dictionary with session (as string) as key and all info as value.
        """
        if not os.path.exists(csv_path):
            raise ValueError(f"Emotion labels CSV not found at {csv_path}!")
        df = pd.read_csv(csv_path)
        label_dict = {}
        for idx, row in df.iterrows():
            session_key = str(int(row['Session']))
            label_dict[session_key] = {
                "HR": float(row["HR"]),
                "feltEmo": int(row["feltEmo"]),
                "feltArsl": int(row["feltArsl"]),
                "feltVlnc": int(row["feltVlnc"]),
                "subject_id": int(row["subject_id"]),
                "3C_Arsl": int(row["3C_Arsl"]),
                "3C_Vlnc": int(row["3C_Vlnc"]),
                "BC_Arsl": int(row["BC_Arsl"]),
                "BC_Vlnc": int(row["BC_Vlnc"]),
                "stim_start_sec": int(row["stim_start_sec"]),
                "stim_end_sec": int(row["stim_end_sec"]),
            }
        return label_dict
    
    def group_sessions(self):
        """Group the preprocessed file paths by session (extracted from file name)."""
        self.session_files = {}
        # self.inputs is already set in load_preprocessed_data() as a sorted list of file paths.
        for file_path in self.inputs:
            # Extract session id from file name (e.g. "501_input0.npy" -> "501")
            session_id = self.extract_session_id(file_path)
            if session_id not in self.session_files:
                self.session_files[session_id] = []
            self.session_files[session_id].append(file_path)
        # Optionally, sort each session's file list by the chunk index
        for session in self.session_files:
            self.session_files[session] = sorted(
                self.session_files[session],
                key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0].replace('input',''))
            )

        
    # If SPLIT_METHOD is empty, grouping will be done in load_preprocessed_data()

    # --------------------------------------------------------------------------
    # Override load_preprocessed_data() to group files by session if SPLIT_METHOD == ''
    # --------------------------------------------------------------------------
    def load_preprocessed_data(self):
        """
        Loads the preprocessed data from the file list CSV.
        Then, if SPLIT_METHOD is empty, groups files by session.
        """
        file_list_df = pd.read_csv(self.file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        # Sort inputs normally
        self.inputs = sorted(inputs)
        if self.config_data.LABEL_SOURCE == 'CSV':
            if self.config_mode == 'unsupervised':
                labels = []
                for file_path in self.inputs:
                    session_id = self.extract_session_id(file_path)
                    label = self.get_label_by_session(session_id)
                    labels.append(label)
                self.labels = labels
            else:
                self.labels = [input_file.replace("input", "label") for input_file in self.inputs]
        else:
            self.labels = None
        self.preprocessed_data_len = len(self.inputs)

        # Group files by session if SPLIT_METHOD is empty
        if self.config_data.SPLIT_METHOD == '':
            self.session_files = {}
            for file_path in self.inputs:
                session_id = self.extract_session_id(file_path)
                if session_id not in self.session_files:
                    self.session_files[session_id] = []
                self.session_files[session_id].append(file_path)
            # Sort each session's file list by chunk id
            for sess in self.session_files:
                self.session_files[sess] = sorted(
                    self.session_files[sess],
                    key=lambda x: int(os.path.basename(x).split('_')[1].replace("input", "").split('.')[0])
                )
    
    # --------------------------------------------------------------------------
    # Override __len__() to return number of sessions in session-level mode.
    # --------------------------------------------------------------------------
    def __len__(self):
        """
        Returns the length of the dataset.
        If SPLIT_METHOD is empty, returns the number of sessions.
        """
        if self.config_data.SPLIT_METHOD == '':
            return len(self.session_files)
        else:
            return len(self.inputs)
    
    # --------------------------------------------------------------------------
    # Override __getitem__() to return session-level data when SPLIT_METHOD is empty.
    # --------------------------------------------------------------------------
    def __getitem__(self, index):
        """
        Return all chunks within stimulation time range for one session, <- wrong info
        along with the session label and session id.
        """
        session_id = list(self.session_files.keys())[index]
        file_list = self.session_files[session_id]
        '''
        # Get stimulation start/end times (in seconds) and chunk/frame info
        stim_start_sec = self.emotion_label_dict[session_id]["stim_start_sec"]
        stim_end_sec = self.emotion_label_dict[session_id]["stim_end_sec"]
        fps = 30  # after downsampling
        chunk_len = self.config_data.PREPROCESS.CHUNK_LENGTH

        stim_start_frame = stim_start_sec * fps
        stim_end_frame = stim_end_sec * fps

        # Select only chunks that overlap with stim window
        selected_files = []
        for file_path in file_list:
            chunk_id = int(os.path.basename(file_path).split('_')[-1].split('.')[0].replace('input', ''))
            chunk_start_frame = chunk_id * chunk_len
            chunk_end_frame = (chunk_id + 1) * chunk_len

            if chunk_end_frame > stim_start_frame and chunk_start_frame < stim_end_frame:
                selected_files.append(file_path)

        if len(selected_files) == 0:
            raise ValueError(f"No chunks selected in stimulation range for session {session_id}")
    
        # Load selected chunks
        chunk_data = [np.load(fp).copy() for fp in selected_files]
        '''
        chunk_data = [np.load(fp).copy() for fp in file_list]

        # Data format conversion
        if self.data_format == 'NDCHW':
            chunk_data = [np.transpose(d, (0, 3, 1, 2)) for d in chunk_data]
        elif self.data_format == 'NCDHW':
            chunk_data = [np.transpose(d, (3, 0, 1, 2)) for d in chunk_data]
        elif self.data_format == 'NDHWC':
            chunk_data = [np.transpose(d, (3, 0, 1, 2)) for d in chunk_data]
        else:
            raise ValueError('Unsupported Data Format!')
        
        # Convert to float32
        session_data = np.stack([np.float32(d) for d in chunk_data], axis=0)
        
        # Convert numpy array to torch.Tensor here
        session_data = torch.from_numpy(session_data)  # (T, C, H, W) already

        # Load label
        if self.config_data.LABEL_SOURCE == 'CSV':
            label = self.get_label_by_session(session_id)
        else:
            label = None

        # Also return chunk ids if needed
        chunk_ids = [int(os.path.basename(fp).split('_')[-1].split('.')[0].replace('input','')) for fp in file_list]

        return session_data, label, session_id, chunk_ids

        
        
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
    '''
    # Something went wrong wit this code <------------------ Need to be fixed !!
    def get_label_by_session(self, session_id):
        """
        Returns the desired label(s) for the given session_id.
        In this modified version, session_id here represents the subject id (cached file name prefix).
        """
        # Use subject id to retrieve the label from any one of its sessions.
        for info in self.emotion_label_dict.values():
            if str(info['subject_id']) == session_id:
                desired_label = self.config_data.LABEL_COLUMN  # e.g., "HR" or list
                if isinstance(desired_label, list):
                    return {col: info[col] for col in desired_label}
                else:
                    return info[desired_label]
        raise ValueError(f"Emotion label for subject {session_id} not found!")
    '''

    def get_raw_data(self, data_path):
        """Returns raw data directories for MAHNOB-HCI dataset.
           Only returns folders whose session id exists in the emotion label CSV.
        """
        data_dirs = glob.glob(os.path.join(data_path, "*"))
        data_dirs = [d for d in data_dirs if os.path.isdir(d)]
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = []
        for data_dir in data_dirs:
            m = re.search(r'(\d+)', os.path.basename(data_dir))
            if m:
                session_id = m.group(1)
                if session_id in self.emotion_label_dict:
                    dirs.append({"index": session_id, "path": data_dir})
        return dirs

    def get_subject_id(self, session_id):
        """
        Returns the subject_id for the given session_id.
        """
        if session_id in self.emotion_label_dict:
            return self.emotion_label_dict[session_id]['subject_id']
        else:
            raise ValueError(f"Subject ID for session {session_id} not found in emotion labels!")

    def get_all_subject_ids(self):
        """
        Returns a sorted list of all unique subject IDs in the dataset.
        This utility function is useful for subject-level cross-validation.
        """
        subject_ids = {str(info['subject_id']) for info in self.emotion_label_dict.values()}
        return sorted(list(subject_ids))

    # For MAHNOB-HCI emotion recognition, we use simple fraction split 
    # Session(or Subject)-level split is not used here !!!
    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data directories based on fraction split."""
        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        return [data_dirs[i] for i in choose_range]
    
    def chunk_frames(self, data, chunk_length):
        """Splits the video data into chunks along time axis."""
        clip_num = data.shape[0] // chunk_length
        frames_clips = [data[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips)
    
    def preprocess(self, frames, dummy_label, config_preprocess):
        """Override preprocess: process only video frames.
           Args:
               frames(np.array): Video frames.
               dummy_label: Not used.
               config_preprocess(CfgNode): Preprocessing settings.
           Returns:
               frames_clips: Processed video chunks.
               None: Dummy label.
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
        if config_preprocess.DO_CHUNK:
            frames_clips = self.chunk_frames(data, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
        return frames_clips, None

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Processes a single session and saves preprocessed video chunks.
           Invoked by preprocess_dataset for multi-processing.
           Processes raw video into chunks and saves only video chunk files with session ID in filename.
           Implements subject-level caching: if a subject's data is already preprocessed, reuse it.
        """
        if self.config_data.SPLIT_METHOD in ['LOSO', 'KFold']:
            session_id = data_dirs[i]['index']
            subject_id = str(self.get_subject_id(session_id))
            # Set cache directory per subject
            subject_cache_dir = os.path.join(self.cached_path, subject_id)
            
            if os.path.exists(subject_cache_dir) and os.listdir(subject_cache_dir):
                print(f"Subject {subject_id} already preprocessed. Using cached data.")
                input_name_list = glob.glob(os.path.join(subject_cache_dir, f"{subject_id}_input*.npy"))
                file_list_dict[i] = input_name_list
                return
            else:
                os.makedirs(subject_cache_dir, exist_ok=True)
                avi_pattern = os.path.join(data_dirs[i]['path'], "*.avi")
                frames = self.read_video(avi_pattern)
                if session_id not in self.emotion_label_dict:
                    raise ValueError(f"Emotion label for session {session_id} not found!")
                dummy_label = None
                frames_clips, _ = self.preprocess(frames, dummy_label, config_preprocess)
                input_name_list = []
                count = 0
                for clip in frames_clips:
                    file_path = os.path.join(subject_cache_dir, f"{subject_id}_input{count}.npy")
                    np.save(file_path, clip)
                    input_name_list.append(file_path)
                    count += 1
                file_list_dict[i] = input_name_list
                
        else:
            saved_filename = data_dirs[i]['index']  # session ID as a string.
            avi_pattern = os.path.join(data_dirs[i]['path'], "*.avi")
            frames = self.read_video(avi_pattern)
            
            if saved_filename not in self.emotion_label_dict:
                raise ValueError(f"Emotion label for session {saved_filename} not found in emotion labels!")
            info = self.emotion_label_dict[saved_filename]
            
            # print(f"Preprocessing session {saved_filename}: HR = {info['HR']}")
            # Instead of duplicating label, set dummy_label to None.
            dummy_label = None

            # Preprocess: Only process frames.
            frames_clips, _ = self.preprocess(frames, dummy_label, config_preprocess)
            # Save only the video chunks; do not save label files.
            input_name_list = self.save_multi_process(frames_clips, None, saved_filename)
            file_list_dict[i] = input_name_list
            
            

    def save_multi_process(self, frames_clips, dummy_label, filename):
        """Saves preprocessed video chunks.
           In this case, filename is the subject_id.
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        for clip in frames_clips:
            input_path_name = os.path.join(self.cached_path, f"{filename}_input{count}.npy")
            np.save(input_path_name, clip)
            input_path_name_list.append(input_path_name)
            count += 1
        return input_path_name_list

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """Builds a file list based on subject-level cached data.
           Instead of using session ids, it groups by subject_id.
        """
        # Get unique subject ids from the raw data directories
        subject_ids = set()
        for d in data_dirs:
            session_id = d['index']
            subject_ids.add(str(self.get_subject_id(session_id)))
        file_list = []
        for subject_id in subject_ids:
            subject_cache_dir = os.path.join(self.cached_path, subject_id)
            subject_files = glob.glob(os.path.join(subject_cache_dir, f"{subject_id}_input*.npy"))
            file_list += subject_files
        if not file_list:
            raise ValueError(self.dataset_name, 'File list empty. Check preprocessed data folder exists and is not empty.')
        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)

    def load_preprocessed_data(self):
        """Loads the preprocessed data from the file list.
           Subject id is extracted from the file name.
        """
        file_list_df = pd.read_csv(self.file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        self.inputs = sorted(inputs)
        if self.config_data.LABEL_SOURCE == 'CSV':
            if self.config_mode == 'unsupervised':
                labels = []
                for file_path in self.inputs:
                    session_id = self.extract_session_id(file_path)
                    label = self.get_label_by_session(session_id)
                    labels.append(label)
                self.labels = labels
            else:
                self.labels = [input_file.replace("input", "label") for input_file in self.inputs]
        else:
            self.labels = None
        self.preprocessed_data_len = len(inputs)

    @staticmethod
    def read_video(video_pattern):
        """Reads a video file matching the given pattern and returns frames (T, H, W, 3).
        Resamples to 30 fps if necessary.
        """
        video_files = glob.glob(video_pattern)
        if not video_files:
            raise ValueError(f"No video files found matching pattern {video_pattern}!")
        
        video_file = video_files[0]
        VidObj = cv2.VideoCapture(video_file)
        
        # Get original FPS of the video
        orig_fps = VidObj.get(cv2.CAP_PROP_FPS)
        target_fps = 30.0

        # Get total frame count and video duration
        frame_count = int(VidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / orig_fps if orig_fps > 0 else 0
        target_frame_count = int(duration_sec * target_fps)

        frames = []

        if orig_fps == target_fps:
            # If FPS matches target, read frames directly
            success, frame = VidObj.read()
            while success:
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frames.append(frame)
                success, frame = VidObj.read()
        else:
            # If FPS differs, resample using interpolation
            all_frames = []
            success, frame = VidObj.read()
            while success:
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                all_frames.append(frame)
                success, frame = VidObj.read()

            # Generate new indices for 30 fps
            orig_indices = np.linspace(0, len(all_frames) - 1, num=target_frame_count).astype(np.int32)
            frames = [all_frames[idx] for idx in orig_indices]

        return np.asarray(frames)


    @staticmethod
    def read_wave(bvp_file):
        """Reads a BVP signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read().split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)
