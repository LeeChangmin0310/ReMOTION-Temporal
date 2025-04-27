# From Raw rPPG Chunks to Emotion Recognition: Learning Temporal Representations via Exploration and Exploitation

This project **extends** the open-source [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox),
creating a **phase-aware, end-to-end representation-focused emotion recognition pipeline**.

---

# 1. 📈 Final Training Pipeline Overview

| Phase | Input ➔ Module Flow | Attention Type | Loss Used | Purpose | Key Modules | Loss Weight |
|:-----:|:----------------------|:--------------|:---------|:--------|:------------|:------------|
| Phase 0 (epoch 0–9) | rPPG ➔ TemporalBranch ➔ AttnScorer ➔ Softmax ➔ Projection ➔ SupConLoss | Softmax (w/ temperature) | SupConLoss, SparsityLoss | Early exploration (diversity) | TemporalBranch, AttnScorer, ProjectionHead | `contrastive_weight`, `sparsity_weight` |
| Phase 0 (epoch 10–19) | rPPG ➔ TemporalBranch ➔ AttnScorer ➔ α-Entmax (α 1.9→1.6) ➔ Projection ➔ SupConLoss | α-Entmax | SupConLoss, SparsityLoss | Sparse attention learning | TemporalBranch, AttnScorer, ProjectionHead | `contrastive_weight`, `sparsity_weight` |
| Phase 1 (epoch 20–34) | rPPG ➔ TemporalBranch ➔ AttnScorer ➔ Entmax15 ➔ Top-K Selection ➔ ChunkAuxClassifier | Entmax15 | Chunk-level CE | Chunk-level discriminativity | TemporalBranch, AttnScorer, ChunkAuxClassifier | `chunk_ce_weight` |
| Phase 2 (epoch 35–end) | rPPG ➔ TemporalBranch ➔ AttnScorer ➔ Softmax ➔ GatedPooling ➔ Classifier | Softmax | Session-level CE | Stable final classification | TemporalBranch, AttnScorer, GatedPooling, Classifier | `ce_weight` |

---

# 2. 🚀 Phase-by-Phase Learning Flow

## Phase 0: Diversity-first Exploration

- **Epoch 0–9**:
  - Attention: Softmax
  - Loss: SupConLoss, SparsityLoss
  - Goal: Maximize embedding diversity
- **Epoch 10–19**:
  - Attention: α-Entmax (α → 1.6)
  - Loss: SupConLoss, SparsityLoss
  - Goal: Sparse, selective attention focusing

---

## Phase 1: Chunk-level Discriminativity (Weak Supervision)

- **Epoch 20–34**:
  - Attention: Entmax15
  - Action: Top-K selection based on raw scores
  - Loss: Chunk-level CE Loss
  - Goal: Train discriminative chunk embeddings

---

## Phase 2: Session-level Final Classification

- **Epoch 35–end**:
  - Attention: Softmax
  - Aggregation: GatedPooling
  - Loss: Session-level CE Loss
  - Goal: Stable final classification

---

# 3. 🧬 Model Architecture Flow

```
rPPG (chunk)
    ➔ TemporalBranch (Multi-Scale Temporal Feature Extractor)
    ➔ AttnScorer (Chunk Attention Scorer)
    ➔
    ├── (Phase 0,1) ➔ Attention-Weighted Projection ➔ SupConLoss
    ├── (Phase 1) ➔ Top-K Selection ➔ ChunkAuxClassifier ➔ Chunk-level CE
    └── (Phase 2) ➔ GatedPooling ➔ Session-level Classifier ➔ Session-level CE
```

---

# 4. 🛠️ Key Modules

| Module | Role | Description |
|:------:|:----:|:-----------|
| TemporalBranch | Feature Extraction | Multi-scale temporal feature extractor for rPPG chunks |
| AttnScorer | Attention Scorer | Computes attention weights per chunk |
| ProjectionHead | Projection for SupCon | Projects embeddings before contrastive learning |
| ChunkAuxClassifier | Chunk-level Classifier | Weak supervision on Top-K chunks |
| GatedPooling | Session Aggregator | Soft attention aggregation for session embedding |
| Classifier | Session Classifier | Final emotion prediction from session embedding |

---

# 5. 🌟 Loss Functions

| Loss | Phase | Purpose |
|:----:|:-----:|:-------|
| SupConLossTopK | Phase 0 | Train diverse and separable representations |
| SparsityLoss (Entropy) | Phase 0 | Regularize attention entropy |
| Chunk-level CE Loss | Phase 1 | Supervise chunk-level discriminativity |
| Session-level CE Loss | Phase 2 | Final session-level emotion classification |

---

# 6. 📂 Dataset

We mainly use **MAHNOB-HCI** fro Emotion Recognition:

- **Data Format**:
  - Videos: `.avi` (NDHWC)
  - Labels: `emotion_labels.csv`

```
MAHNOB_HCI_Emotion/
  ├─ emotion_labels.csv
  ├─ 2/
  │    └─ 2.avi
  ├─ 4/
  │    └─ 4.avi
  └─ ...
```

> **Dataset loading handled by `MAHNOBHCILoader`**
> Data engineering might be needed depends on your emotion recognition task
  - refer EDAandFiltering.ipynb

---

# 7. 📝 Citation

If you use this work, cite **rPPG-Toolbox**:

```bibtex
@article{
}
```
If you find our [paper]() or this toolbox useful for your research, please cite our work.

---

# 8. :wrench: Setup

You can use either [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [`uv`](https://docs.astral.sh/uv/getting-started/installation/) with this toolbox. Most users are already familiar with `conda`, but `uv` may be a bit less familiar - check out some highlights about `uv` [here](https://docs.astral.sh/uv/#highlights). If you use `uv`, it's highly recommended you do so independently of `conda`, meaning you should make sure you're not installing anything in the base `conda` environment or any other `conda` environment. If you're having trouble making sure you're not in your base `conda` environment, try setting `conda config --set auto_activate_base false`.

STEP 1: `bash setup.sh conda` or `bash setup.sh uv` 

STEP 2: `conda activate remotion` or, when using `uv`, `source .venv/bin/activate`

NOTE: the above setup should work without any issues on machines using Linux or MacOS. If you run into compiler-related issues using `uv` when installing tools related to mamba, try checking to see if `clang++` is in your path using `which clang++`. If nothing shows up, you can install `clang++` using `sudo apt-get install clang` on Linux or `xcode-select --install` on MacOS.

If you use Windows or other operating systems, consider using [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) and following the steps within `setup.sh` independently.

---

## :computer: Example of Using Pre-trained rPPG Models 

Please use config files under `./configs/infer_configs`

For example, if you want to run The model pre-trained on UBFC-rPPG and train on MAHNOB-HCI, use `python main.py --config_file ./configs/infer_configs/python main.py --config ./configs/train_configs/Arsl_BC_Normal_PHYSMAMBA.yaml`

**(will be updated)**
If you want to test unsupervised signal processing  methods, you can use `python main.py --config_file ./configs/infer_configs/UBFC-rPPG_UNSUPERVISED.yaml`

## :computer: Examples of Neural Network Training

Please use config files under `./configs/train_configs`

### Training on MAHNOB-HCI

STEP 1: Download the MAHNOB-HCI raw data by asking the [link](http://mahnob-db.eu/hci-tagging).

STEP 2: Modify `./configs/train_configs/Arsl_BC_Normal_PHYSMAMBA.yaml` 

STEP 3: Run `python main.py --config_file ./configs/train_configs/Arsl_BC_Normal_PHYSMAMBA.yaml` 

Note 1: Preprocessing requires only once; thus turn it off on the yaml file when you train the network after the first time. 

Note 2: The example yaml setting will allow 80% of MAHNOB-HCI to train and 10% of MAHNOB-HCI to valid. 
After training, it will use the best model(with the least validation loss) to test on MAHNOB-HCI.

## :zap: Inference With Unsupervised Methods **(will be updated)**

STEP 1: Download the MAHNOB-HCI raw data by asking the [link](http://mahnob-db.eu/hci-tagging).

STEP 2: Modify `./configs/infer_configs/MAHNOB-HCI_UNSUPERVISED_BC.yaml` 

STEP 3: Run `python main.py --config_file ./configs/infer_configs/MAHNOB-HCI_UNSUPERVISED_BC.yaml`

---

# 9. 📝 License

This work inherits the [Responsible AI License](https://www.licenses.ai/source-code-license)
from the original rPPG-Toolbox.

---


# + Additional Informations

## :notebook: Algorithms
***This repo currently supports the following algorithms as a feature extractor:***

* Supervised Neural Algorithms 
  - [PhysMamba: Efficient Remote Physiological Measurement with SlowFast Temporal Difference Mamba](https://doi.org/10.48550/arXiv.2409.12031), by Luo *et al.*, 2024

This repo **will(or can) supports** the following algorithms as a feature extractor:

* Traditional Unsupervised Algorithms
  - [Remote plethysmographic imaging using ambient light (GREEN)](https://pdfs.semanticscholar.org/7cb4/46d61a72f76e774b696515c55c92c7aa32b6.pdf?_gl=1*1q7hzyz*_ga*NTEzMzk5OTY3LjE2ODYxMDg1MjE.*_ga_H7P4ZT52H5*MTY4NjEwODUyMC4xLjAuMTY4NjEwODUyMS41OS4wLjA), by Verkruysse *et al.*, 2008
  - [Advancements in noncontact multiparameter physiological measurements using a webcam (ICA)](https://affect.media.mit.edu/pdfs/11.Poh-etal-TBME.pdf), by Poh *et al.*, 2011
  - [Robust pulse rate from chrominance-based rppg (CHROM)](https://ieeexplore.ieee.org/document/6523142), by Haan *et al.*, 2013
  - [Local group invariance for heart rate estimation from face videos in the wild (LGI)](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w27/Pilz_Local_Group_Invariance_CVPR_2018_paper.pdf), by Pilz *et al.*, 2018
  - [Improved motion robustness of remote-PPG by using the blood volume pulse signature (PBV)](https://iopscience.iop.org/article/10.1088/0967-3334/35/9/1913), by Haan *et al.*, 2014
  - [Algorithmic principles of remote ppg (POS)](https://ieeexplore.ieee.org/document/7565547), by Wang *et al.*, 2016
  - [Face2PPG: An Unsupervised Pipeline for Blood Volume Pulse Extraction From Faces (OMIT)](https://ieeexplore.ieee.org/document/10227326), by Álvarez *et al.*, 2023


* Supervised Neural Algorithms 
  - [DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks (DeepPhys)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Weixuan_Chen_DeepPhys_Video-Based_Physiological_ECCV_2018_paper.pdf), by Chen *et al.*, 2018
  - [Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks (PhysNet)](https://bmvc2019.org/wp-content/uploads/papers/0186-paper.pdf), by Yu *et al.*, 2019
  - [Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement (TS-CAN)](https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf), by Liu *et al.*, 2020
  - [EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement (EfficientPhys)](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf), by Liu *et al.*, 2023
  - [BigSmall: Efficient Multi-Task Learning for Disparate Spatial and Temporal Physiological Measurements
 (BigSmall)](https://arxiv.org/abs/2303.11573), by Narayanswamy *et al.*, 2023
  - [PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer (PhysFormer)](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_PhysFormer_Facial_Video-Based_Physiological_Measurement_With_Temporal_Difference_Transformer_CVPR_2022_paper.pdf), by Yu *et al.*, 2022
  - [iBVPNet: 3D-CNN architecture introduced in iBVP dataset paper](https://doi.org/10.3390/electronics13071334), by Joshi *et al.*, 2024
  - [RhythmFormer: Extracting rPPG Signals Based on Hierarchical Temporal Periodic Transformer](https://doi.org/10.48550/arXiv.2402.12788), by Zou *et al.*, 2024

## :file_folder: Datasets
rPPG extractor's pretrained weights are trained from seven datasets, namely SCAMPS, UBFC-rPPG, PURE, BP4D+, UBFC-Phys, MMPD and iBVP. Please cite the corresponding papers when using these datasets. For now, we recommend training with UBFC-rPPG, PURE, iBVP or SCAMPS due to the level of synchronization and volume of the datasets. **To use these datasets in a deep learning model, you should organize the files as follows.**
* [MMPD](https://github.com/McJackTang/MMPD_rPPG_dataset)
    * Jiankai Tang, Kequan Chen, Yuntao Wang, Yuanchun Shi, Shwetak Patel, Daniel McDuff, Xin Liu, "MMPD: Multi-Domain Mobile Video Physiology Dataset", IEEE EMBC, 2023
    -----------------
         data/MMPD/
         |   |-- subject1/
         |       |-- p1_0.mat
         |       |-- p1_1.mat
         |       |...
         |       |-- p1_19.mat
         |   |-- subject2/
         |       |-- p2_0.mat
         |       |-- p2_1.mat
         |       |...
         |...
         |   |-- subjectn/
         |       |-- pn_0.mat
         |       |-- pn_1.mat
         |       |...
    -----------------
    
* [SCAMPS](https://arxiv.org/abs/2206.04197)
    * D. McDuff, M. Wander, X. Liu, B. Hill, J. Hernandez, J. Lester, T. Baltrusaitis, "SCAMPS: Synthetics for Camera Measurement of Physiological Signals", NeurIPS, 2022
    -----------------
         data/SCAMPS/Train/
            |-- P00001.mat
            |-- P00002.mat
         |...
         data/SCAMPS/Val/
            |-- P00001.mat
            |-- P00002.mat
         |...
         data/SCAMPS/Test/
            |-- P00001.mat
            |-- P00002.mat
         |...
    -----------------

* [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)
    * S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
    -----------------
         data/UBFC-rPPG/
         |   |-- subject1/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |   |-- subject2/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |...
         |   |-- subjectn/
         |       |-- vid.avi
         |       |-- ground_truth.txt
    -----------------
   
* [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)
    * Stricker, R., Müller, S., Gross, H.-M.Non-contact "Video-based Pulse Rate Measurement on a Mobile Service Robot"
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
    -----------------
         data/PURE/
         |   |-- 01-01/
         |      |-- 01-01/
         |      |-- 01-01.json
         |   |-- 01-02/
         |      |-- 01-02/
         |      |-- 01-02.json
         |...
         |   |-- ii-jj/
         |      |-- ii-jj/
         |      |-- ii-jj.json
    -----------------
    
* [BP4D+](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
    * Zhang, Z., Girard, J., Wu, Y., Zhang, X., Liu, P., Ciftci, U., Canavan, S., Reale, M., Horowitz, A., Yang, H., Cohn, J., Ji, Q., Yin, L. "Multimodal Spontaneous Emotion Corpus for Human Behavior Analysis", IEEE International Conference on Computer Vision and Pattern Recognition (CVPR) 2016.   
    -----------------
        RawData/
         |   |-- 2D+3D/
         |       |-- F001.zip/
         |       |-- F002.zip
         |       |...
         |   |-- 2DFeatures/
         |       |-- F001_T1.mat
         |       |-- F001_T2.mat
         |       |...
         |   |-- 3DFeatures/
         |       |-- F001_T1.mat
         |       |-- F001_T2.mat
         |       |...
         |   |-- AUCoding/
         |       |-- AU_INT/
         |            |-- AU06/
         |               |-- F001_T1_AU06.csv
         |               |...
         |           |...
         |       |-- AU_OCC/
         |           |-- F00_T1.csv 
         |           |...
         |   |-- IRFeatures/
         |       |-- F001_T1.txt
         |       |...
         |   |-- Physiology/
         |       |-- F001/
         |           |-- T1/
         |               |-- BP_mmHg.txt
         |               |-- microsiemens.txt
         |               |--LA Mean BP_mmHg.txt
         |               |--LA Systolic BP_mmHg.txt
         |               |-- BP Dia_mmHg.txt
         |               |-- Pulse Rate_BPM.txt
         |               |-- Resp_Volts.txt
         |               |-- Respiration Rate_BPM.txt
         |       |...
         |   |-- Thermal/
         |       |-- F001/
         |           |-- T1.mv
         |           |...
         |       |...
         |   |-- BP4D+UserGuide_v0.2.pdf
    -----------------

* [UBFC-Phys](https://sites.google.com/view/ybenezeth/ubfc-phys)
    * Sabour, R. M., Benezeth, Y., De Oliveira, P., Chappe, J., & Yang, F. (2021). Ubfc-phys: A multimodal database for psychophysiological studies of social stress. IEEE Transactions on Affective Computing.  
    -----------------
          RawData/
          |   |-- s1/
          |       |-- vid_s1_T1.avi
          |       |-- vid_s1_T2.avi
          |       |...
          |       |-- bvp_s1_T1.csv
          |       |-- bvp_s1_T2.csv
          |   |-- s2/
          |       |-- vid_s2_T1.avi
          |       |-- vid_s2_T2.avi
          |       |...
          |       |-- bvp_s2_T1.csv
          |       |-- bvp_s2_T2.csv
          |...
          |   |-- sn/
          |       |-- vid_sn_T1.avi
          |       |-- vid_sn_T2.avi
          |       |...
          |       |-- bvp_sn_T1.csv
          |       |-- bvp_sn_T2.csv
    -----------------

* [iBVP](https://github.com/PhysiologicAILab/iBVP-Dataset)
    * Joshi, J.; Cho, Y. iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels. Electronics 2024, 13, 1334.
    -----------------
          iBVP_Dataset/
          |   |-- p01_a/
          |      |-- p01_a_rgb/
          |      |-- p01_a_t/
          |      |-- p01_a_bvp.csv
          |   |-- p01_b/
          |      |-- p01_b_rgb/
          |      |-- p01_b_t/
          |      |-- p01_b_bvp.csv
          |...
          |   |-- pii_x/
          |      |-- pii_x_rgb/
          |      |-- pii_x_t/
          |      |-- pii_x_bvp.csv
    -----------------

---

## :scroll: YAML File Setting
The rPPG-Toolbox uses yaml file to control all parameters for training and evaluation. 
You can modify the existing yaml files to meet your own training and testing requirements.

Here are some explanation of parameters:
* #### ReMOTION_MODE: 
  * `train_and_test`: train on the dataset and use the newly trained model to test.
  * `only_test`: you need to set INFERENCE-MODEL_PATH, and it will use pre-trained model initialized with the MODEL_PATH to test.

  
* #### MODEL : Set used model (Deepphys, TSCAN, Physnet, EfficientPhys, BigSmall, and PhysFormer and their paramaters are supported).
* #### UNSUPERVISED METHOD: Set used unsupervised method. Example: ["ICA", "POS", "CHROM", "GREEN", "LGI", "PBV"]
* #### METRICS: Set used metrics. Example: ['MAE','RMSE','MAPE','Pearson','SNR','BA']
  * 'BA' metric corresponds to the generation of a Bland-Altman plot to graphically compare two measurement techniques (e.g., differences between measured and ground truth heart rates versus mean of measured and ground truth heart rates). This metric saves the plot in the `LOG.PATH` (`runs/exp` by default).
* #### INFERENCE:
  * `USE_SMALLER_WINDOW`: If `True`, use an evaluation window smaller than the video length for evaluation.

    
## :open_file_folder: Adding a New Dataset

* STEP 1: Create a new python file in `dataset/data_loader`, e.g. MyLoader.py

* STEP 2: Implement the required functions, including:

  ```python
  def preprocess_dataset(self, config_preprocess):
  ```
  ```python
  @staticmethod
  def read_video(video_file):
  ```
  ```python
  @staticmethod
  def read_wave(bvp_file):
  ```

* STEP 3:[Optional] Override optional functions. In principle, all functions in BaseLoader can be override, but we **do not** recommend you to override *\_\_len\_\_, \_\_get\_item\_\_,save,load*.
* STEP 4:Set or add configuration parameters.  To set paramteters, create new yaml files in configs/ .  Adding parameters requires modifying config.py, adding new parameters' definition and initial values.

## :robot: Adding a New rPPG Extractor

* STEP 1: Define a model in a new python file in `neural_extractor/model`, e.g. NewModel.py.

* STEP 2: Implement the corresponding training/testing routines just like **rPPG-Toolbox repository**, e.g. NewModelTrainer.py. Ensure to implement the following functions:

  ```python
  def __init__(self, config, data_loader):
  ```
  ```python
  def train(self, data_loader):
  ```
  ```python
  def valid(self, data_loader):
  ```

  ```python
  def test(self, data_loader)
  ```

  ```python
  def save_model(index)
  ```

* STEP 3: Add logic to `main.py` to use the models in the following `train_and_test` and `test` functions like **rPPG-Toolbox repository**. 

* STEP 4: Create new yaml files in configs/ corresponding to the new algorithm.

* STEP 5: Pre-train the rPPG extractor and apply it as an extractor e.g. `neural_extractors/model` and `pretrained_extractors`

## :chart_with_upwards_trend: Adding a New Unsupervised Algorithms

* STEP 1: Define a algorithm in a new python file in `unsupervised_extractors/methods`, e.g. NewMethod.py.

* STEP 2: Add logic to `main.py` to use the models in the following `unsupervised_method_inference` function. 

* STEP 4: Create new yaml files in configs/ corresponding to the new algorithm.
