# Emotion Recognition from rPPG

*Physiologically-Inspired Temporal Encoding & Curriculum Learning*

[![License](https://img.shields.io/badge/license-Responsible%20AI-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-yellow.svg)]()

---

## 📋 Contents

1. [Overview](#overview) 
2. [Quick Start](#quick-start) 
3. [Model Flow](#model-flow) 
4. [Training Curriculum](#training-curriculum) 
5. [Cite Us](#how-to-cite) 
6. [License](#license) 
7. [Detailed Docs](#detailed-docs)

---

## 📝 Overview

End-to-end **phase-aware** emotion recognition on top of [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox).
Three phases (explore → chunk-refine → session-finetune) within **50 epochs**.

---

## 🚀 Quick Start

```bash
git clone https://github.com/LeeChangmin0310/ReMOTION-Temporal.git
cd EmotionRecognition-rPPG
bash setup.sh conda          # or: bash setup.sh uv
conda activate remotion

# train / validate / test
python main.py --config configs/train_configs/Arsl_BC_Normal_PHYSMAMBA.yaml
```

---

## �\uddna Model Flow

<p align="center"><img src="docs/figures/architecture.png" width="80%" alt="Full Architecture"/></p>

<details>
<summary>Phase-by-Phase (0–14 | 15–29 | 30–49)</summary>

| Phase | Epochs |            Core Losses           |          Main Goal          |
| :---: | :----: | :------------------------------: | :-------------------------: |
|   0   |  0–14  |       SupConTopK + Entropy       | Diverse temporal embeddings |
|   1   |  15–29 | Focal Chunk-CE (+0.2 Session-CE) |    Chunk discriminability   |
|   2   |  30–49 |         Session-CE (only)        |  Stable session classifier  |

</details>

---

## 🎯 Training Curriculum (50 Epochs)

| Phase | Epochs |       Attention Kernel       |               Main Losses (weights)               |        Trainable Blocks        |
| :---: | :----: | :--------------------------: | :-----------------------------------------------: | :----------------------------: |
|   0   |  0–14  | softmax → α-entmax (1.0→1.4) |        SupCon 0.50 • Entropy λ (0.05→≤0.20)       |  MTDE, AttnScorer, Projection  |
|   1   |  15–29 |      α-entmax (1.4→1.6)      | Chunk-CE 0.30→1.0 • SupCon 0.10 • Session-CE 0.20 | +ChunkAux, Pooling, Classifier |
|   2   |  30–49 |          raw scores          |   Session-CE 1.0 (Classifier LR ×2 for 6 epochs)  |    MTDE, Pooling, Classifier   |

---

## 📁 Detailed Docs

* [📂 Datasets](docs/datasets.md)
* [🛠️ Model & Loss Details](docs/model_details.md)
* [🔧 Extending the Toolbox](docs/extensions.md)

---

## 📁 How to Cite

```bibtex
@article{lee2025emotion,
  title  = {Emotion Recognition from rPPG via Physiologically-Inspired Temporal Encoding and Attention-based Curriculum Learning},
  author = {Lee, Changmin and Lee, Hyunwoo and Whang, Mincheol},
  journal= {Sensors},
  year   = {2025},
  volume = {25},
  number = {6},
  pages  = {XXXX}
}
```

---

## 🛡️ License

Inherited from the original rPPG-Toolbox — Responsible AI License.
