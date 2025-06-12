# 🧩 Extending the Toolbox

This guide documents how to extend the pipeline with new datasets, models, algorithms, and training configurations, fully synchronized with the 50-epoch curriculum schedule.

---

## 📂 Add a New Dataset

1. Create a loader under `dataset/data_loader/`, e.g., `MyLoader.py`
2. Inherit from `BaseLoader`
3. Implement the following methods:

```python
@staticmethod
def read_video(video_path):
    # load video frames

@staticmethod
def read_wave(signal_path):
    # load physiological signal

def preprocess_dataset(self, config_preprocess):
    # frame selection, ROI extraction, etc.
```

4. Add a corresponding `.yaml` config in `configs/train_configs/`
5. Register dataset in `config.py`

---

## 🧠 Add a New rPPG Extractor

1. Define the model in `neural_extractors/model/NewModel.py`
2. Create a trainer in `neural_extractors/trainers/NewModelTrainer.py`
3. Modify `main.py` to include:

```python
if model_name == 'NewModel':
    model = NewModel(config)
    trainer = NewModelTrainer(config, model)
```

4. Prepare corresponding YAML in `configs/train_configs/`
5. Place pretrained weights in `pretrained_extractors/` (if any))
---

## 🤖 Add a New Unsupervised Algorithm

1. Create `unsupervised_extractors/methods/NewMethod.py`
2. Implement:

```python
def extract(self, video):
    # signal extraction algorithm
```

3. Register the method in `main.py` under `unsupervised_method_inference`
4. Add `.yaml` under `configs/infer_configs/`

---

## 🧪 Modify YAML Parameters

```yaml
ReMOTION_MODE: train_and_test
DATASET: MAHNOB_HCI
MODEL: PhysMamba
USE_PRETRAINED: True
LOSS: SupConTopK
...
```

* `PREPROCESS: True/False`: toggle frame extraction
* `TOP_K_RATIO`: chunk selection
* `SPARSITY_WEIGHT`: entropy regularization

---

## 📌 Tips

* You may reuse `ChunkAuxClassifier` or `GatedPooling` across models
* Each model should define its own projection head if used with SupConLoss

---

For more advanced customization, refer to `main.py`, `config.py`, and `trainers/`.



---

## 🔍 Notes on Key Extensions

* **Entropy Regularization Weight** (`LAMBDA_ENTROPY`) is adaptive, based on attention entropy in early epochs.
* **Classifier LR Boost** applies only for the first 6 epochs of Phase 2, multiplying the base learning rate.
* **Top-K Ratio** is scheduled from 0.70 (start of P1) → 0.40 (end of P1) using a cosine or linear decay.
* **Attention Kernel** progresses through: `softmax` → `α-entmax` (α = 1.0→1.6) → raw scores (Phase-2).

---