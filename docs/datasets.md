# 📂 Dataset Overview

This project uses the **MAHNOB-HCI** dataset for emotion recognition tasks using rPPG signals.

---

## 📁 Folder Structure (MAHNOB-HCI)

```
MAHNOB_HCI_Emotion/
├── emotion_labels.csv
├── 2/
│   └── 2.avi
├── 4/
│   └── 4.avi
...
```

* Videos are in `.avi` format
* Labels are provided in `emotion_labels.csv`
* Input video is split into 128-frame chunks

---

## 📐 Data Format

* **Video Shape**: NDHWC (Num, Depth, Height, Width, Channel)
* **Chunk Length**: 128 frames (≈ 4 seconds)
* **Chunk Labels**: None (only session-level labels are provided)

---

## 🧩 Preprocessing Pipeline

> Controlled via `MAHNOBHCILoader` and YAML config

1. Load AVI video
2. Extract facial ROI using pretrained detector
3. Normalize frame brightness
4. Extract rPPG signal from each chunk (via PhysMamba or others)

---

## 🛠 Custom Dataset Integration

To add your own dataset:

1. Create `YourLoader.py` under `dataset/data_loader/`
2. Inherit from `BaseLoader`
3. Implement:

   * `preprocess_dataset(self, config)`
   * `read_video()`
   * `read_wave()`

Add config in `configs/`, and define parameters in `config.py`.

---

## 📖 Notes

* Ensure your video format supports OpenCV (e.g. `.avi`, `.mp4`)
* Each session must have a unique ID, matching the label file
* Preprocessing can be skipped on re-runs by toggling `PREPROCESS: False`
