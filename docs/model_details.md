# 🛠️ Model & Loss Details

---

## 🔑 Main Blocks

| Module                 | Role                                                 |
| ---------------------- | ---------------------------------------------------- |
| **MTDE**               | Multi-scale 1D CNN (RF ≈ 0.15 / 1.6 / 3.7 s)         |
| **AttnScorer**         | Chunk importance via softmax → α-entmax → raw logits |
| **ProjectionHead**     | 256-D → 128-D projection for contrastive learning    |
| **ChunkAuxClassifier** | Focal CE on Top-K chunks (Phase-1)                   |
| **GatedPooling**       | α-entmax pooling + sigmoid gate                      |
| **Classifier**         | Two-layer MLP (96 hidden units)                      |

---

## 🦜 Flow Diagram
<p align="center"><img src="docs/figures/architecture.png" width="80%" alt="Full Architecture"/></p>
```
rPPG  →  MTDE (encoder)  →  chunk-embeddings
                         ↓
                    AttnScorer (raw scores)
                         ↓
        ┌────────────┬────────────┐
        │            │            │
   SupConLoss    Chunk-CE*    GatedPooling
        ↓            │              ↓
   (Phase-0)     (Phase-1)    session-embedding
                                        ↓
                               ClassificationHead
                                        ↓
                           CrossEntropyLoss + (optional) SparsityLoss
```

---

## 🌟 Loss per Phase

| Phase | Losses (active → weight)                                 |
| :---: | -------------------------------------------------------- |
|   0   | SupConTopK → 0.50,  Entropy(λ) → adaptive                |
|   1   | Focal Chunk-CE → 1.0,  SupCon → 0.10,  Session-CE → 0.20 |
|   2   | Session-CE → 1.0                                         |

---

## ⚙️ Attention Schedule

| Epoch Range | Kernel     | α Value   |
| ----------- | ---------- | --------- |
| 0–4         | softmax    | –         |
| 5–14        | α-entmax   | 1.0 → 1.4 |
| 15–29       | α-entmax   | 1.4 → 1.6 |
| 30–49       | raw scores | –         |

---

## 🔍 Top-K Strategy

* Ratio decays: **0.70 → 0.40** over Phase-1
* Strategy: **hard forward / soft backward (Straight-Through)**
* Safeguard: **minimum 6 chunks** selected at all times

---

## 🗓️ Component Trainability

| Component          |      P0      |  P1 |  P2 |
| ------------------ | :----------: | :-: | :-: |
| MTDE               |       ✅      |  ✅  |  ✅  |
| AttnScorer         |       ✅      |  ✅  |  ❌  |
| ProjectionHead     |       ✅      |  ✅  |  ❌  |
| ChunkAuxClassifier |       ❌      |  ✅  |  ❌  |
| GatedPooling       |      🌓      |  ✅  |  ✅  |
| Classifier         | 🌓 (cushion) |  ✅  |  ✅  |

Legend: ✅ = trainable, ❌ = frozen, 🌓 = partial (lower lr or weight).

---
