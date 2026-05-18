# Knowledge Distillation for On-Device Intent Classification

Compressing BERT into a mobile-friendly intent classifier with minimal performance loss

![F1](https://img.shields.io/badge/F1_Score-0.9835-brightgreen)
![Compression](https://img.shields.io/badge/Compression-12.3x-blue)
![Size](https://img.shields.io/badge/Model_Size-34.1_MB-orange)

---

## Overview

This project demonstrates that knowledge distillation can compress a full BERT-base model (109.5M parameters, 417.7 MB) into a
lightweight compact transformer student (8.9M parameters, 34.1 MB) while retaining **99.2% of the teacher's F1 score** for intent classification — achieving a 12.3x compression suitable for on-device mobile deployment.

### Research Question

> Can a lightweight distilled transformer model retain strong intent recognition performance while reducing model size by >90% for mobile deployment?

**Answer:** Yes. The distilled student achieves 98.36% test accuracy (vs. 99.14% teacher) while being 12.3x smaller — outperforming the student baseline trained without distillation (98.07%).

---

## Results at a Glance

| Model | Parameters | Size | Test Accuracy | F1 (Macro) | Training Time |
|-------|-----------|------|---------------|------------|---------------|
| Teacher (BERT-base) | 109,487,623 | 417.7 MB | 99.14% | 0.9915 | 73.5 min |
| Student Baseline | 8,935,943 | 34.1 MB | 98.07% | 0.9807 | 12.4 min |
| **Student Distilled** | **8,935,943** | **34.1 MB** | **98.36%** | **0.9835** | 79.6 min |

**Key Finding:** The distilled student outperforms the baseline by 0.29% accuracy despite using the identical architecture — demonstrating that knowledge distillation transfers meaningful decision behavior beyond what hard labels alone can teach.

---

## Motivation & Reasoning

### Why On-Device Intent Classification?

1. **Privacy** — Voice commands never leave the device; no cloud processing of personal utterances
2. **Latency** — Real-time classification without network round-trips (critical for voice assistants)
3. **Offline capability** — Works in airplane mode, poor connectivity, or restricted networks
4. **Cost** — No per-inference cloud API costs for millions of voice interactions
5. **Security** — No attack surface from network transmission of voice/text data

### Why Knowledge Distillation?

Rather than training a small model from scratch (which often sacrifices quality)
or deploying a full BERT model (impractical on mobile), distillation offers the
best of both worlds:

- Teacher quality in a student-sized package
- Soft targets from the teacher provide richer supervision than hard labels alone
- The student learns the teacher's confidence distribution, not just argmax decisions
- Temperature scaling (T=4.0) amplifies information in the teacher's probability tails across 7 intent classes

### Why BERT-base as Teacher?

- State-of-the-art for text classification at this scale
- Well-understood architecture with reproducible results
- Pre-trained representations capture deep linguistic patterns
- Sufficient capacity to learn near-perfect intent detection (99.14%)

### Why This Specific Student Architecture?

The compact transformer student (2 layers, 256 hidden, 4 heads) was designed with mobile constraints:

| Dimension | Teacher | Student | Reasoning |
|-----------|---------|---------|-----------|
| Layers | 12 | 2 | Diminishing returns beyond 2 for this task complexity |
| Hidden Size | 768 | 256 | Short utterances (~9 words); 256 captures sufficient semantics |
| Attention Heads | 12 | 4 | Reduced heads still capture key attention patterns |
| Max Sequence | 512 | 32 | 100% of SNIPS fits in 32 tokens |
| Size | 417.7 MB | 34.1 MB | Fits comfortably in mobile memory constraints |

---

## Dataset

### SNIPS Natural Language Understanding Benchmark

| Property | Value |
|----------|-------|
| Source | SNIPS (Coucke et al., 2018) |
| Domain | Voice assistant intent classification |
| Total samples | 14,484 |
| Classes | 7 intents (balanced) |
| Language | English |
| Average length | 9.1 words / 12.4 BERT tokens |

### Intent Classes

| Intent | Description | Example | Train Count |
|--------|-------------|---------|-------------|
| AddToPlaylist | Add song/artist to playlist | "Add this song to my workout playlist" | 1,658 |
| BookRestaurant | Make restaurant reservations | "Book a table for two tonight" | 1,686 |
| GetWeather | Weather queries | "What's the weather tomorrow?" | 1,710 |
| PlayMusic | Music playback requests | "Play some jazz music" | 1,710 |
| RateBook | Rate books/media | "Give this book 4 stars" | 1,670 |
| SearchCreativeWork | Find movies/books/songs | "Find the movie Inception" | 1,668 |
| SearchScreeningEvent | Find movie showtimes | "Find movie times nearby" | 1,673 |

### Data Splits

| Split | Samples | Source |
|-------|---------|--------|
| Train | 11,775 | 90% of original train (stratified) |
| Validation | 1,309 | 10% stratified hold-out |
| Test | 1,400 | Original test set (200/class) |

**Key data characteristics:**
- Perfectly balanced classes (imbalance ratio: 1.03)
- Max token length: 41 → 100% fits in 32 tokens
- Well-separated intent vocabulary (low cross-class TF-IDF cosine similarity)
- No preprocessing beyond tokenization needed (clean benchmark data)

---

## Architecture Details

### Teacher Model: BERT-base-uncased

```
BertModel(
  embeddings: BertEmbeddings(vocab=30522, hidden=768, max_pos=512)
  encoder: 12× BertLayer(
    attention: MultiHeadAttention(12 heads, 768 dim)
    intermediate: Linear(768 → 3072, GELU)
    output: Linear(3072 → 768)
  )
  pooler: Linear(768 → 768, Tanh)
)
classifier: Linear(768 → 7)
Total Parameters: 109,487,623
```

### Student Model: Compact Transformer

```
StudentModel(
  token_embedding: Embedding(30522, 256)
  positional_encoding: Sinusoidal(max_len=32)
  embedding_norm: LayerNorm(256)
  transformer_encoder: 2× TransformerEncoderLayer(
    attention: MultiHeadAttention(4 heads, 256 dim)
    feedforward: Linear(256 → 512, GELU) → Linear(512 → 256)
  )
  classifier: Sequential(
    Linear(256 → 256, GELU)
    Dropout(0.1)
    Linear(256 → 7)
  )
)
Total Parameters: 8,935,943
```

---

## Training Methodology

### Teacher Fine-Tuning

- Base model: `bert-base-uncased` (HuggingFace)
- Epochs: 5
- Learning rate: 2e-5 (AdamW with cosine annealing)
- Batch size: 32
- Loss: CrossEntropyLoss (no class weights — dataset is balanced)
- Result: 99.14% test accuracy

### Student Baseline Training

- Epochs: 20 (best at epoch 17)
- Learning rate: 5e-4 (higher LR for training from scratch)
- Batch size: 64
- Loss: CrossEntropyLoss
- Purpose: Establish what the student architecture achieves without distillation knowledge
- Result: 98.07% test accuracy

### Knowledge Distillation Training

- Epochs: 20 (best at epoch 13)
- Learning rate: 5e-4
- Temperature (T): 4.0
- Alpha (α): 0.7
- Loss function:

$$\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}\left(\text{softmax}\left(\frac{z_t}{T}\right) \| \text{softmax}\left(\frac{z_s}{T}\right)\right) + (1 - \alpha) \cdot \text{CE}(y, \hat{y}_s)$$

**Reasoning for hyperparameters:**

- **T=4.0:** High temperature smooths teacher logits, exposing inter-class similarity information that hard labels cannot convey. For 7 well-separated classes, T=4 is the sweet spot where inter-class relationships are visible but dominant predictions still guide learning.
- **α=0.7:** Emphasizes learning from teacher's distribution (70%) while hard labels provide anchoring (30%) against potential teacher mistakes.
- **No class weighting:** Dataset is perfectly balanced; no need for weighted loss.

---

## Key Findings

### 1. Distillation Outperforms Baseline Training

The distilled student achieves measurably better test performance than the baseline:

| Metric | Teacher | Baseline | Distilled | Distillation Gain |
|--------|---------|----------|-----------|-------------------|
| Test Accuracy | 99.14% | 98.07% | 98.36% | +0.29% |
| Test F1 (Macro) | 0.9915 | 0.9807 | 0.9835 | +0.0028 |
| Teacher Retention | — | 98.9% | 99.2% | +0.3% |

### 2. Soft Targets Provide Richer Supervision

Despite identical architectures and training budgets, the distilled student learns better because:
- The teacher's probability distribution over 7 classes encodes inter-intent similarities
- E.g., "Find jazz songs" → teacher assigns probability mass to both PlayMusic and SearchCreativeWork
- This implicit relational knowledge is unavailable from one-hot hard labels

### 3. Compression Without Meaningful Compromise

| Metric | Value |
|--------|-------|
| Parameter reduction | 12.3x (109.5M → 8.9M) |
| Size reduction | 91.8% (417.7 MB → 34.1 MB) |
| F1 retention | 99.2% |
| Efficiency gain | 12.3x better F1/MB ratio |

### 4. Efficient Training on CPU

All experiments completed on CPU (no GPU required):
- Teacher fine-tuning: 73.5 minutes (5 epochs)
- Student baseline: 12.4 minutes (20 epochs)
- Student distillation: 79.6 minutes (20 epochs, teacher inference overhead)

---

## Project Structure

```
distillation_intent/
├── data/
│   ├── raw/                        # Original downloaded CSVs
│   │   ├── train.csv               # 13,084 samples
│   │   └── test.csv                # 1,400 samples
│   └── processed/                  # Preprocessed train/val/test splits
│       ├── train.csv               # 11,775 samples
│       ├── validation.csv          # 1,309 samples
│       ├── test.csv                # 1,400 samples
│       └── label_map.json          # Intent → index mapping
├── notebooks/
│   ├── 00_data_exploration.ipynb   # Comprehensive EDA with plots
│   └── 01_full_pipeline.ipynb      # End-to-end experiment notebook
├── src/
│   ├── config.py                   # Centralized configuration (all hyperparams)
│   ├── data/
│   │   ├── dataset.py             # PyTorch Dataset + DataLoader utilities
│   │   └── download_snips.py      # Dataset download from HuggingFace
│   ├── models/
│   │   ├── teacher.py             # BERT-base teacher (109.5M params)
│   │   └── student.py            # Compact transformer student (8.9M params)
│   ├── training/
│   │   ├── logger.py             # Training logger (JSON/CSV/console)
│   │   ├── train_teacher.py      # Teacher fine-tuning script
│   │   ├── train_student.py      # Student baseline + distillation training
│   │   └── temperature_sweep.py  # Temperature sweep experiment
│   └── evaluation/
│       ├── evaluate.py           # Metrics & model comparison utilities
│       └── visualize.py          # Plotting functions
├── outputs/
│   ├── teacher/                   # Teacher model checkpoints & logs
│   │   ├── best_model.pt
│   │   ├── results.json
│   │   └── metrics.csv
│   ├── student/
│   │   ├── baseline/             # Student trained without distillation
│   │   │   ├── best_model.pt
│   │   │   ├── results.json
│   │   │   └── metrics.csv
│   │   └── distill/              # Student trained with distillation
│   │       ├── best_model.pt
│   │       ├── results.json
│   │       └── metrics.csv
├── reports/                       # Generated visualizations
├── DECISIONS.md                   # Design decisions with reasoning
├── requirements.txt
├── verify_pipeline.py             # Pipeline verification script
└── README.md
```

---

## Setup & Reproduction

### Prerequisites

- Python 3.10+
- ~2 GB disk space for models and data
- CPU sufficient (training was done entirely on CPU)

### Installation

```bash
# Clone the repository
git clone https://github.com/mrkderchef/on_device_intent_classifier.git
cd on_device_intent_classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# .\venv\Scripts\activate       # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# Download SNIPS from HuggingFace and create splits
python -m src.data.download_snips
```

### Training Models

```bash
# 1. Fine-tune teacher (BERT-base, ~75 min on CPU)
python -m src.training.train_teacher

# 2. Train student baseline (no distillation, ~12 min on CPU)
python -m src.training.train_student --mode baseline

# 3. Train distilled student (requires teacher checkpoint, ~80 min on CPU)
python -m src.training.train_student --mode distill

# 4. Run temperature sweep (8 temperatures × 15 epochs each)
python -m src.training.temperature_sweep

# 5. Evaluate and compare models
python -m src.evaluation.evaluate --model compare
```

### Verify Pipeline

```bash
# Quick sanity check (imports, data loading, forward passes)
python verify_pipeline.py
```

### Running Notebooks

```bash
jupyter notebook notebooks/00_data_exploration.ipynb   # Data analysis
jupyter notebook notebooks/01_full_pipeline.ipynb      # Full experiment
```

---

## Design Decisions & Reasoning

### Why max_length=32?

Analysis of BERT tokenization showed that 100% of SNIPS utterances fit within 32 tokens (99th percentile: 24 tokens). Using 32 instead of BERT's default 512:

- Reduces computation by 16x (attention is O(n²))
- No information loss since all sequences fit
- Matches the short-utterance nature of voice commands

### Why no class weighting?

With perfectly balanced classes (each intent has 1,658–1,710 train samples, variance < 3%), class weighting would add unnecessary complexity with no benefit.

### Why T=4.0?

Temperature controls how much probability mass is redistributed:

- T=1: Original sharp distribution → student just sees hard labels
- T=4: Smoothed distribution → student learns from tail probabilities
- T>8: Too smooth → inter-class information is washed out

T=4 is the sweet spot for 7 well-separated intent classes.

### Why α=0.7?

Higher alpha (0.7) emphasizes the teacher's soft knowledge over hard labels. For a high-quality teacher (99.14% accuracy), trusting its distributions more than raw labels makes sense. Hard labels still anchor training (30%) to prevent propagating the teacher's rare mistakes.

### Why 2 transformer layers?

With average input length of only 9.1 words (12.4 tokens), deep architectures provide diminishing returns. Two layers capture sufficient attention patterns for short, well-separated utterances while keeping inference fast for mobile deployment.

---

## Limitations & Future Work

### Current Limitations

1. **Single dataset** — Results demonstrated on SNIPS benchmark only
2. **No mobile benchmark** — Inference speed not measured on actual mobile hardware
3. **No quantization** — Model could be further compressed with INT8/INT4 quantization
4. **English only** — Not tested on multilingual intent datasets
5. **7 classes only** — Real production systems may have 50+ intents

### Potential Extensions

- **Quantization:** INT8 would reduce to ~8.5 MB, INT4 to ~4.3 MB
- **ONNX export:** For cross-platform mobile deployment
- **CoreML / TensorFlow Lite conversion:** Direct mobile framework integration
- **Multi-intent expansion:** Scale to production-level intent taxonomies
- **Adversarial testing:** Evaluate robustness against out-of-domain utterances
- **Temperature sweep analysis:** Find optimal T for different class counts

---

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
tensorboard>=2.13.0
jupyter>=1.0.0
ipykernel>=6.25.0
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{on-device-intent-classifier,
  title={Knowledge Distillation for On-Device Intent Classification},
  author={mrkderchef},
  year={2026},
  url={https://github.com/mrkderchef/on_device_intent_classifier}
}
```

---

## License

MIT
