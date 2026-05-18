# Knowledge Distillation for On-Device Intent Classification

## Model Compression: Distillation of Large Language Models

---

## Overview

This project investigates whether **knowledge distillation** can compress transformer-based language models into lightweight student models suitable for mobile and edge deployment.

The focus is on **intent classification** for voice assistants and conversational AI systems (e.g., Siri, Google Assistant, Alexa, smart home assistants).

### Research Question

> Can knowledge distillation preserve strong intent recognition performance while significantly reducing the computational and memory requirements of transformer-based language models?

---

## Motivation

Modern NLU systems rely on large transformer models that are too computationally expensive for efficient on-device deployment. This creates challenges around:

- **Latency** — real-time responses required
- **Battery usage** — constrained mobile power budgets
- **Memory** — limited RAM on edge devices
- **Privacy** — on-device inference avoids cloud data transmission
- **Cost** — reduced cloud infrastructure dependency

This project explores knowledge distillation as a path toward efficient, deployment-friendly intent classifiers.

---

## Dataset

### SNIPS Natural Language Understanding Dataset

A benchmark dataset for intent classification in voice assistant scenarios.

| Split | Samples | Notes |
|-------|---------|-------|
| Train | 11,775 | 90% of original train |
| Validation | 1,309 | 10% stratified hold-out |
| Test | 1,400 | Original test set |

**Intent classes (7, balanced):**
| Intent | Example | Train Count |
|--------|---------|-------------|
| AddToPlaylist | "Add this song to my workout playlist" | 1,658 |
| BookRestaurant | "Book a table for two tonight" | 1,686 |
| GetWeather | "What's the weather tomorrow?" | 1,710 |
| PlayMusic | "Play some jazz music" | 1,710 |
| RateBook | "Give this book 4 stars" | 1,670 |
| SearchCreativeWork | "Find the movie Inception" | 1,668 |
| SearchScreeningEvent | "Find movie times nearby" | 1,673 |

**Key data characteristics** (from exploration):
- Mean utterance length: 9.1 words / 12.4 BERT tokens
- Max token length: 41 (100% fits in 32 tokens)
- Perfectly balanced classes (imbalance ratio: 1.03)
- Well-separated intent vocabulary (low TF-IDF cosine similarity)

---

## Architecture

### Teacher Model
- **BERT-base-uncased** — 109.5M parameters, 417 MB
- 12 transformer layers, hidden size 768
- Fine-tuned for 5 epochs (standard BERT fine-tuning)

### Student Model
- **Compact Transformer** — 8.9M parameters, 34 MB
- 2 transformer encoder layers, hidden size 256, 4 attention heads
- **Compression ratio: 12.3x**

### Design Rationale
See [`DECISIONS.md`](DECISIONS.md) for detailed reasoning behind all architectural and hyperparameter choices.

---

## Knowledge Distillation

The student learns from both hard labels and teacher soft probability distributions:

```
L = α · T² · KL(p_teacher ∥ p_student) + (1 - α) · L_CE
```

Where:
- `T = 4.0` — temperature (moderate softening for 7-class task)
- `α = 0.7` — 70% distillation loss, 30% hard-label loss
- `KL` = Kullback-Leibler divergence

---

## Experiments

1. **Teacher benchmark** — Fine-tune BERT-base on SNIPS (5 epochs)
2. **Student baseline** — Train student from scratch, hard labels only (20 epochs)
3. **Distilled student** — Train student with teacher knowledge (20 epochs)
4. **Temperature sweep** — Evaluate T ∈ {1, 2, 4, 6, 8, 10, 15, 20}
5. **Compression analysis** — Compare parameter counts, memory, speed

### Metrics (logged per epoch)
- Accuracy, Precision (macro), Recall (macro), F1-score (macro)
- Training loss, validation loss
- Per-class breakdown
- Parameter count & model size
- Inference latency

### Logging
All experiments produce structured logs in `outputs/`:
- `config.json` — hyperparameters and setup
- `metrics.csv` — epoch-by-epoch metrics
- `results.json` — final evaluation results
- `training.log` — timestamped console log

---

## Project Structure

```
distillation_intent/
├── data/
│   ├── raw/                     # Original downloaded CSVs
│   └── processed/               # Train/val/test splits + label_map.json
├── notebooks/
│   ├── 00_data_exploration.ipynb  # Dataset analysis & plots
│   └── 01_full_pipeline.ipynb     # End-to-end experiment notebook
├── src/
│   ├── config.py                # Centralized configuration (all hyperparams)
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset + DataLoader
│   │   └── download_snips.py    # Dataset download script
│   ├── models/
│   │   ├── teacher.py           # BERT-base teacher model
│   │   └── student.py           # Compact transformer student
│   ├── training/
│   │   ├── logger.py            # Training logger (JSON/CSV/console)
│   │   ├── train_teacher.py     # Teacher training script
│   │   ├── train_student.py     # Student training + distillation
│   │   └── temperature_sweep.py # Temperature sweep experiment
│   └── evaluation/
│       ├── evaluate.py          # Metrics & model comparison
│       └── visualize.py         # Plotting utilities
├── outputs/                     # Model checkpoints & logs (gitignored)
├── reports/                     # Generated figures
├── DECISIONS.md                 # Design decisions with reasoning
├── requirements.txt
├── verify_pipeline.py           # Pipeline verification script
└── README.md
```

---

## Setup & Usage

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset (already included in repo)
python -m src.data.download_snips

# Verify pipeline works
python verify_pipeline.py
```

### Training Commands

```bash
# 1. Train teacher model (BERT-base, ~10-20 min on CPU)
python -m src.training.train_teacher

# 2. Train student baseline (no distillation, ~5-10 min on CPU)
python -m src.training.train_student --mode baseline

# 3. Train student with distillation (~5-10 min on CPU)
python -m src.training.train_student --mode distill

# 4. Run temperature sweep (8 temperatures × 15 epochs each)
python -m src.training.temperature_sweep

# 5. Evaluate and compare models
python -m src.evaluation.evaluate --model compare
```

---

## Key Hyperparameters

| Parameter | Teacher | Student | Reasoning |
|-----------|---------|---------|-----------|
| Epochs | 5 | 20 | BERT converges fast; student needs more |
| Learning rate | 2e-5 | 5e-4 | Fine-tuning vs. training from scratch |
| Batch size | 32 | 64 | Smaller model allows larger batches |
| Max length | 32 | 32 | 100% of data fits (saves 50% vs. 64) |
| Optimizer | AdamW | AdamW | Standard for transformers |
| Scheduler | Cosine | Cosine | Smooth learning rate decay |

---

## Expected Results

Knowledge distillation is expected to:
- Retain **90%+** of teacher accuracy with **12x** fewer parameters
- Outperform student baseline (trained without distillation)
- Show optimal temperature in the T=4-6 range
- Demonstrate meaningful inference speedup

---

## License

MIT License
