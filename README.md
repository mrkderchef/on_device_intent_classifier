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

**Intent classes (7):**
| Intent | Example |
|--------|---------|
| PlayMusic | "Play some jazz music" |
| BookRestaurant | "Book a table for two tonight" |
| GetWeather | "What's the weather tomorrow?" |
| AddToPlaylist | "Add this song to my workout playlist" |
| SearchCreativeWork | "Find the movie Inception" |
| RateBook | "Give this book 4 stars" |
| SearchScreeningEvent | "Find movie times nearby" |

---

## Architecture

### Teacher Model
- **BERT-base-uncased** (~110M parameters)
- 12 transformer layers, hidden size 768

### Student Model
- **Compact Transformer** (~5–15M parameters)
- 2–4 transformer layers, hidden size 256–384

---

## Knowledge Distillation

The student learns from both hard labels and teacher soft probability distributions:

```
L = α · T² · KL(p_teacher ∥ p_student) + (1 - α) · L_CE
```

Where:
- `T` = temperature (softens probability distributions)
- `α` = weight balancing distillation vs. hard-label loss
- `KL` = Kullback-Leibler divergence

---

## Experiments

1. **Teacher benchmark** — Fine-tune BERT-base on SNIPS
2. **Student baseline** — Train student from scratch (no distillation)
3. **Distilled student** — Train student with teacher knowledge
4. **Temperature sweep** — Evaluate T ∈ {1, 2, 3, 5, 10, 20}
5. **Compression analysis** — Compare parameter counts, memory, speed

### Metrics
- Accuracy, Precision, Recall, F1-score
- Parameter count & model size
- Memory footprint
- Inference latency

---

## Project Structure

```
distillation_intent/
├── data/                    # Raw and processed dataset
├── notebooks/               # Jupyter notebooks for experiments
├── src/
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Teacher and student model definitions
│   ├── training/            # Training loops and distillation
│   └── evaluation/          # Evaluation and metrics
├── outputs/                 # Model checkpoints, logs
├── reports/                 # Figures, results tables
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset
python src/data/download_snips.py
```

---

## Usage

```bash
# Train teacher model
python -m src.training.train_teacher

# Train student baseline (no distillation)
python -m src.training.train_student --mode baseline

# Train student with distillation
python -m src.training.train_student --mode distill

# Evaluate models
python -m src.evaluation.evaluate --model teacher
python -m src.evaluation.evaluate --model student

# Run temperature sweep
python -m src.training.temperature_sweep
```

---

## Expected Results

Knowledge distillation is expected to:
- Retain **90%+** of teacher accuracy with **10–20x** fewer parameters
- Enable theoretical on-device deployment scenarios
- Demonstrate meaningful knowledge transfer through soft labels

---

## License

MIT License
