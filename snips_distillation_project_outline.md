# Knowledge Distillation for On-Device Intent Classification

## Official Topic Category
Model Compression: Distillation of Large Language Models

---

# Project Title
## Knowledge Distillation for On-Device Intent Classification

---

# 1. Core Project Idea

The goal of this project is to investigate whether knowledge distillation can compress transformer-based language models into lightweight student models suitable for theoretical mobile and edge deployment.

The project focuses on intent classification for voice assistants and conversational AI systems.

The central research question is:

> Can knowledge distillation preserve strong intent recognition performance while significantly reducing the computational and memory requirements of transformer-based language models?

The project investigates the tradeoff between:
- model performance
- computational efficiency
- deployment feasibility
- mobile inference constraints

---

# 2. Story and Motivation

Modern voice assistants rely heavily on natural language understanding (NLU).

Systems such as:
- Siri
- Google Assistant
- Alexa
- smart home assistants

must rapidly understand user intent.

However, modern transformer language models are often too computationally expensive for efficient on-device deployment.

This creates several challenges:
- latency
- battery usage
- memory requirements
- cloud infrastructure cost
- privacy concerns

The project investigates whether knowledge distillation can create lightweight transformer models capable of maintaining strong intent classification performance while being significantly more deployment-friendly.

The project is framed around:
- edge AI
- mobile NLP
- efficient transformers
- on-device inference

---

# 3. Dataset Selection

## Main Dataset
### SNIPS Natural Language Understanding Dataset

Example utterances:

> “Play some jazz music”

> “Book a table for two tonight”

> “What’s the weather tomorrow?”

Intent classes:
- PlayMusic
- BookRestaurant
- GetWeather
- AddToPlaylist
- SearchCreativeWork
- RateBook
- SearchScreeningEvent

---

# 4. Planned Model Architecture

## Teacher Model
- BERT-base
- ~110M parameters
- 12 layers
- hidden size 768

## Student Model
- 2–4 transformer layers
- hidden size 256–384
- ~5M–15M parameters

Goal:
- maintain strong performance
- significantly reduce compute and memory requirements

---

# 5. Knowledge Distillation

Softmax with temperature:

p_i = exp(z_i/T) / sum_j exp(z_j/T)

Distillation loss:

L = alpha * T^2 * KL(p_t || p_s) + (1-alpha) * L_CE

The student learns:
- hard labels
- teacher probability distributions
- semantic class relationships

---

# 6. Planned Experiments

1. Teacher benchmark
2. Student baseline
3. Distilled student
4. Temperature sweep
5. Compression comparison

Metrics:
- Accuracy
- Precision
- Recall
- F1
- Parameter count
- Memory usage
- Inference speed

---

# 7. Mobile AI Story

The project investigates whether compressed transformer models could theoretically support:
- mobile NLP
- edge AI
- offline assistants
- low latency inference
- privacy-preserving AI

without requiring large cloud infrastructure.

---

# 8. Repository Structure

project-root/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
├── outputs/
├── reports/
├── requirements.txt
├── README.md

---

# 9. Expected Conclusion

Knowledge distillation can significantly reduce transformer model size while preserving much of the original NLP performance, enabling more realistic mobile and edge deployment scenarios.
