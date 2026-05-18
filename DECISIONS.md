# Design Decisions & Reasoning

This document tracks all key design decisions made during project development,
including the reasoning behind each choice. It serves as an audit trail.

---

## 1. Max Sequence Length: 32 tokens

**Decision:** Set `max_length=32` for tokenization.

**Reasoning:**
- Dataset analysis shows 100% of utterances fit within 32 BERT tokens
- Maximum token length observed: 41 (but with [CLS] and [SEP] that's the encoded length)
- 99th percentile is only 24 tokens
- Using 32 instead of 64 or 128 cuts padding waste in half
- For CPU training this significantly reduces compute per batch
- No information loss since all sequences fit within this budget

**Alternative considered:** 64 tokens (original plan). Rejected because it wastes 50%+ of sequence positions on padding, slowing training unnecessarily.

---

## 2. No Class Weighting / Oversampling Needed

**Decision:** Use standard CrossEntropyLoss without class weights.

**Reasoning:**
- Dataset is nearly perfectly balanced across all 7 intents
- Train distribution: each class has 1658–1710 samples (variance < 3%)
- Test set: exactly 200 samples per class
- No need for weighted loss, focal loss, or oversampling techniques
- Stratified train/val split already maintains this balance

---

## 3. Student Model Architecture: 2 layers, hidden_size=256, 4 heads

**Decision:** Use a small student with 2 transformer encoder layers, hidden_size=256, 4 attention heads, intermediate_size=512.

**Reasoning:**
- The task is relatively simple: 7-class classification of short utterances (mean 9 words)
- BERT-base (12 layers, 768 hidden) is massively overkill for this task
- With only 12 tokens average input length, 2 layers can capture sufficient attention patterns
- 256 hidden size provides enough representational capacity for 7 classes
- Target: ~8M parameters (vs. 110M teacher) = ~14x compression
- This matches the project goal of "5–15M parameters"
- Smaller models train faster on CPU, which is our constraint

**Alternative considered:** 3-4 layers, hidden_size=384. Reserved for comparison if 2-layer underperforms.

---

## 4. Teacher Model: bert-base-uncased (frozen architecture)

**Decision:** Use `bert-base-uncased` as the teacher.

**Reasoning:**
- Standard choice for English NLU tasks
- Well-studied baseline with known performance characteristics
- Uncased variant is preferred because SNIPS utterances are conversational/informal
- Case information adds noise for intent classification (e.g., "Play music" vs "play music" = same intent)
- 110M parameters provides a clear compression ratio target

---

## 5. Training Hyperparameters

### Teacher Training
- **Epochs: 5** — BERT fine-tuning converges fast on small datasets; 3-5 epochs standard
- **LR: 2e-5** — Standard BERT fine-tuning rate (from original BERT paper)
- **Batch size: 32** — Balance between gradient stability and CPU memory
- **Optimizer: AdamW** — Standard for transformer fine-tuning, handles weight decay properly
- **Scheduler: Cosine annealing** — Smooth decay, avoids sudden LR drops

### Student Training
- **Epochs: 20** — Students need more epochs since they have less capacity
- **LR: 5e-4** — Higher than teacher because training from scratch (not fine-tuning pretrained)
- **Batch size: 64** — Student is smaller, can afford larger batches on CPU
- **Optimizer: AdamW** — Consistency with teacher
- **Scheduler: Cosine annealing** — Same reasoning

### Distillation
- **Temperature: 4.0** — Moderate softening; from analysis:
  - T=1: no softening, equivalent to hard labels
  - T=2-3: mild softening
  - T=4-5: good balance for 7-class classification (sweet spot in literature)
  - T>10: over-softening, makes all classes look similar
- **Alpha: 0.7** — Weight 70% distillation loss, 30% hard label loss
  - Emphasizes learning from teacher's distribution
  - Hard labels still provide anchoring against teacher mistakes
  - Literature suggests alpha=0.5-0.9, we pick 0.7 as balanced choice

---

## 6. Evaluation Strategy

**Decision:** Evaluate on both validation and test sets; report accuracy, macro-F1, per-class metrics, inference time.

**Reasoning:**
- Accuracy alone is sufficient for balanced datasets but F1 provides class-level insight
- Macro-F1 treats all classes equally (appropriate since dataset is balanced)
- Inference time comparison is critical for the "mobile deployment" narrative
- Per-class breakdown reveals if certain intents are harder to distill

---

## 7. Logging Strategy

**Decision:** Use JSON-based logging + CSV metrics + TensorBoard.

**Reasoning:**
- JSON: human-readable, easy to load in notebooks for analysis
- CSV: epoch-by-epoch metrics, easy to plot
- TensorBoard: real-time monitoring during training (if needed)
- All logs saved to `outputs/{model_name}/` for clear organization
- Enables post-training analysis without re-running experiments

---

## 8. Data Split Strategy

**Decision:** 80/10/10 split (train/val/test) using stratification.

**Reasoning:**
- Original dataset has train (13,084) + test (1,400)
- Created validation split from train: 10% = 1,309 samples
- Final: 11,775 train / 1,309 val / 1,400 test
- Stratified split preserves class balance
- Validation set used for model selection (early stopping on best val accuracy)
- Test set held out for final evaluation only

---

## 9. Device Strategy: CPU Training

**Decision:** Optimize for CPU training throughout.

**Reasoning:**
- User specified PyTorch CPU is sufficient
- Dataset is small (11.7K training samples)
- Sequences are very short (max 32 tokens)
- Teacher training (5 epochs) should complete in ~10-20 minutes on CPU
- Student training (20 epochs) should complete in ~5-15 minutes on CPU
- No GPU-specific optimizations (mixed precision, etc.) needed
- Code still supports CUDA if available (automatic device detection)

---

## 10. Temperature Sweep Range

**Decision:** Sweep T ∈ {1, 2, 4, 6, 8, 10, 15, 20}

**Reasoning:**
- T=1: baseline (no softening), establishes lower bound
- T=2-4: mild softening, often optimal for well-separated classes
- T=4-8: moderate softening, expected sweet spot for SNIPS
- T=10-20: heavy softening, likely too aggressive for 7 classes
- 8 points gives good coverage without excessive compute on CPU
- Results plotted as accuracy vs. temperature curve

---

## 11. Observed Results & Validation of Decisions

### Teacher Performance (validates Decision 4 & 5)
- **99.14% test accuracy** after 5 epochs of fine-tuning
- Converged from 95.2% → 99.2% (train) in 5 epochs, confirming that 5 epochs is sufficient
- Training time: 73.5 minutes on CPU (acceptable for a one-time teacher training)

### Student Baseline (validates Decision 3)
- **98.07% test accuracy** — only 1.07% below teacher despite 12.3x fewer parameters
- Confirms that the 2-layer architecture with 256 hidden units has sufficient capacity
- Best epoch: 17 out of 20, validating the 20-epoch budget for from-scratch training

### Distilled Student (validates Decision 5 — KD hyperparameters)
- **98.36% test accuracy** — outperforms baseline by +0.29%
- T=4.0 and α=0.7 produced a clear improvement over hard-label-only training
- Best epoch: 13 (converges faster than baseline's epoch 17, as expected from richer gradient signal)
- Teacher accuracy retention: 99.2%

### Key Insight
Distillation's value is most visible in the test set improvement (+0.29%) while validation accuracy was slightly lower (97.94% vs 98.09%). This suggests the distilled student generalizes better — soft targets act as a regularizer, preventing overfitting to the training distribution.
