"""
Configuration for all training experiments.

Central place for hyperparameters and paths, derived from dataset exploration.
See DECISIONS.md for reasoning behind each choice.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_dir: str = "data/processed"
    max_length: int = 32  # 100% of SNIPS fits in 32 BERT tokens
    num_labels: int = 7
    tokenizer_name: str = "bert-base-uncased"


@dataclass
class TeacherConfig:
    """Teacher model (BERT-base) training configuration."""
    model_name: str = "bert-base-uncased"
    epochs: int = 5  # BERT converges fast on small datasets
    batch_size: int = 32
    learning_rate: float = 2e-5  # Standard BERT fine-tuning rate
    weight_decay: float = 0.01
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    output_dir: str = "outputs/teacher"


@dataclass
class StudentConfig:
    """Student model training configuration."""
    hidden_size: int = 256
    num_layers: int = 2  # Sufficient for short, well-separated texts
    num_heads: int = 4
    intermediate_size: int = 512
    dropout: float = 0.1
    epochs: int = 20  # Students need more epochs (training from scratch)
    batch_size: int = 64  # Smaller model allows larger batches
    learning_rate: float = 5e-4  # Higher LR for training from scratch
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    output_dir: str = "outputs/student"


@dataclass
class DistillationConfig:
    """Knowledge distillation configuration."""
    temperature: float = 4.0  # Moderate softening for 7-class task
    alpha: float = 0.7  # 70% distillation, 30% hard labels
    teacher_path: str = "outputs/teacher/best_model.pt"


@dataclass
class TemperatureSweepConfig:
    """Temperature sweep experiment configuration."""
    temperatures: list = field(default_factory=lambda: [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0])
    epochs_per_temp: int = 15
    output_dir: str = "outputs/temperature_sweep"
