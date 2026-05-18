"""
Training script for the student model with optional knowledge distillation.

Supports two modes:
- baseline: Train student from scratch with hard labels only
- distill: Train student with knowledge distillation from teacher
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.dataset import create_dataloaders, get_label_map
from src.models.student import StudentModel, create_student_medium
from src.models.teacher import TeacherModel
from src.evaluation.evaluate import evaluate_model


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss.

    L = alpha * T^2 * KL(p_teacher || p_student) + (1 - alpha) * CE(y, p_student)

    Args:
        temperature: Softmax temperature for softening distributions
        alpha: Weight for distillation loss vs. hard-label loss
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        Compute distillation loss.

        Args:
            student_logits: Raw logits from student model
            teacher_logits: Raw logits from teacher model
            labels: Ground truth labels

        Returns:
            dict with total_loss, distill_loss, ce_loss
        """
        T = self.temperature

        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        # KL divergence (scaled by T^2)
        distill_loss = F.kl_div(
            student_log_probs, teacher_probs, reduction="batchmean"
        ) * (T ** 2)

        # Hard label cross-entropy
        ce_loss = self.ce_loss(student_logits, labels)

        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * ce_loss

        return {
            "loss": total_loss,
            "distill_loss": distill_loss.item(),
            "ce_loss": ce_loss.item(),
        }


def train_student(
    mode: str = "distill",
    data_dir: str = "data/processed",
    output_dir: str = "outputs/student",
    teacher_path: str = "outputs/teacher/best_model.pt",
    hidden_size: int = 256,
    num_layers: int = 3,
    num_heads: int = 4,
    intermediate_size: int = 512,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    temperature: float = 3.0,
    alpha: float = 0.7,
    max_length: int = 64,
    device: str = None,
):
    """
    Train student model.

    Args:
        mode: 'baseline' (hard labels only) or 'distill' (with teacher KD)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = Path(output_dir) / mode
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Temperature: {temperature}")
    print(f"Alpha: {alpha}")
    print("=" * 60)

    # Create dataloaders
    label_map = get_label_map(data_dir)
    num_labels = len(label_map)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        tokenizer_name="bert-base-uncased",
        max_length=max_length,
        batch_size=batch_size,
    )

    # Initialize student model
    student = StudentModel(
        num_labels=num_labels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_length=max_length,
    )
    student = student.to(device)
    print(f"Student parameters: {student.get_num_parameters():,}")
    print(f"Student size: {student.get_model_size_mb():.2f} MB")

    # Load teacher for distillation mode
    teacher = None
    if mode == "distill":
        teacher = TeacherModel(num_labels=num_labels)
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        teacher = teacher.to(device)
        teacher.eval()
        print(f"Teacher loaded from: {teacher_path}")
        print(f"Teacher parameters: {teacher.get_num_parameters():,}")
        compression_ratio = teacher.get_num_parameters() / student.get_num_parameters()
        print(f"Compression ratio: {compression_ratio:.1f}x")

    print("=" * 60)

    # Loss and optimizer
    if mode == "distill":
        criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        student.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    best_val_acc = 0.0
    training_history = []

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # Student forward pass
            student_outputs = student(input_ids, attention_mask)
            student_logits = student_outputs["logits"]

            if mode == "distill":
                # Get teacher predictions (no gradient needed)
                with torch.no_grad():
                    teacher_outputs = teacher(input_ids, attention_mask)
                    teacher_logits = teacher_outputs["logits"]

                loss_dict = criterion(student_logits, teacher_logits, labels)
                loss = loss_dict["loss"]
            else:
                loss = criterion(student_logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            predictions = student_logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}"
            )

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        val_metrics = evaluate_model(student, val_loader, device) if val_loader else {}
        val_acc = val_metrics.get("accuracy", 0.0)

        print(
            f"\nEpoch {epoch+1}: "
            f"Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Acc={val_acc:.4f}"
        )

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), output_path / "best_model.pt")
            print(f"  -> New best model saved (Val Acc: {val_acc:.4f})")

    # Save final model
    torch.save(student.state_dict(), output_path / "final_model.pt")

    # Final test evaluation
    if test_loader:
        student.load_state_dict(torch.load(output_path / "best_model.pt"))
        test_metrics = evaluate_model(student, test_loader, device)
        print("\n" + "=" * 60)
        print(f"TEST RESULTS ({mode} student):")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        print("=" * 60)

    # Save training config and history
    config = {
        "mode": mode,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "intermediate_size": intermediate_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "alpha": alpha,
        "best_val_acc": best_val_acc,
        "num_parameters": student.get_num_parameters(),
        "model_size_mb": student.get_model_size_mb(),
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(output_path / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    return student, training_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student model")
    parser.add_argument(
        "--mode", type=str, default="distill", choices=["baseline", "distill"]
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs/student")
    parser.add_argument("--teacher_path", type=str, default="outputs/teacher/best_model.pt")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    train_student(
        mode=args.mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        teacher_path=args.teacher_path,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        alpha=args.alpha,
        max_length=args.max_length,
    )
