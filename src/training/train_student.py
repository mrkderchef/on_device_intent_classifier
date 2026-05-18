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

from src.config import DataConfig, StudentConfig, DistillationConfig
from src.data.dataset import create_dataloaders, get_label_map
from src.models.student import StudentModel
from src.models.teacher import TeacherModel
from src.evaluation.evaluate import evaluate_model
from src.training.logger import TrainingLogger


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss.

    L = alpha * T^2 * KL(p_teacher || p_student) + (1 - alpha) * CE(y, p_student)

    Args:
        temperature: Softmax temperature for softening distributions
        alpha: Weight for distillation loss vs. hard-label loss
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
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
    data_config: DataConfig = None,
    student_config: StudentConfig = None,
    distill_config: DistillationConfig = None,
    device: str = None,
):
    """
    Train student model.

    Args:
        mode: 'baseline' (hard labels only) or 'distill' (with teacher KD)
    """
    if data_config is None:
        data_config = DataConfig()
    if student_config is None:
        student_config = StudentConfig()
    if distill_config is None:
        distill_config = DistillationConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = f"{student_config.output_dir}/{mode}"

    # Setup logger
    logger = TrainingLogger(output_dir, f"student_{mode}")
    logger.log_config({
        "mode": mode,
        "hidden_size": student_config.hidden_size,
        "num_layers": student_config.num_layers,
        "num_heads": student_config.num_heads,
        "intermediate_size": student_config.intermediate_size,
        "epochs": student_config.epochs,
        "batch_size": student_config.batch_size,
        "learning_rate": student_config.learning_rate,
        "max_length": data_config.max_length,
        "temperature": distill_config.temperature if mode == "distill" else None,
        "alpha": distill_config.alpha if mode == "distill" else None,
        "device": device,
    })

    print("=" * 60)
    print(f"STUDENT MODEL TRAINING (mode={mode})")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Architecture: {student_config.num_layers} layers, h={student_config.hidden_size}")
    if mode == "distill":
        print(f"Temperature: {distill_config.temperature}")
        print(f"Alpha: {distill_config.alpha}")
    print(f"Epochs: {student_config.epochs}")
    print(f"Batch size: {student_config.batch_size}")
    print(f"Learning rate: {student_config.learning_rate}")
    print("=" * 60)

    # Create dataloaders
    label_map = get_label_map(data_config.data_dir)
    num_labels = len(label_map)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_config.data_dir,
        tokenizer_name=data_config.tokenizer_name,
        max_length=data_config.max_length,
        batch_size=student_config.batch_size,
    )

    # Initialize student model
    student = StudentModel(
        num_labels=num_labels,
        hidden_size=student_config.hidden_size,
        num_layers=student_config.num_layers,
        num_heads=student_config.num_heads,
        intermediate_size=student_config.intermediate_size,
        max_length=data_config.max_length,
        dropout=student_config.dropout,
    )
    student = student.to(device)
    print(f"Student parameters: {student.get_num_parameters():,}")
    print(f"Student size: {student.get_model_size_mb():.2f} MB")

    # Load teacher for distillation mode
    teacher = None
    if mode == "distill":
        teacher = TeacherModel(num_labels=num_labels)
        teacher.load_state_dict(
            torch.load(distill_config.teacher_path, map_location=device, weights_only=True)
        )
        teacher = teacher.to(device)
        teacher.eval()
        print(f"Teacher loaded from: {distill_config.teacher_path}")
        print(f"Teacher parameters: {teacher.get_num_parameters():,}")
        compression_ratio = teacher.get_num_parameters() / student.get_num_parameters()
        print(f"Compression ratio: {compression_ratio:.1f}x")

    print("=" * 60)

    # Loss and optimizer
    if mode == "distill":
        criterion = DistillationLoss(
            temperature=distill_config.temperature, alpha=distill_config.alpha
        )
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        student.parameters(),
        lr=student_config.learning_rate,
        weight_decay=student_config.weight_decay,
    )
    total_steps = len(train_loader) * student_config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    logger.start_training()
    best_val_acc = 0.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, student_config.epochs + 1):
        student.train()
        total_loss = 0.0
        total_distill_loss = 0.0
        total_ce_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{student_config.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # Student forward pass
            student_outputs = student(input_ids, attention_mask)
            student_logits = student_outputs["logits"]

            if mode == "distill":
                with torch.no_grad():
                    teacher_outputs = teacher(input_ids, attention_mask)
                    teacher_logits = teacher_outputs["logits"]

                loss_dict = criterion(student_logits, teacher_logits, labels)
                loss = loss_dict["loss"]
                total_distill_loss += loss_dict["distill_loss"]
                total_ce_loss += loss_dict["ce_loss"]
            else:
                loss = criterion(student_logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=student_config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            predictions = student_logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        val_metrics = evaluate_model(student, val_loader, device) if val_loader else {}
        val_acc = val_metrics.get("accuracy", 0.0)
        val_f1 = val_metrics.get("f1_macro", 0.0)

        # Log epoch metrics
        epoch_metrics = {
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": scheduler.get_last_lr()[0],
        }
        if mode == "distill":
            epoch_metrics["distill_loss"] = total_distill_loss / len(train_loader)
            epoch_metrics["ce_loss"] = total_ce_loss / len(train_loader)

        logger.log_epoch(epoch, epoch_metrics)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), output_path / "best_model.pt")
            print(f"  → New best model saved (Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f})")

    # Save final model
    torch.save(student.state_dict(), output_path / "final_model.pt")

    # Final test evaluation
    final_results = {"best_val_acc": best_val_acc, "mode": mode}
    if test_loader:
        student.load_state_dict(torch.load(output_path / "best_model.pt", weights_only=True))
        test_metrics = evaluate_model(student, test_loader, device)
        final_results["test"] = test_metrics
        print("\n" + "=" * 60)
        print(f"TEST RESULTS ({mode} student):")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        print("=" * 60)

    final_results["num_parameters"] = student.get_num_parameters()
    final_results["model_size_mb"] = student.get_model_size_mb()
    if mode == "distill" and teacher:
        final_results["compression_ratio"] = teacher.get_num_parameters() / student.get_num_parameters()

    logger.log_final_results(final_results)
    return student


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student model")
    parser.add_argument("--mode", type=str, default="distill", choices=["baseline", "distill"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs/student")
    parser.add_argument("--teacher_path", type=str, default="outputs/teacher/best_model.pt")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--max_length", type=int, default=32)
    args = parser.parse_args()

    data_cfg = DataConfig(data_dir=args.data_dir, max_length=args.max_length)
    student_cfg = StudentConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
    distill_cfg = DistillationConfig(
        temperature=args.temperature,
        alpha=args.alpha,
        teacher_path=args.teacher_path,
    )
    train_student(
        mode=args.mode,
        data_config=data_cfg,
        student_config=student_cfg,
        distill_config=distill_cfg,
    )
