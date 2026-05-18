"""
Training script for the teacher model (BERT-base fine-tuning).
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import DataConfig, TeacherConfig
from src.data.dataset import create_dataloaders, get_label_map
from src.models.teacher import TeacherModel
from src.evaluation.evaluate import evaluate_model
from src.training.logger import TrainingLogger


def train_teacher(
    data_config: DataConfig = None,
    teacher_config: TeacherConfig = None,
    device: str = None,
):
    """Train BERT-base teacher model on SNIPS dataset."""
    if data_config is None:
        data_config = DataConfig()
    if teacher_config is None:
        teacher_config = TeacherConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup logger
    logger = TrainingLogger(teacher_config.output_dir, "teacher_bert_base")
    logger.log_config({
        "model_name": teacher_config.model_name,
        "epochs": teacher_config.epochs,
        "batch_size": teacher_config.batch_size,
        "learning_rate": teacher_config.learning_rate,
        "weight_decay": teacher_config.weight_decay,
        "max_length": data_config.max_length,
        "device": device,
    })

    print("=" * 60)
    print("TEACHER MODEL TRAINING (BERT-base)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {teacher_config.model_name}")
    print(f"Max length: {data_config.max_length}")
    print(f"Epochs: {teacher_config.epochs}")
    print(f"Batch size: {teacher_config.batch_size}")
    print(f"Learning rate: {teacher_config.learning_rate}")
    print("=" * 60)

    # Create dataloaders
    label_map = get_label_map(data_config.data_dir)
    num_labels = len(label_map)
    print(f"Number of intent classes: {num_labels}")
    print(f"Labels: {list(label_map.keys())}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_config.data_dir,
        tokenizer_name=data_config.tokenizer_name,
        max_length=data_config.max_length,
        batch_size=teacher_config.batch_size,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader) if val_loader else 0}")
    print(f"Test batches: {len(test_loader) if test_loader else 0}")

    # Initialize model
    model = TeacherModel(
        num_labels=num_labels,
        model_name=teacher_config.model_name,
        dropout=teacher_config.dropout,
    )
    model = model.to(device)
    print(f"\nModel parameters: {model.get_num_parameters():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    print("=" * 60)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=teacher_config.learning_rate,
        weight_decay=teacher_config.weight_decay,
    )
    total_steps = len(train_loader) * teacher_config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    logger.start_training()
    best_val_acc = 0.0
    output_path = Path(teacher_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, teacher_config.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{teacher_config.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=teacher_config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            predictions = outputs["logits"].argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device) if val_loader else {}
        val_acc = val_metrics.get("accuracy", 0.0)
        val_f1 = val_metrics.get("f1_macro", 0.0)

        # Log epoch
        logger.log_epoch(epoch, {
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": scheduler.get_last_lr()[0],
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path / "best_model.pt")
            print(f"  → New best model saved (Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f})")

    # Save final model
    torch.save(model.state_dict(), output_path / "final_model.pt")

    # Final test evaluation
    final_results = {"best_val_acc": best_val_acc}
    if test_loader:
        model.load_state_dict(torch.load(output_path / "best_model.pt", weights_only=True))
        test_metrics = evaluate_model(model, test_loader, device)
        final_results["test"] = test_metrics
        print("\n" + "=" * 60)
        print("TEST RESULTS (Best Model):")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        print("=" * 60)

    final_results["num_parameters"] = model.get_num_parameters()
    final_results["model_size_mb"] = model.get_model_size_mb()
    logger.log_final_results(final_results)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT teacher model")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs/teacher")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=32)
    args = parser.parse_args()

    data_cfg = DataConfig(data_dir=args.data_dir, max_length=args.max_length)
    teacher_cfg = TeacherConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
    train_teacher(data_config=data_cfg, teacher_config=teacher_cfg)
