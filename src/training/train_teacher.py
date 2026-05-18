"""
Training script for the teacher model (BERT-base fine-tuning).
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.dataset import create_dataloaders, get_label_map
from src.models.teacher import TeacherModel
from src.evaluation.evaluate import evaluate_model


def train_teacher(
    data_dir: str = "data/processed",
    output_dir: str = "outputs/teacher",
    model_name: str = "bert-base-uncased",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 64,
    device: str = None,
):
    """Train BERT-base teacher model on SNIPS dataset."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)

    # Create dataloaders
    label_map = get_label_map(data_dir)
    num_labels = len(label_map)
    print(f"Number of intent classes: {num_labels}")
    print(f"Labels: {list(label_map.keys())}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        tokenizer_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    # Initialize model
    model = TeacherModel(num_labels=num_labels, model_name=model_name)
    model = model.to(device)
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    print("=" * 60)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    best_val_acc = 0.0
    training_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            predictions = outputs["logits"].argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}"
            )

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device) if val_loader else {}
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
            torch.save(model.state_dict(), output_path / "best_model.pt")
            print(f"  -> New best model saved (Val Acc: {val_acc:.4f})")

    # Save final model
    torch.save(model.state_dict(), output_path / "final_model.pt")

    # Final test evaluation
    if test_loader:
        model.load_state_dict(torch.load(output_path / "best_model.pt"))
        test_metrics = evaluate_model(model, test_loader, device)
        print("\n" + "=" * 60)
        print("TEST RESULTS (Best Model):")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
        print("=" * 60)

    # Save training history
    import json
    with open(output_path / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    return model, training_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT teacher model")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs/teacher")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    train_teacher(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )
