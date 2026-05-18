"""
Temperature sweep experiment for knowledge distillation.

Evaluates distillation performance across different temperature values
to find the optimal softening of probability distributions.
"""

import argparse
import json
from pathlib import Path

import torch

from src.data.dataset import create_dataloaders, get_label_map
from src.models.student import StudentModel
from src.models.teacher import TeacherModel
from src.training.train_student import train_student
from src.evaluation.evaluate import evaluate_model


def temperature_sweep(
    temperatures: list = None,
    data_dir: str = "data/processed",
    output_dir: str = "outputs/temperature_sweep",
    teacher_path: str = "outputs/teacher/best_model.pt",
    hidden_size: int = 256,
    num_layers: int = 3,
    num_heads: int = 4,
    intermediate_size: int = 512,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    alpha: float = 0.7,
    max_length: int = 64,
    device: str = None,
):
    """Run distillation experiments across multiple temperatures."""
    if temperatures is None:
        temperatures = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TEMPERATURE SWEEP EXPERIMENT")
    print(f"Temperatures: {temperatures}")
    print(f"Device: {device}")
    print("=" * 60)

    results = []

    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"Training with Temperature = {temp}")
        print(f"{'='*60}")

        temp_output_dir = str(output_path / f"T_{temp}")

        student, history = train_student(
            mode="distill",
            data_dir=data_dir,
            output_dir=temp_output_dir,
            teacher_path=teacher_path,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            temperature=temp,
            alpha=alpha,
            max_length=max_length,
            device=device,
        )

        # Get best validation accuracy from history
        best_val = max(h["val_acc"] for h in history)

        results.append({
            "temperature": temp,
            "best_val_acc": best_val,
            "final_train_acc": history[-1]["train_acc"],
            "final_train_loss": history[-1]["train_loss"],
        })

        print(f"T={temp}: Best Val Acc = {best_val:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("TEMPERATURE SWEEP RESULTS")
    print("=" * 60)
    print(f"{'Temperature':<15} {'Best Val Acc':<15} {'Final Train Acc':<15}")
    print("-" * 45)
    for r in results:
        print(f"{r['temperature']:<15.1f} {r['best_val_acc']:<15.4f} {r['final_train_acc']:<15.4f}")

    best_result = max(results, key=lambda x: x["best_val_acc"])
    print(f"\nBest temperature: T={best_result['temperature']} (Val Acc: {best_result['best_val_acc']:.4f})")

    # Save results
    with open(output_path / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temperature sweep experiment")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs/temperature_sweep")
    parser.add_argument("--teacher_path", type=str, default="outputs/teacher/best_model.pt")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1, 2, 3, 5, 10, 20])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()

    temperature_sweep(
        temperatures=args.temperatures,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        teacher_path=args.teacher_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        alpha=args.alpha,
    )
