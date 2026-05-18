"""
Temperature sweep experiment for knowledge distillation.

Evaluates distillation performance across different temperature values
to find the optimal softening of probability distributions.
"""

import argparse
import json
from pathlib import Path

import torch

from src.config import DataConfig, StudentConfig, DistillationConfig, TemperatureSweepConfig
from src.training.train_student import train_student
from src.training.logger import TrainingLogger


def temperature_sweep(
    data_config: DataConfig = None,
    student_config: StudentConfig = None,
    sweep_config: TemperatureSweepConfig = None,
    device: str = None,
):
    """Run distillation experiments across multiple temperatures."""
    if data_config is None:
        data_config = DataConfig()
    if student_config is None:
        student_config = StudentConfig()
    if sweep_config is None:
        sweep_config = TemperatureSweepConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = Path(sweep_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sweep logger
    logger = TrainingLogger(sweep_config.output_dir, "temperature_sweep")
    logger.log_config({
        "temperatures": sweep_config.temperatures,
        "epochs_per_temp": sweep_config.epochs_per_temp,
        "student_hidden_size": student_config.hidden_size,
        "student_num_layers": student_config.num_layers,
        "alpha": 0.7,
        "device": device,
    })

    print("=" * 60)
    print("TEMPERATURE SWEEP EXPERIMENT")
    print(f"Temperatures: {sweep_config.temperatures}")
    print(f"Epochs per temperature: {sweep_config.epochs_per_temp}")
    print(f"Device: {device}")
    print("=" * 60)

    results = []
    logger.start_training()

    for i, temp in enumerate(sweep_config.temperatures, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(sweep_config.temperatures)}] Training with Temperature = {temp}")
        print(f"{'='*60}")

        # Override configs for this temperature
        student_cfg_copy = StudentConfig(
            hidden_size=student_config.hidden_size,
            num_layers=student_config.num_layers,
            num_heads=student_config.num_heads,
            intermediate_size=student_config.intermediate_size,
            dropout=student_config.dropout,
            epochs=sweep_config.epochs_per_temp,
            batch_size=student_config.batch_size,
            learning_rate=student_config.learning_rate,
            weight_decay=student_config.weight_decay,
            output_dir=str(output_path / f"T_{temp}"),
        )

        distill_cfg = DistillationConfig(
            temperature=temp,
            alpha=0.7,
        )

        student = train_student(
            mode="distill",
            data_config=data_config,
            student_config=student_cfg_copy,
            distill_config=distill_cfg,
            device=device,
        )

        # Load the results from the run
        results_file = Path(student_cfg_copy.output_dir) / "distill" / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                run_results = json.load(f)
            best_val = run_results.get("final_results", {}).get("best_val_acc", 0)
            test_acc = run_results.get("final_results", {}).get("test", {}).get("accuracy", 0)
            test_f1 = run_results.get("final_results", {}).get("test", {}).get("f1_macro", 0)
        else:
            best_val = 0
            test_acc = 0
            test_f1 = 0

        result = {
            "temperature": temp,
            "best_val_acc": best_val,
            "test_acc": test_acc,
            "test_f1": test_f1,
        }
        results.append(result)

        logger.log_epoch(i, {
            "temperature": temp,
            "best_val_acc": best_val,
            "test_acc": test_acc,
            "test_f1": test_f1,
        })

    # Summary
    print("\n" + "=" * 60)
    print("TEMPERATURE SWEEP RESULTS")
    print("=" * 60)
    print(f"{'Temperature':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 48)
    for r in results:
        print(f"{r['temperature']:<12.1f} {r['best_val_acc']:<12.4f} {r['test_acc']:<12.4f} {r['test_f1']:<12.4f}")

    best_result = max(results, key=lambda x: x["best_val_acc"])
    print(f"\nBest: T={best_result['temperature']} (Val Acc: {best_result['best_val_acc']:.4f})")

    # Save sweep results
    with open(output_path / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.log_final_results({"sweep_results": results, "best": best_result})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temperature sweep experiment")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs/temperature_sweep")
    parser.add_argument("--teacher_path", type=str, default="outputs/teacher/best_model.pt")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1, 2, 4, 6, 8, 10, 15, 20])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    data_cfg = DataConfig(data_dir=args.data_dir)
    student_cfg = StudentConfig(batch_size=args.batch_size, learning_rate=args.lr)
    sweep_cfg = TemperatureSweepConfig(
        temperatures=args.temperatures,
        epochs_per_temp=args.epochs,
        output_dir=args.output_dir,
    )
    temperature_sweep(
        data_config=data_cfg,
        student_config=student_cfg,
        sweep_config=sweep_cfg,
    )
