"""
Evaluation utilities for intent classification models.

Provides comprehensive metrics including accuracy, precision, recall, F1,
as well as model efficiency metrics (parameter count, memory, inference speed).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    device: str = "cpu",
    return_predictions: bool = False,
) -> dict:
    """
    Evaluate model on a dataloader.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader with test/val data
        device: Device to run evaluation on
        return_predictions: Whether to return raw predictions

    Returns:
        dict with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"]
        predictions = logits.argmax(dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision_macro": precision_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            all_labels, all_predictions, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            all_labels, all_predictions, average="weighted", zero_division=0
        ),
    }

    if return_predictions:
        metrics["predictions"] = all_predictions
        metrics["labels"] = all_labels
        metrics["confusion_matrix"] = confusion_matrix(all_labels, all_predictions).tolist()

    return metrics


@torch.no_grad()
def measure_inference_speed(
    model: nn.Module,
    dataloader,
    device: str = "cpu",
    num_batches: int = 50,
) -> dict:
    """
    Measure model inference speed.

    Returns:
        dict with timing metrics
    """
    model.eval()
    times = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Warmup
        if i == 0:
            _ = model(input_ids, attention_mask)

        start = time.perf_counter()
        _ = model(input_ids, attention_mask)
        end = time.perf_counter()

        times.append(end - start)

    times = np.array(times)
    batch_size = dataloader.batch_size

    return {
        "avg_batch_time_ms": float(np.mean(times) * 1000),
        "std_batch_time_ms": float(np.std(times) * 1000),
        "avg_sample_time_ms": float(np.mean(times) * 1000 / batch_size),
        "throughput_samples_per_sec": float(batch_size / np.mean(times)),
    }


def compare_models(
    teacher: nn.Module,
    student: nn.Module,
    dataloader,
    device: str = "cpu",
    label_names: list = None,
) -> dict:
    """
    Compare teacher and student models comprehensively.

    Returns:
        dict with comparison metrics for both models
    """
    print("Evaluating teacher model...")
    teacher_metrics = evaluate_model(teacher, dataloader, device, return_predictions=True)
    teacher_speed = measure_inference_speed(teacher, dataloader, device)

    print("Evaluating student model...")
    student_metrics = evaluate_model(student, dataloader, device, return_predictions=True)
    student_speed = measure_inference_speed(student, dataloader, device)

    # Model size comparison
    teacher_params = teacher.get_num_parameters()
    student_params = student.get_num_parameters()
    teacher_size = teacher.get_model_size_mb()
    student_size = student.get_model_size_mb()

    comparison = {
        "teacher": {
            "accuracy": teacher_metrics["accuracy"],
            "f1_macro": teacher_metrics["f1_macro"],
            "parameters": teacher_params,
            "size_mb": teacher_size,
            **teacher_speed,
        },
        "student": {
            "accuracy": student_metrics["accuracy"],
            "f1_macro": student_metrics["f1_macro"],
            "parameters": student_params,
            "size_mb": student_size,
            **student_speed,
        },
        "compression": {
            "parameter_ratio": teacher_params / student_params,
            "size_ratio": teacher_size / student_size,
            "speed_ratio": teacher_speed["avg_sample_time_ms"] / student_speed["avg_sample_time_ms"],
            "accuracy_retention": student_metrics["accuracy"] / teacher_metrics["accuracy"] * 100,
            "f1_retention": student_metrics["f1_macro"] / teacher_metrics["f1_macro"] * 100,
        },
    }

    # Print summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Teacher':<20} {'Student':<20} {'Ratio':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {teacher_metrics['accuracy']:<20.4f} {student_metrics['accuracy']:<20.4f} {comparison['compression']['accuracy_retention']:.1f}%")
    print(f"{'F1 (macro)':<25} {teacher_metrics['f1_macro']:<20.4f} {student_metrics['f1_macro']:<20.4f} {comparison['compression']['f1_retention']:.1f}%")
    print(f"{'Parameters':<25} {teacher_params:<20,} {student_params:<20,} {comparison['compression']['parameter_ratio']:.1f}x")
    print(f"{'Size (MB)':<25} {teacher_size:<20.2f} {student_size:<20.2f} {comparison['compression']['size_ratio']:.1f}x")
    print(f"{'Inference (ms/sample)':<25} {teacher_speed['avg_sample_time_ms']:<20.3f} {student_speed['avg_sample_time_ms']:<20.3f} {comparison['compression']['speed_ratio']:.1f}x")
    print("=" * 70)

    return comparison


def generate_report(
    comparison: dict,
    output_path: str = "reports/comparison_report.json",
) -> None:
    """Save comparison report to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument(
        "--model", type=str, required=True, choices=["teacher", "student", "compare"]
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--teacher_path", type=str, default="outputs/teacher/best_model.pt")
    parser.add_argument("--student_path", type=str, default="outputs/student/distill/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    from src.data.dataset import create_dataloaders, get_label_map

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_map = get_label_map(args.data_dir)
    num_labels = len(label_map)

    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    if args.model == "teacher":
        from src.models.teacher import TeacherModel
        model = TeacherModel(num_labels=num_labels)
        model.load_state_dict(torch.load(args.teacher_path, map_location=device))
        model = model.to(device)
        metrics = evaluate_model(model, test_loader, device, return_predictions=True)
        print(json.dumps({k: v for k, v in metrics.items() if isinstance(v, float)}, indent=2))

    elif args.model == "student":
        from src.models.student import StudentModel
        model = StudentModel(num_labels=num_labels)
        model.load_state_dict(torch.load(args.student_path, map_location=device))
        model = model.to(device)
        metrics = evaluate_model(model, test_loader, device, return_predictions=True)
        print(json.dumps({k: v for k, v in metrics.items() if isinstance(v, float)}, indent=2))

    elif args.model == "compare":
        from src.models.teacher import TeacherModel
        from src.models.student import StudentModel

        teacher = TeacherModel(num_labels=num_labels)
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
        teacher = teacher.to(device)

        student = StudentModel(num_labels=num_labels)
        student.load_state_dict(torch.load(args.student_path, map_location=device))
        student = student.to(device)

        comparison = compare_models(teacher, student, test_loader, device)
        generate_report(comparison)
