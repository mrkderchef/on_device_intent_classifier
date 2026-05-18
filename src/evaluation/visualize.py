"""
Visualization utilities for experiment results.
"""

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_training_curves(
    history_path: str,
    output_path: str = "reports/training_curves.png",
    title: str = "Training Curves",
) -> None:
    """Plot training loss and accuracy curves."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, "b-o", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, "b-o", label="Train Accuracy")
    ax2.plot(epochs, val_acc, "r-o", label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_temperature_sweep(
    results_path: str = "outputs/temperature_sweep/sweep_results.json",
    output_path: str = "reports/temperature_sweep.png",
) -> None:
    """Plot temperature sweep results."""
    with open(results_path) as f:
        results = json.load(f)

    temps = [r["temperature"] for r in results]
    val_accs = [r["best_val_acc"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temps, val_accs, "g-o", linewidth=2, markersize=8)
    ax.set_xlabel("Temperature (T)", fontsize=12)
    ax.set_ylabel("Best Validation Accuracy", fontsize=12)
    ax.set_title("Temperature Sweep: Effect on Distillation Performance", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Highlight best
    best_idx = np.argmax(val_accs)
    ax.annotate(
        f"Best: T={temps[best_idx]}\nAcc={val_accs[best_idx]:.4f}",
        xy=(temps[best_idx], val_accs[best_idx]),
        xytext=(temps[best_idx] + 1, val_accs[best_idx] - 0.01),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison(
    comparison_path: str = "reports/comparison_report.json",
    output_path: str = "reports/model_comparison.png",
) -> None:
    """Plot bar chart comparing teacher vs student metrics."""
    with open(comparison_path) as f:
        comparison = json.load(f)

    teacher = comparison["teacher"]
    student = comparison["student"]

    metrics = ["accuracy", "f1_macro"]
    teacher_vals = [teacher[m] for m in metrics]
    student_vals = [student[m] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Performance comparison
    x = np.arange(len(metrics))
    width = 0.35
    axes[0].bar(x - width/2, teacher_vals, width, label="Teacher (BERT)", color="steelblue")
    axes[0].bar(x + width/2, student_vals, width, label="Student (Distilled)", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Accuracy", "F1 (Macro)"])
    axes[0].set_ylabel("Score")
    axes[0].set_title("Performance Comparison")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Size comparison
    sizes = [teacher["parameters"] / 1e6, student["parameters"] / 1e6]
    axes[1].bar(["Teacher", "Student"], sizes, color=["steelblue", "coral"])
    axes[1].set_ylabel("Parameters (Millions)")
    axes[1].set_title("Model Size")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Speed comparison
    speeds = [teacher["avg_sample_time_ms"], student["avg_sample_time_ms"]]
    axes[2].bar(["Teacher", "Student"], speeds, color=["steelblue", "coral"])
    axes[2].set_ylabel("Inference Time (ms/sample)")
    axes[2].set_title("Inference Speed")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Teacher vs. Distilled Student: Comprehensive Comparison", fontsize=14)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    output_path: str = "reports/confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
