"""
Generate all report visualizations from experiment results.
Run from project root: python -m src.evaluation.generate_reports
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


def load_results():
    """Load all experiment results."""
    with open("outputs/teacher/results.json") as f:
        teacher = json.load(f)
    with open("outputs/student/baseline/results.json") as f:
        baseline = json.load(f)
    with open("outputs/student/distill/results.json") as f:
        distill = json.load(f)
    return teacher, baseline, distill


def plot_model_comparison(teacher, baseline, distill, output_dir):
    """Generate model comparison bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = ["Teacher\n(BERT-base)", "Student\n(Baseline)", "Student\n(Distilled)"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # Accuracy
    accs = [r["final_results"]["test"]["accuracy"] for r in [teacher, baseline, distill]]
    bars = axes[0].bar(models, accs, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylim(0.97, 1.0)
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    # F1
    f1s = [r["final_results"]["test"]["f1_macro"] for r in [teacher, baseline, distill]]
    bars = axes[1].bar(models, f1s, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("F1 Score (Macro)")
    axes[1].set_title("Test F1 (Macro)")
    axes[1].set_ylim(0.97, 1.0)
    for bar, val in zip(bars, f1s):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    # Size
    sizes = [r["final_results"]["model_size_mb"] for r in [teacher, baseline, distill]]
    bars = axes[2].bar(models, sizes, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].set_ylabel("Model Size (MB)")
    axes[2].set_title("Model Size")
    for bar, val in zip(bars, sizes):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{val:.1f} MB", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Teacher vs. Student Models: Performance & Efficiency", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'model_comparison.png'}")


def plot_training_curves(teacher, baseline, distill, output_dir):
    """Generate training dynamics plots."""
    teacher_metrics = pd.read_csv("outputs/teacher/metrics.csv")
    baseline_metrics = pd.read_csv("outputs/student/baseline/metrics.csv")
    distill_metrics = pd.read_csv("outputs/student/distill/metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Teacher loss
    axes[0, 0].plot(teacher_metrics["epoch"], teacher_metrics["train_loss"], "b-o", markersize=4)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Teacher: Training Loss")

    # Teacher accuracy
    axes[0, 1].plot(teacher_metrics["epoch"], teacher_metrics["val_acc"], "r-o", markersize=4)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Teacher: Validation Accuracy")
    axes[0, 1].set_ylim(0.98, 1.0)

    # Student val accuracy comparison
    axes[1, 0].plot(baseline_metrics["epoch"], baseline_metrics["val_acc"], "--o", color="#DD8452",
                   markersize=3, label="Baseline")
    axes[1, 0].plot(distill_metrics["epoch"], distill_metrics["val_acc"], "-o", color="#55A868",
                   markersize=3, label="Distilled")
    axes[1, 0].axhline(y=teacher["final_results"]["best_val_acc"], color="#4C72B0",
                       linestyle=":", alpha=0.7, label="Teacher Best Val")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Validation Accuracy")
    axes[1, 0].set_title("Student Models: Validation Accuracy")
    axes[1, 0].legend()

    # Student training loss comparison
    axes[1, 1].plot(baseline_metrics["epoch"], baseline_metrics["train_loss"], "--o", color="#DD8452",
                   markersize=3, label="Baseline (CE Loss)")
    axes[1, 1].plot(distill_metrics["epoch"], distill_metrics["train_loss"], "-o", color="#55A868",
                   markersize=3, label="Distilled (KD + CE)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Training Loss")
    axes[1, 1].set_title("Student Models: Training Loss")
    axes[1, 1].legend()

    plt.suptitle("Training Dynamics Across All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'training_curves.png'}")


def plot_distillation_losses(distill, output_dir):
    """Plot distillation loss component breakdown."""
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = [h["epoch"] for h in distill["epoch_history"]]
    distill_loss = [h["distill_loss"] for h in distill["epoch_history"]]
    ce_loss = [h["ce_loss"] for h in distill["epoch_history"]]
    total_loss = [h["train_loss"] for h in distill["epoch_history"]]

    ax.plot(epochs, distill_loss, "-o", color="#C44E52", markersize=4, label="KL Divergence (soft targets)")
    ax.plot(epochs, ce_loss, "-s", color="#4C72B0", markersize=4, label="Cross-Entropy (hard labels)")
    ax.plot(epochs, total_loss, "-^", color="#55A868", markersize=4, label="Combined Loss (α=0.7)")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Knowledge Distillation: Loss Component Breakdown", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "distillation_loss_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'distillation_loss_breakdown.png'}")


def plot_efficiency_scatter(teacher, baseline, distill, output_dir):
    """Plot F1 vs model size scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))

    models_data = [
        ("Teacher (BERT-base)", teacher["final_results"]["model_size_mb"],
         teacher["final_results"]["test"]["f1_macro"], teacher["final_results"]["num_parameters"]),
        ("Student (Baseline)", baseline["final_results"]["model_size_mb"],
         baseline["final_results"]["test"]["f1_macro"], baseline["final_results"]["num_parameters"]),
        ("Student (Distilled)", distill["final_results"]["model_size_mb"],
         distill["final_results"]["test"]["f1_macro"], distill["final_results"]["num_parameters"]),
    ]

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for (name, size, f1, params), color in zip(models_data, colors):
        ax.scatter(size, f1, s=params/500000, alpha=0.7, color=color, edgecolors="black", linewidth=0.5)
        ax.annotate(name, (size, f1), textcoords="offset points", xytext=(10, 10), fontsize=9)

    ax.set_xlabel("Model Size (MB)", fontsize=12)
    ax.set_ylabel("Test F1 (Macro)", fontsize=12)
    ax.set_title("Model Efficiency: F1 Score vs. Size\n(bubble size proportional to parameter count)", fontsize=13)
    ax.set_ylim(0.975, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'efficiency_scatter.png'}")


def generate_comparison_json(teacher, baseline, distill, output_dir):
    """Generate structured comparison report."""
    report = {
        "teacher": {
            "accuracy": teacher["final_results"]["test"]["accuracy"],
            "f1_macro": teacher["final_results"]["test"]["f1_macro"],
            "parameters": teacher["final_results"]["num_parameters"],
            "size_mb": teacher["final_results"]["model_size_mb"],
            "training_time_min": teacher["total_training_time_minutes"],
        },
        "student_baseline": {
            "accuracy": baseline["final_results"]["test"]["accuracy"],
            "f1_macro": baseline["final_results"]["test"]["f1_macro"],
            "parameters": baseline["final_results"]["num_parameters"],
            "size_mb": baseline["final_results"]["model_size_mb"],
            "training_time_min": baseline["total_training_time_minutes"],
        },
        "student_distilled": {
            "accuracy": distill["final_results"]["test"]["accuracy"],
            "f1_macro": distill["final_results"]["test"]["f1_macro"],
            "parameters": distill["final_results"]["num_parameters"],
            "size_mb": distill["final_results"]["model_size_mb"],
            "training_time_min": distill["total_training_time_minutes"],
            "temperature": distill["config"]["temperature"],
            "alpha": distill["config"]["alpha"],
        },
        "compression": {
            "parameter_ratio": teacher["final_results"]["num_parameters"] / distill["final_results"]["num_parameters"],
            "size_reduction_pct": (1 - distill["final_results"]["model_size_mb"] / teacher["final_results"]["model_size_mb"]) * 100,
            "accuracy_retention_pct": distill["final_results"]["test"]["accuracy"] / teacher["final_results"]["test"]["accuracy"] * 100,
            "f1_retention_pct": distill["final_results"]["test"]["f1_macro"] / teacher["final_results"]["test"]["f1_macro"] * 100,
            "distillation_gain_over_baseline": distill["final_results"]["test"]["accuracy"] - baseline["final_results"]["test"]["accuracy"],
        },
    }

    with open(output_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {output_dir / 'comparison_report.json'}")


def main():
    """Generate all reports."""
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    print("Loading experiment results...")
    teacher, baseline, distill = load_results()

    print("\nGenerating visualizations:")
    plot_model_comparison(teacher, baseline, distill, output_dir)
    plot_training_curves(teacher, baseline, distill, output_dir)
    plot_distillation_losses(distill, output_dir)
    plot_efficiency_scatter(teacher, baseline, distill, output_dir)
    generate_comparison_json(teacher, baseline, distill, output_dir)

    print("\nAll reports generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
