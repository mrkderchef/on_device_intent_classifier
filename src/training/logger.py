"""
Training logger for experiment tracking.

Logs metrics to:
- JSON file (structured, easy to reload in notebooks)
- CSV file (epoch-by-epoch, easy to plot)
- Console (real-time monitoring)
"""

import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Optional


class TrainingLogger:
    """Unified logger for training experiments."""

    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # File paths
        self.metrics_csv_path = self.output_dir / "metrics.csv"
        self.config_path = self.output_dir / "config.json"
        self.results_path = self.output_dir / "results.json"
        self.log_path = self.output_dir / "training.log"

        # State
        self.epoch_metrics = []
        self.start_time = None
        self.config = {}

        # Initialize CSV
        self._csv_initialized = False

    def log_config(self, config: dict) -> None:
        """Log experiment configuration."""
        self.config = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            **config,
        }
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        self._log(f"Config saved to {self.config_path}")

    def start_training(self) -> None:
        """Mark training start."""
        self.start_time = time.time()
        self._log(f"Training started: {self.experiment_name}")

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log metrics for one epoch."""
        metrics["epoch"] = epoch
        metrics["elapsed_seconds"] = time.time() - self.start_time if self.start_time else 0
        self.epoch_metrics.append(metrics)

        # Write to CSV
        self._write_csv_row(metrics)

        # Console output
        parts = [f"Epoch {epoch:3d}"]
        for k, v in metrics.items():
            if k in ("epoch", "elapsed_seconds"):
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        self._log(" | ".join(parts))

    def log_final_results(self, results: dict) -> None:
        """Log final evaluation results."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        final = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_training_time_seconds": elapsed,
            "total_training_time_minutes": elapsed / 60,
            "epoch_history": self.epoch_metrics,
            "final_results": results,
            "config": self.config,
        }
        with open(self.results_path, "w") as f:
            json.dump(final, f, indent=2)
        self._log(f"Final results saved to {self.results_path}")
        self._log(f"Total training time: {elapsed/60:.1f} minutes")

    def _write_csv_row(self, metrics: dict) -> None:
        """Append one row to the CSV metrics file."""
        if not self._csv_initialized:
            with open(self.metrics_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                writer.writeheader()
                writer.writerow(metrics)
            self._csv_initialized = True
        else:
            with open(self.metrics_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                writer.writerow(metrics)

    def _log(self, message: str) -> None:
        """Write to console and log file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        with open(self.log_path, "a") as f:
            f.write(formatted + "\n")
