"""
Dataset and DataLoader utilities for SNIPS intent classification.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class SNIPSDataset(Dataset):
    """PyTorch Dataset for SNIPS intent classification."""

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 64,
        label_map: Optional[dict] = None,
    ):
        self.df = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Build or load label mapping
        if label_map is not None:
            self.label_map = label_map
        else:
            unique_labels = sorted(self.df["label"].unique())
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        self.num_labels = len(self.label_map)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = self.label_map[row["label"]]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def get_label_map(processed_dir: str = "data/processed") -> dict:
    """Load label mapping from processed data directory."""
    label_map_path = Path(processed_dir) / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            idx_to_label = json.load(f)
        # Convert to label -> idx
        return {v: int(k) for k, v in idx_to_label.items()}

    # Fallback: build from train data
    train_path = Path(processed_dir) / "train.csv"
    if train_path.exists():
        df = pd.read_csv(train_path)
        unique_labels = sorted(df["label"].unique())
        return {label: idx for idx, label in enumerate(unique_labels)}

    raise FileNotFoundError(f"No label map or training data found in {processed_dir}")


def create_dataloaders(
    data_dir: str = "data/processed",
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 64,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    data_path = Path(data_dir)
    label_map = get_label_map(data_dir)

    dataloaders = []
    for split in ["train", "validation", "test"]:
        split_path = data_path / f"{split}.csv"
        if not split_path.exists():
            # Try alternate naming
            alt_path = data_path / f"{split.replace('validation', 'val')}.csv"
            if alt_path.exists():
                split_path = alt_path
            else:
                print(f"Warning: {split_path} not found, skipping")
                dataloaders.append(None)
                continue

        dataset = SNIPSDataset(
            data_path=str(split_path),
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            label_map=label_map,
        )

        shuffle = split == "train"
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        dataloaders.append(loader)

    return tuple(dataloaders)
