"""
Download and prepare the SNIPS dataset for intent classification.

Uses the Hugging Face datasets library to download the SNIPS dataset
and saves train/validation/test splits locally.
"""

import json
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from sklearn.model_selection import train_test_split


# SNIPS intent label mapping
INTENT_LABELS = [
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
]


def download_snips(data_dir: str = "data") -> None:
    """Download SNIPS dataset and save as CSV files."""
    data_path = Path(data_dir)
    raw_path = data_path / "raw"
    processed_path = data_path / "processed"

    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    print("Downloading SNIPS dataset from Hugging Face (benayas/snips)...")
    dataset = load_dataset("benayas/snips")
    print(f"Dataset structure: {dataset}")

    # Save raw data
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        df = split_data.to_pandas()
        raw_file = raw_path / f"{split_name}.csv"
        df.to_csv(raw_file, index=False)
        print(f"Saved raw {split_name}: {len(df)} samples -> {raw_file}")

    # Process: standardize column names (text, label)
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Rename 'category' -> 'label'
    train_df = train_df.rename(columns={"category": "label"})
    test_df = test_df.rename(columns={"category": "label"})

    # Create validation split from training data (10%)
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["label"]
    )

    # Save processed splits
    train_df.to_csv(processed_path / "train.csv", index=False)
    val_df.to_csv(processed_path / "validation.csv", index=False)
    test_df.to_csv(processed_path / "test.csv", index=False)

    print(f"\nProcessed splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Save label mapping
    unique_labels = sorted(train_df["label"].unique())
    label_map = {i: label for i, label in enumerate(unique_labels)}
    with open(processed_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nLabel mapping:")
    for idx, label in label_map.items():
        count = len(train_df[train_df["label"] == label])
        print(f"  {idx}: {label} ({count} train samples)")

    print(f"\nDataset download complete!")
    print(f"Raw data: {raw_path}")
    print(f"Processed data: {processed_path}")


if __name__ == "__main__":
    download_snips()
