"""Data loading, cleaning and window generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import (
    ANOMALOUS_S007_LABEL,
    CYCLING_LABELS,
    MERGED_STAIRS_LABEL,
    SENSOR_COLUMNS,
    STAIRS_LABELS,
)
from .features import calculate_enmo


def read_csv_files(paths: Sequence[Path]) -> pd.DataFrame:
    """Read and concatenate multiple CSV files."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise FileNotFoundError("No CSV files found. Please check the dataset path.")

    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def get_train_paths(data_dir: str | Path) -> list[Path]:
    """Return training CSV paths.

    If S007_cleaned.csv exists, the original S007.csv is excluded and the cleaned
    version is used instead.
    """
    data_dir = Path(data_dir)
    train_paths = sorted(data_dir.glob("*.csv"))
    cleaned_s007 = data_dir / "S007_cleaned.csv"

    if cleaned_s007.exists():
        train_paths = [p for p in train_paths if p.name != "S007.csv"]
    return train_paths


def get_test_paths(data_dir: str | Path) -> list[Path]:
    """Return test CSV paths."""
    return sorted((Path(data_dir) / "test-set").glob("*.csv"))


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that required columns exist."""
    required = ["label", *SENSOR_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_activity_name(df: pd.DataFrame, label_map: dict[int, str]) -> pd.DataFrame:
    """Add a readable activity_name column."""
    out = df.copy()
    out["activity_name"] = out["label"].map(label_map)
    return out


def drop_cycling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all cycling-related labels."""
    validate_columns(df)
    original_size = len(df)
    out = df[~df["label"].isin(CYCLING_LABELS)].copy()
    print(f"Original dataset size: {original_size} rows")
    print(f"Rows dropped: {original_size - len(out)}")
    print(f"New dataset size: {len(out)} rows")
    return out


def merge_stairs_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merge ascending and descending stairs into one unified label."""
    validate_columns(df)
    out = df.copy()
    old_count = out[out["label"].isin(STAIRS_LABELS)].shape[0]
    out.loc[out["label"].isin(STAIRS_LABELS), "label"] = MERGED_STAIRS_LABEL
    new_count = out[out["label"] == MERGED_STAIRS_LABEL].shape[0]
    print(f"Merged stairs samples: {old_count} -> {new_count}")
    return out


def clean_s007_data(data_dir: str | Path) -> Path:
    """Clean S007.csv and save S007_cleaned.csv.

    Cleaning strategy:
    1. Remove anomalous label 10.
    2. Remove frozen sensor segments using rolling standard deviation.
    3. Remove extreme outliers where absolute acceleration exceeds 16g.
    """
    data_dir = Path(data_dir)
    input_file = data_dir / "S007.csv"
    output_file = data_dir / "S007_cleaned.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Cannot find {input_file}")

    df = pd.read_csv(input_file)
    validate_columns(df)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    original_count = len(df)
    print("--- Starting S007 Dataset Cleaning ---")
    print(f"Original number of rows: {original_count}")

    df_clean = df[df["label"] != ANOMALOUS_S007_LABEL].copy()
    rows_after_label = len(df_clean)
    print(f"1. Removed label {ANOMALOUS_S007_LABEL}: {original_count - rows_after_label} rows")

    rolling_std = df_clean["back_x"].rolling(window=50).std().fillna(1.0)
    df_clean = df_clean[rolling_std >= 1e-6].copy()
    rows_after_freeze = len(df_clean)
    print(f"2. Removed frozen segments: {rows_after_label - rows_after_freeze} rows")

    outlier_mask = (df_clean[SENSOR_COLUMNS].abs() > 16).any(axis=1)
    df_clean = df_clean[~outlier_mask].copy()
    rows_after_outlier = len(df_clean)
    print(f"3. Removed extreme outliers: {rows_after_freeze - rows_after_outlier} rows")

    total_removed = original_count - len(df_clean)
    print(f"Final number of rows: {len(df_clean)}")
    print(f"Total rows removed: {total_removed} ({total_removed / original_count * 100:.2f}%)")

    df_clean.to_csv(output_file, index=False)
    print(f"Saved cleaned data to: {output_file}")
    return output_file


def load_raw_dataset_for_eda(data_dir: str | Path) -> pd.DataFrame:
    """Load all available train and test CSV files for exploratory analysis."""
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*.csv")) + sorted((data_dir / "test-set").glob("*.csv"))
    paths = [p for p in paths if p.name != "S007_cleaned.csv"]
    return read_csv_files(paths)


def prepare_train_test_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean and label-process train/test dataframes."""
    train_paths = get_train_paths(data_dir)
    test_paths = get_test_paths(data_dir)

    print(f"Training files: {len(train_paths)}")
    print(f"Test files: {len(test_paths)}")

    train_df = read_csv_files(train_paths)
    test_df = read_csv_files(test_paths)

    train_df = merge_stairs_data(drop_cycling_data(train_df))
    test_df = merge_stairs_data(drop_cycling_data(test_df))

    print(f"Training dataframe shape: {train_df.shape}")
    print(f"Test dataframe shape: {test_df.shape}")
    return train_df, test_df


def create_sliding_windows(
    df: pd.DataFrame,
    window_seconds: float = 2,
    overlap_seconds: float = 1,
    sampling_rate: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert continuous sensor data into sliding windows.

    Returns:
        X: shape (num_windows, time_steps, 6)
        y: shape (num_windows,)
    """
    validate_columns(df)
    window_size = int(window_seconds * sampling_rate)
    step_size = int((window_seconds - overlap_seconds) * sampling_rate)
    if step_size <= 0:
        raise ValueError("overlap_seconds must be smaller than window_seconds")

    data_values = df[SENSOR_COLUMNS].to_numpy()
    label_values = df["label"].to_numpy()

    X, y = [], []
    for start in range(0, len(df) - window_size + 1, step_size):
        window_data = data_values[start : start + window_size]
        window_labels = label_values[start : start + window_size]
        vals, counts = np.unique(window_labels, return_counts=True)
        mode_label = vals[np.argmax(counts)]
        X.append(window_data)
        y.append(mode_label)

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    print(f"Generated windows: X={X_arr.shape}, y={y_arr.shape}")
    return X_arr, y_arr


def create_cnn_windows(
    df: pd.DataFrame,
    window_seconds: float = 2,
    overlap_seconds: float = 1,
    sampling_rate: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Create CNN windows with 6 raw axes plus 2 ENMO channels.

    Returns:
        X: shape (num_windows, time_steps, 8)
        y: shape (num_windows,)
    """
    validate_columns(df)
    window_size = int(window_seconds * sampling_rate)
    step_size = int((window_seconds - overlap_seconds) * sampling_rate)
    if step_size <= 0:
        raise ValueError("overlap_seconds must be smaller than window_seconds")

    data_values = df[SENSOR_COLUMNS].to_numpy()
    label_values = df["label"].to_numpy()

    X, y = [], []
    for start in range(0, len(df) - window_size + 1, step_size):
        raw_window = data_values[start : start + window_size]
        enmo_back, enmo_thigh = calculate_enmo(raw_window)
        augmented_window = np.hstack([
            raw_window,
            enmo_back.reshape(-1, 1),
            enmo_thigh.reshape(-1, 1),
        ])
        labels = label_values[start : start + window_size]
        vals, counts = np.unique(labels, return_counts=True)
        X.append(augmented_window)
        y.append(vals[np.argmax(counts)])

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    print(f"Generated CNN windows: X={X_arr.shape}, y={y_arr.shape}")
    return X_arr, y_arr
