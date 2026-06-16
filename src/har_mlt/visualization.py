"""Visualization helpers for EDA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import LABEL_MAP_FULL


def plot_class_balance(df: pd.DataFrame, save_path: str | Path | None = None) -> pd.DataFrame:
    """Plot class distribution and return a class balance report."""
    df = df.copy()
    df["activity_name"] = df["label"].map(LABEL_MAP_FULL)
    counts = df["activity_name"].value_counts()
    percentages = df["activity_name"].value_counts(normalize=True) * 100
    report = pd.DataFrame({"Count": counts, "Percentage (%)": percentages})

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=counts.values, y=counts.index, hue=counts.index, legend=False)
    plt.title(f"Dataset Class Distribution (N={len(df)})")
    plt.xlabel("Number of Samples")
    plt.ylabel("Activity Type")

    for i, value in enumerate(counts.values):
        pct = percentages.values[i]
        ax.text(value, i, f" {value} ({pct:.2f}%)", va="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close()
    return report


def visualize_walking_pattern(df: pd.DataFrame, window_size: int = 300, start_index: int = 1000) -> None:
    """Visualise a sampled walking segment from back and thigh sensors."""
    walking_data = df[df["label"] == 1].copy().reset_index(drop=True)
    if walking_data.empty:
        print("No Walking samples found.")
        return

    if len(walking_data) < start_index + window_size:
        sample = walking_data
    else:
        sample = walking_data.iloc[start_index : start_index + window_size]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x_data = sample.index.to_numpy()

    axes[0].plot(x_data, sample["back_x"].to_numpy(), label="Back X")
    axes[0].plot(x_data, sample["back_y"].to_numpy(), label="Back Y")
    axes[0].plot(x_data, sample["back_z"].to_numpy(), label="Back Z")
    axes[0].set_title("Walking Pattern - Back Sensor")
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].legend(loc="upper right")

    axes[1].plot(x_data, sample["thigh_x"].to_numpy(), label="Thigh X")
    axes[1].plot(x_data, sample["thigh_y"].to_numpy(), label="Thigh Y")
    axes[1].plot(x_data, sample["thigh_z"].to_numpy(), label="Thigh Z")
    axes[1].set_title("Walking Pattern - Thigh Sensor")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Acceleration (g)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()
