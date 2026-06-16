"""Evaluation and reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

from .config import LABEL_MAP_FINAL


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return accuracy, macro recall and classification report dictionary."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def make_result_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Create a readable per-class precision/recall table."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for cls in sorted(np.unique(y_true)):
        cls_str = str(cls)
        if cls_str not in report:
            continue
        precision = report[cls_str]["precision"]
        recall = report[cls_str]["recall"]
        rows.append({
            "Label": int(cls),
            "Activity": LABEL_MAP_FINAL.get(int(cls), str(cls)),
            "Precision": precision,
            "Recall": recall,
            "Status": "PASS" if precision >= 0.75 and recall >= 0.50 else "FAIL",
        })
    return pd.DataFrame(rows)


def print_summary(name: str, train_acc: float, cv_acc: float, cv_std: float, test_acc: float, test_recall: float) -> None:
    """Print a concise model performance summary."""
    print(f"\n{name} Summary")
    print("-" * (len(name) + 8))
    print(f"Average training accuracy: {train_acc:.4f}")
    print(f"Average CV accuracy:       {cv_acc:.4f} (+/- {cv_std:.4f})")
    print(f"Final test accuracy:       {test_acc:.4f}")
    print(f"Final test macro recall:   {test_recall:.4f}")
    print(f"Training error:            {1 - train_acc:.4f}")
    print(f"Generalisation error:      {1 - test_acc:.4f}")
    print(f"Generalisation gap:        {train_acc - test_acc:.4f}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    """Plot and optionally save a confusion matrix."""
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tick_labels = [LABEL_MAP_FINAL.get(int(label), str(label)) for label in labels]

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close()
