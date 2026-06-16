"""Feature extraction utilities."""

from __future__ import annotations

import numpy as np


def calculate_enmo(window_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate ENMO for back and thigh accelerometer signals.

    ENMO = max(0, ||a|| - 1)
    """
    back_acc = window_data[:, 0:3]
    thigh_acc = window_data[:, 3:6]

    norm_back = np.linalg.norm(back_acc, axis=1)
    norm_thigh = np.linalg.norm(thigh_acc, axis=1)

    enmo_back = np.maximum(0, norm_back - 1)
    enmo_thigh = np.maximum(0, norm_thigh - 1)
    return enmo_back, enmo_thigh


def extract_features_baseline(X_windows: np.ndarray) -> np.ndarray:
    """Extract a 10-dimensional baseline feature vector per window."""
    features = []
    for window in X_windows:
        enmo_back, enmo_thigh = calculate_enmo(window)
        features.append([
            np.mean(enmo_back), np.std(enmo_back),
            np.mean(enmo_thigh), np.std(enmo_thigh),
            np.mean(window[:, 0]), np.std(window[:, 0]),
            np.mean(window[:, 2]), np.std(window[:, 2]),
            np.mean(window[:, 3]), np.std(window[:, 3]),
        ])
    return np.asarray(features)


def extract_features_finetuned(X_windows: np.ndarray) -> np.ndarray:
    """Extract a 20-dimensional fine-tuned feature vector per window."""
    features = []
    for window in X_windows:
        enmo_back, enmo_thigh = calculate_enmo(window)
        energy_back = np.sum(enmo_back ** 2) / len(enmo_back)
        energy_thigh = np.sum(enmo_thigh ** 2) / len(enmo_thigh)

        features.append([
            np.mean(enmo_back), np.std(enmo_back), np.max(enmo_back), energy_back,
            np.mean(enmo_thigh), np.std(enmo_thigh), np.max(enmo_thigh), energy_thigh,
            np.mean(window[:, 0]), np.std(window[:, 0]),
            np.mean(window[:, 1]), np.std(window[:, 1]),
            np.mean(window[:, 2]), np.std(window[:, 2]),
            np.mean(window[:, 3]), np.std(window[:, 3]),
            np.mean(window[:, 4]), np.std(window[:, 4]),
            np.mean(window[:, 5]), np.std(window[:, 5]),
        ])
    return np.asarray(features)


def get_feature_matrix(X_windows: np.ndarray, mode: str = "Baseline") -> np.ndarray:
    """Select baseline or fine-tuned feature extraction."""
    if mode == "Baseline":
        return extract_features_baseline(X_windows)
    if mode == "Fine-Tuned":
        return extract_features_finetuned(X_windows)
    raise ValueError("mode must be either 'Baseline' or 'Fine-Tuned'")
