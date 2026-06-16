"""Small smoke tests using synthetic data."""

import numpy as np
import pandas as pd

from har_mlt.data import create_cnn_windows, create_sliding_windows, drop_cycling_data, merge_stairs_data
from har_mlt.features import extract_features_baseline, extract_features_finetuned


def _fake_df(n=300):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="10ms"),
        "label": np.repeat([1, 4, 13], repeats=n // 3),
        "back_x": rng.normal(size=n),
        "back_y": rng.normal(size=n),
        "back_z": rng.normal(size=n),
        "thigh_x": rng.normal(size=n),
        "thigh_y": rng.normal(size=n),
        "thigh_z": rng.normal(size=n),
    })


def test_window_and_features():
    df = _fake_df()
    df = merge_stairs_data(drop_cycling_data(df))
    X, y = create_sliding_windows(df, window_seconds=2, overlap_seconds=1, sampling_rate=100)
    assert X.ndim == 3
    assert y.ndim == 1
    assert extract_features_baseline(X).shape[1] == 10
    assert extract_features_finetuned(X).shape[1] == 20


def test_cnn_windows():
    df = _fake_df()
    df = merge_stairs_data(drop_cycling_data(df))
    X, y = create_cnn_windows(df, window_seconds=2, overlap_seconds=1, sampling_rate=100)
    assert X.shape[-1] == 8
    assert len(X) == len(y)
