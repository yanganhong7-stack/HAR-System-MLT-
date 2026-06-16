"""Command-line interface for running HAR experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import clean_s007_data, create_cnn_windows, create_sliding_windows, prepare_train_test_data
from .models.kmeans_model import run_kmeans_evaluation
from .models.random_forest import run_rf_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HAR MLT experiments.")
    parser.add_argument("--data-dir", default="MLT-CW-Dataset", help="Path to dataset directory.")
    parser.add_argument("--model", choices=["kmeans", "rf", "cnn", "all"], default="rf")
    parser.add_argument("--mode", choices=["Baseline", "Fine-Tuned", "both"], default="Fine-Tuned")
    parser.add_argument("--clean-s007", action="store_true", help="Clean S007.csv before loading data.")
    parser.add_argument("--results-dir", default="results", help="Directory for saving figures/tables.")
    parser.add_argument("--show-plots", action="store_true", help="Show plots instead of saving confusion matrices.")
    return parser.parse_args()


def _modes(mode: str) -> list[str]:
    return ["Baseline", "Fine-Tuned"] if mode == "both" else [mode]


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)

    if args.clean_s007:
        clean_s007_data(data_dir)

    train_df, test_df = prepare_train_test_data(data_dir)

    if args.model in {"kmeans", "rf", "all"}:
        X_train, y_train = create_sliding_windows(train_df)
        X_test, y_test = create_sliding_windows(test_df)

        for mode in _modes(args.mode):
            if args.model in {"kmeans", "all"}:
                fig_path = None if args.show_plots else results_dir / "figures" / f"kmeans_{mode.lower().replace('-', '_')}_cm.png"
                run_kmeans_evaluation(X_train, y_train, X_test, y_test, mode=mode, save_figure_path=fig_path)

            if args.model in {"rf", "all"}:
                fig_path = None if args.show_plots else results_dir / "figures" / f"rf_{mode.lower().replace('-', '_')}_cm.png"
                run_rf_evaluation(X_train, y_train, X_test, y_test, mode=mode, save_figure_path=fig_path)

    if args.model in {"cnn", "all"}:
        from .models.cnn import run_cnn_evaluation

        X_train_cnn, y_train_cnn = create_cnn_windows(train_df)
        X_test_cnn, y_test_cnn = create_cnn_windows(test_df)

        for mode in _modes(args.mode):
            fig_path = None if args.show_plots else results_dir / "figures" / f"cnn_{mode.lower().replace('-', '_')}_cm.png"
            run_cnn_evaluation(X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn, mode=mode, save_figure_path=fig_path)


if __name__ == "__main__":
    main()
