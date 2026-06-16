"""Clean and inspect the dataset before model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from har_mlt.data import clean_s007_data, load_raw_dataset_for_eda
from har_mlt.visualization import plot_class_balance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="MLT-CW-Dataset")
    parser.add_argument("--clean-s007", action="store_true")
    parser.add_argument("--save-class-balance", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.clean_s007:
        clean_s007_data(data_dir)

    if args.save_class_balance:
        df = load_raw_dataset_for_eda(data_dir)
        report = plot_class_balance(df, save_path="results/figures/class_balance.png")
        Path("results/tables").mkdir(parents=True, exist_ok=True)
        report.to_csv("results/tables/class_balance.csv")
        print(report)


if __name__ == "__main__":
    main()
