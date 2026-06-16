"""Project-level constants."""

from pathlib import Path

DEFAULT_DATA_DIR = Path("MLT-CW-Dataset")
DEFAULT_RESULTS_DIR = Path("results")

SENSOR_COLUMNS = [
    "back_x", "back_y", "back_z",
    "thigh_x", "thigh_y", "thigh_z",
]

CYCLING_LABELS = [13, 14, 130, 140]
STAIRS_LABELS = [4, 5]
MERGED_STAIRS_LABEL = 9
ANOMALOUS_S007_LABEL = 10

LABEL_MAP_FULL = {
    1: "Walking",
    2: "Running",
    3: "Shuffling",
    4: "Stairs (ascending)",
    5: "Stairs (descending)",
    6: "Standing",
    7: "Sitting",
    8: "Lying",
    9: "Stairs",
    13: "Cycling (sit)",
    14: "Cycling (stand)",
    130: "Cycling (sit, inactive)",
    140: "Cycling (stand, inactive)",
}

LABEL_MAP_FINAL = {
    1: "Walking",
    2: "Running",
    3: "Shuffling",
    6: "Standing",
    7: "Sitting",
    8: "Lying",
    9: "Stairs",
}
