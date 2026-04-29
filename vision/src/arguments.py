from pathlib import Path

import argparse

parser = argparse.ArgumentParser(
    prog="yolo-drone",
    description="Train and validate YOLO drone models",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--config",
    type=Path,
    required=True,
    help="Path to YAML configuration file"
)

parser.add_argument(
    "-q", "--quiet",
    action="store_true",
    help="Suppress output",
)

args = parser.parse_args()
