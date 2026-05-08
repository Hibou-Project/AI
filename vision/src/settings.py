from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

import logging
import shutil
import os

current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Paths
source_file = os.path.join(project_root, ".env.example")
target_file = os.path.join(project_root, ".env")

# Copy .env if it does not exist
if not os.path.exists(target_file) and os.path.exists(source_file):
    shutil.copy2(source_file, target_file)
    logging.info(f"Copied {source_file} → {target_file}")

if not load_dotenv():
    raise FileNotFoundError("Failed to load .env file.")


@dataclass
class Settings:
    # TOKENS
    TOKEN_HF: str
    TOKEN_WANDB: str
    TOKEN_AWS: str

    # DATASET
    DATASET_PATH: str

    # MODELS
    MODEL_DIRECTORY: Path

    #RUNS
    RUNS_DIRECTORY: Path

    # AI
    AI_DEVICE: str

    # LOGS
    LOG_LEVEL: str
    LOG_WANDB_ENABLE: bool

    # DB
    DB_FILE: str


def parse_list(value: str):
    """Split a comma-separated string and strip whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_bool(value: str) -> bool:
    """Parse a boolean from string (True/False, yes/no)."""
    return str(value).strip().lower() in ("true", "1", "yes")


try:
    SETTINGS = Settings(
        TOKEN_HF=os.getenv("TOKEN_HF"),
        TOKEN_WANDB=os.getenv("TOKEN_WANDB"),
        TOKEN_AWS=os.getenv("TOKEN_AWS"),
        DATASET_PATH=os.getenv("DATASET_PATH"),
        MODEL_DIRECTORY=Path(os.getenv("MODEL_DIRECTORY")),
        RUNS_DIRECTORY=Path(os.getenv("RUNS_DIRECTORY")),
        AI_DEVICE=os.getenv("AI_DEVICE"),
        LOG_LEVEL=os.getenv("LOG_LEVEL"),
        LOG_WANDB_ENABLE=parse_bool(os.getenv("LOG_WANDB_ENABLE", "False")),
        DB_FILE=os.getenv("DB_FILE"),
    )
except TypeError as e:
    raise ValueError(f"Invalid value in .env: {e}. Please check the .env file.")
