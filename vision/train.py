from src.managers.dataset.dataset import Dataset
from src.logger import CustomLogger
from src.settings import SETTINGS
from src.arguments import args
from src.model import Model

import yaml

logger = CustomLogger("train").get_logger()

if __name__ == "__main__":
    #  Check arguments
    if not args.config.exists():
        raise ValueError(f"Config file {args.config} does not exist.")

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset = Dataset(
        providers=config["dataset"]["providers"],
        image_transform= config["dataset"]["image_transform"],
        save_dir=SETTINGS.DATASET_PATH,
        split_ratio=config["dataset"]["split_ratio"],
        seed = config["reproducibility"]["seed"],
    )

    dataset.download()
    dataset.merge()

    logger.info("Splitting dataset into train, validation and test sets.")
    dataset.split()

    logger.info("Saving dataset settings.")
    dataset.save_dataset_settings()

    logger.info("Starting training.")
    model = Model(
        runs_directory=SETTINGS.RUNS_DIRECTORY,
        selected_size=config["model"]["model_size"],
        selected_version=config["model"]["yolo_version"],
        model_directory=SETTINGS.MODEL_DIRECTORY,
        device=SETTINGS.AI_DEVICE,
        mode="train",
        **config["train"],
    )

    logger.info("Model config: %s", model.get_config())
    model.train(
        dataset_config_path=dataset.get_config_path()
    )

    if not args.quiet:
        model.show_trained_results()
