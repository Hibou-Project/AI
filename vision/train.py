from src.settings import SETTINGS
from logger import CustomLogger
from src.dataset import Dataset
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
        hf_url=config["dataset"]["hf_name"],
        save_dir=SETTINGS.HF_DATASET_PATH,
        image_transform=config["dataset"]["image_transform"],
        load_label_other=config["labels"]["load_other"]
    )
    logger.info("Splitting dataset into train, validation and test sets.")
    dataset.split(
        seed=config["reproducibility"]["seed"],
        base_split="train_validation_test",
        label_column="class_id",
    )
    logger.info("Exporting dataset to YOLO format.")
    dataset.export_to_yolo()

    logger.info("Saving dataset settings.")
    dataset.save_dataset_settings()

    logger.info("Starting training.")
    model = Model(
        runs_directory=SETTINGS.RUNS_DIRECTORY,
        selected_size=config["model"]["model_size"],
        selected_version=config["model"]["yolo_version"],
        model_directory=SETTINGS.MODEL_DIRECTORY,
        device=SETTINGS.AI_DEVICE,
        **config["train"],
        **config["augmentation"]
    )

    model.train(
        dataset_config_path=dataset.get_config_path()
    )
