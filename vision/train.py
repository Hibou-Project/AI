from src.settings import SETTINGS
from src.dataset import Dataset
from src.arguments import args

import yaml

if __name__ == "__main__":
    #  Check arguments
    if not args.config.exists():
        raise ValueError(f"Config file {args.config} does not exist.")

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset = Dataset(hf_url=config["dataset"]["hf_name"], save_dir=SETTINGS.HF_DATASET_PATH)
    dataset.split(
        seed=config["reproducibility"]["seed"],
        base_split="train_validation_test",
        label_column="class_id"
    )
    dataset.export_to_yolo(
        image_transform=config["global"]["image_transform"],
        load_label_other=config["labels"]["load_other"]
    )

    dataset.save_dataset_settings(
        image_transform=config["global"]["image_transform"],
        load_label_other=config["labels"]["load_other"]
    )