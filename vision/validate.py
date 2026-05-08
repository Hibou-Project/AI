from src.logger import CustomLogger
from src.settings import SETTINGS
from src.dataset import Dataset
from src.arguments import args
from src.model import Model

import yaml

logger = CustomLogger("validation").get_logger()

if __name__ == "__main__":
    #  Check arguments
    if not args.config.exists():
        raise ValueError(f"Config file {args.config} does not exist.")

    if not args.model_name:
        raise ValueError("Please provide a mode name.")

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset = Dataset(
        hf_url=config["dataset"]["hf_name"],
        save_dir=SETTINGS.DATASET_PATH,
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

    logger.info("Starting validaton.")
    model = Model(
        runs_directory=SETTINGS.RUNS_DIRECTORY,
        device=SETTINGS.AI_DEVICE,
        model_directory=SETTINGS.MODEL_DIRECTORY,
        mode="validate",
        model_name=args.model_name,
        uploads_metrics=SETTINGS.LOG_WANDB_ENABLE,
        **config["validation"],
    )

    logger.info("Model config: %s", model.get_config())
    model.validate(
        dataset_config_path=dataset.get_config_path()
    )

    if not args.quiet:
        metrics = model.get_val_metrics()
        f1_curve = metrics.box.f1_curve[0]
        conf_curve = metrics.box.px

        best_idx = f1_curve.argmax()
        best_f1 = f1_curve[best_idx]
        best_conf = conf_curve[best_idx]

        score = metrics.box.map

        print(f"mAP50-95: {score:.2f}")
        print(f"Best F1: {best_f1:.2f}")
        print(f"Confidence threshold: {best_conf:.2f}")

        model.show_validation_results()
