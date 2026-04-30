from utils.db import create_engine_and_session_factory, sqlite_add_missing_columns
from src.logger import CustomLogger, update_global_log_level
from src.settings import SETTINGS
from src.dataset import Dataset
from src.arguments import args
from models.base import Base
from src.model import Model

import asyncio
import yaml

logger = CustomLogger("benchmark").get_logger()


async def init_db():
    logger.info(f"Initializing database at {SETTINGS.DB_FILE}")
    engine, session_factory = create_engine_and_session_factory(
        "sqlite+aiosqlite:///" + SETTINGS.DB_FILE
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(sqlite_add_missing_columns)
    await engine.dispose()


if __name__ == "__main__":
    #  Check arguments
    if not args.config.exists():
        raise ValueError(f"Config file {args.config} does not exist.")

    if not args.model_name:
        raise ValueError("Please provide a mode name.")

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    asyncio.run(init_db())

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

    logger.info("Starting validaton.")
    model = Model(
        runs_directory=SETTINGS.RUNS_DIRECTORY,
        device=SETTINGS.AI_DEVICE,
        model_directory=SETTINGS.MODEL_DIRECTORY,
        mode="benchmark",
        model_name=args.model_name,
        **config["benchmark"],
    )

    formats = ["", "onnx", "openvino"]

    selected_device = model.get_selected_device()
    print(f"Selected device: {selected_device}")

    for yolo_format in formats:
        print(f"Validating model in {yolo_format} format.")
        model.validate(
            model_format=yolo_format,
            dataset_config_path=dataset.get_config_path()
        )
        metrics = model.get_val_metrics()
        number_of_images = metrics.nt_per_image[0]
        speed = metrics.speed["inference"]
        fps = round(1000 / (speed + 1e-3), 2)

        print(f"Number of images: {number_of_images}")
        print(f"Speed: {speed} ms")
        print(f"Inference time per image: {speed / number_of_images} ms")
        print(f"FPS: {fps}")


    logger.info("Model config: %s", model.get_config())
