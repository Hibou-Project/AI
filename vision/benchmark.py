from src.models import Model as DbModel
from src.logger import CustomLogger
from src.settings import SETTINGS
from src.models import Benchmark
from src.dataset import Dataset
from src.models import Hardware
from src.arguments import args
from src.models import Format
from sqlalchemy import select
from utils.db import Database
from src.model import Model

import asyncio
import yaml

logger = CustomLogger("benchmark").get_logger()


async def main():
    # Initialize DB
    await Database.init_db()

    is_half = config["benchmark"]["half"]
    is_int8 = config["benchmark"]["int8"]
    if is_half and is_int8:
        raise ValueError(
            "Cannot use half and int8 precision at the same time. Edit the config file and set one of them to False.")

    # Load dataset
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
    dataset.save_dataset_settings()

    # Load model
    model = Model(
        runs_directory=SETTINGS.RUNS_DIRECTORY,
        device=SETTINGS.AI_DEVICE,
        model_directory=SETTINGS.MODEL_DIRECTORY,
        mode="benchmark",
        model_name=args.model_name,
        **config["benchmark"],
    )

    run_results = []

    # Benchmark per format
    formats = ["", "onnx", "openvino"]
    for yolo_format in formats:
        print(f"Validating model in {yolo_format} format.")
        model.validate(
            model_format=yolo_format,
            dataset_config_path=dataset.get_config_path()
        )
        metrics = model.get_val_metrics()

        mAP50_95 = metrics.box.map
        number_of_images = metrics.nt_per_image[0]
        speed = metrics.speed["inference"]
        fps = round(1000 / (speed + 1e-3), 2)

        model_config = model.get_config()

        print(f"Number of images: {number_of_images}")
        print(f"Speed: {speed} ms")
        print(f"Inference time per image: {speed / number_of_images} ms")
        print(f"FPS: {fps}")

        if yolo_format == "":
            yolo_format = "pytorch"

        results = {
            "format": yolo_format,
            "half": model_config["half"],
            "int8": model_config["int8"],
            "number_of_images": number_of_images,
            "speed": speed,
            "map50_95": mAP50_95,
            "fps": fps,
            "inference_time_per_image": speed / number_of_images,
        }

        run_results.append(results)

    db_session = Database.get_session_factory()
    async with db_session() as session:
        # Add hardware
        hardware_type, selected_device = model.get_selected_device()
        print(f"Selected device: {selected_device}")

        hardware_result = await session.execute(select(Hardware).where(Hardware.name == selected_device))
        hardware = hardware_result.scalars().first()
        if hardware:
            print(f"Hardware {selected_device} already exists in the database.")
        else:
            hardware = Hardware(
                name=selected_device,
                type=hardware_type
            )
            session.add(hardware)

        # Add model
        db_model = await Database.get_or_create(
            session,
            DbModel,
            name=args.model_name,
            size=model.get_model_size(),
            yolo_version=int(config["model"]["yolo_version"]),
        )

        model_id: int = int(db_model.id.__str__())
        hardware_id: int = int(hardware.id.__str__())

        for result in run_results:
            # Add model_format
            precision = (
                "fp16" if result["half"]
                else "int8" if result["int8"]
                else "fp32"
            )

            model_format = await Database.get_or_create(
                session,
                Format,
                name=result["format"],
                precision=precision
            )

            format_id: int = model_format.id

            # Add the final benchmark
            benchmark = Benchmark(
                model_id=model_id,
                hardware_id=hardware_id,
                format_id=format_id,
                batch_size=config["benchmark"]["batch"],
                ms_per_image=result["inference_time_per_image"],
                throughput_fps=result["fps"],
                map50_95=result["map50_95"],
            )
            session.add(benchmark)
        await session.commit()


if __name__ == "__main__":

    # Load YAML config first
    if not args.config.exists():
        raise ValueError(f"Config file {args.config} does not exist.")

    if not args.model_name:
        raise ValueError("Please provide a model name.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Run the async main
    asyncio.run(main())
