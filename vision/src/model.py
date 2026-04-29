from src.utils.image import plot_image_grid
from src.utils.common import get_device
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

import uuid


class Model:
    DEFAULT_TRAIN_CONFIG = {
        'epochs': 200,
        'imgsz': 640,
        'batch': 16,
        'warmup_epochs': 3,
        'momentum': 0.9,
        'lr0': 0.0003,
        'lrf': 0.01,
        'patience': 40,
        'optimizer': 'adamW',
        'cache': False,
        'multi_scale': 0.25,

        # Augmentation
        'degrees': 5,
        'perspective': 0.0002,
        'fliplr': 0.3,
        'shear': 5,
        'scale': 0.3,
        'mosaic': 0.3,
        'close_mosaic': 50,
        'cutmix': 0.3,
    }

    DEFAULT_VAL_CONFIG = {
        "visualize": True,
        "save": True,
        "split": "test"
    }

    YOLO_MODEL_SIZE = {
        "nano": "n",
        "small": "s",
        "medium": "m",
        "large": "l",
        "xlarge": "x",
    }

    def __init__(
            self,
            runs_directory: Path,
            model_directory,
            selected_size=None,
            selected_version=None,
            device="auto",
            mode: str = "train",
            model_validation_name: str = None,
            **kwargs):

        self._mode = mode
        self._val_metrics = None

        if mode == "train":
            self.config = self.DEFAULT_TRAIN_CONFIG.copy()

            if selected_size not in self.YOLO_MODEL_SIZE:
                raise ValueError(f"Invalid size '{selected_size}'")

            model_name = f"yolo{selected_version}{self.YOLO_MODEL_SIZE[selected_size]}.pt"
            model_path = model_directory / model_name

            if not model_path.exists():
                raise FileNotFoundError(model_path)

            self.run_name = self._create_run_name(selected_version, selected_size)

        elif mode == "validate":
            self.config = self.DEFAULT_VAL_CONFIG.copy()
            model_path = model_directory / model_validation_name
            self.run_name = "val_yolo" + model_validation_name.split(".")[0]

        else:
            raise ValueError(f"Invalid mode: {mode}")

        self._runs_directory = runs_directory
        self._runs_directory.is_absolute()
        if not self._runs_directory.is_absolute():
            project_dir = Path(__file__).resolve().parents[2]
            self._runs_directory = Path(project_dir, self._runs_directory)

        self.config["device"] = get_device() if device == "auto" else device

        # validate + update config
        for k, v in kwargs.items():
            if k not in self.config:
                raise ValueError(f"Unknown training parameter: {k}")
            self.config[k] = v

        self._model = YOLO(model_path, task="detect")

    def train(self, dataset_config_path: Path):
        if self._mode != "train":
            raise ValueError("Model is not in train mode")
        self._model.train(
            **self.config,
            project=self._runs_directory,
            data=dataset_config_path,
            name=f"train_{self.run_name}"
        )

    def load_model(self, model_path: Path):
        self._model = YOLO(model_path, task="detect")

    def validate(self, dataset_config_path: Path):
        if self._mode != "validate":
            raise ValueError("Model is not in validate mode")
        if Path(self._runs_directory / self.run_name).exists():
            raise FileExistsError(f"Run {self.run_name} already exists, please delete it first.")
        self._val_metrics = self._model.val(
            **self.config,
            project=self._runs_directory,
            data=dataset_config_path,
            name=self.run_name)

    def get_val_metrics(self):
        return self._val_metrics

    def get_config(self):
        return self.config

    def show_results(self, mode: str):
        # Ensure the mode is correct
        if self._mode != mode:
            raise ValueError(f"Model is not in {mode} mode")

        # Set result directory based on mode
        if mode == "train":
            result_dir = self._runs_directory / f"train_{self.run_name}"
        elif mode == "validate":
            result_dir = self._runs_directory / self.run_name
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if mode == "train":
            # Show results.png (common for both)
            image_path = result_dir / "results.png"
            img = Image.open(image_path)
            img.show()

        # Process validation batches (common for both)
        val_batch = []
        i = 0
        batch_file = result_dir / f"val_batch{i}_labels.jpg"
        while batch_file.exists():
            val_batch.append(batch_file)
            val_batch.append(result_dir / f"val_batch{i}_pred.jpg")
            i += 1
            batch_file = result_dir / f"val_batch{i}_labels.jpg"

        # Retrieve confusion matrix and metric images (common for both)
        confusion_matrix_path = [
            result_dir / "confusion_matrix.png",
            result_dir / "confusion_matrix_normalized.png"
        ]

        boxes_path = [
            result_dir / "BoxF1_curve.png",
            result_dir / "BoxP_curve.png",
            result_dir / "BoxPR_curve.png",
            result_dir / "BoxR_curve.png",
        ]

        # Show images
        plot_image_grid(val_batch, nb_cols=2, show_title=True)
        plot_image_grid(confusion_matrix_path, nb_cols=2)
        plot_image_grid(boxes_path, nb_cols=2)

    def show_trained_results(self):
        self.show_results("train")

    def show_validation_results(self):
        self.show_results("validate")

    def _create_run_name(self, version, size):
        session_id = uuid.uuid4().hex[:6]
        return f"yolo{version}-{size}-{session_id}"
