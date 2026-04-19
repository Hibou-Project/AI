from vision.src.utils.common import get_device
from ultralytics import YOLO
from pathlib import Path
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

    YOLO_MODEL_SIZE = {
        "nano": "n",
        "small": "s",
        "medium": "m",
        "large": "l",
        "xlarge": "x",
    }

    def __init__(self, runs_directory: Path, selected_size, selected_version, model_directory, device="auto", **kwargs):
        self.train_config = self.DEFAULT_TRAIN_CONFIG.copy()
        self.train_config["device"] = get_device() if device == "auto" else device

        # validate + update config
        for k, v in kwargs.items():
            if k not in self.train_config:
                raise ValueError(f"Unknown training parameter: {k}")
            self.train_config[k] = v

        if selected_size not in self.YOLO_MODEL_SIZE:
            raise ValueError(f"Invalid size '{selected_size}'")

        model_name = f"yolo{selected_version}{self.YOLO_MODEL_SIZE[selected_size]}.pt"
        model_path = model_directory / model_name

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        self._model = YOLO(model_path, task="detect")
        self.run_name = self._create_run_name(selected_version, selected_size)
        self._runs_directory = runs_directory

    def train(self, dataset_config_path: Path):
        self._runs_directory.is_absolute()
        if not self._runs_directory.is_absolute():
            project_dir = Path(__file__).resolve().parents[2]
            self._runs_directory = Path(project_dir, self._runs_directory)
        self._model.train(
            **self.train_config,
            project=self._runs_directory,
            data=dataset_config_path,
            name=f"train_{self.run_name}"
        )

    def _create_run_name(self, version, size):
        session_id = uuid.uuid4().hex[:6]
        return f"yolo{version}-{size}-{session_id}"
