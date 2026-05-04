from ultralytics.utils.torch_utils import get_cpu_info, get_gpu_info
from src.utils.image import plot_image_grid
from src.utils.common import get_device
from ultralytics import YOLO, settings
from settings import SETTINGS
from pathlib import Path
from PIL import Image
import ultralytics

import torch
import wandb
import uuid
import os


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

    DEFAULT_BENCHMARK_CONFIG = {
        "imgsz": 640,
        "half": False,
        "int8": False,
        "batch": 1,
        "split": "test",
        "save_json": True
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
            model_name: str = None,
            uploads_metrics: bool = False,
            **kwargs):

        self._mode = mode
        self._val_metrics = None
        self._benchmark_metrics = None
        self._uploads_metrics = uploads_metrics

        print(ultralytics.checks())

        if mode == "train":
            self.config = self.DEFAULT_TRAIN_CONFIG.copy()

            if selected_size not in self.YOLO_MODEL_SIZE:
                raise ValueError(f"Invalid size '{selected_size}'")

            model_name = f"yolo{selected_version}{self.YOLO_MODEL_SIZE[selected_size]}.pt"
            model_path = model_directory / model_name

            if not model_path.exists():
                raise FileNotFoundError(model_path)

            self.run_name = "train_" + self._create_run_name(selected_version, selected_size)

        elif mode == "validate":
            self.config = self.DEFAULT_VAL_CONFIG.copy()
            model_path = model_directory / model_name
            self.run_name = "val_yolo" + model_name.split(".")[0]

        elif mode == "benchmark":
            self.config = self.DEFAULT_BENCHMARK_CONFIG.copy()
            model_path = model_directory / model_name
            self.run_name = "benchmark_yolo" + model_name.split(".")[0]
            # self._model_extracted_name = self._extract_name(model_name)

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

        if self._uploads_metrics:
            settings.update({"wandb": True})
            wandb.login(key=SETTINGS.WANDB_API_KEY)
            wandb.init(
                project="yolo-drone-detection",
                reinit=True,
                name=self.run_name,
                resume="allow"
            )

    def train(self, dataset_config_path: Path):
        if self._mode != "train":
            raise ValueError("Model is not in train mode")
        self._model.train(
            **self.config,
            project=self._runs_directory,
            data=dataset_config_path,
            name=f"{self.run_name}"
        )

    def load_model(self, model_path: Path):
        self._model = YOLO(model_path, task="detect")

    def validate(self, dataset_config_path: Path, model_format=""):
        if self._mode == "validate" or self._mode == "benchmark":
            model_format_str = "pytorch" if model_format == "" else model_format
            full_run_name = self.run_name + f"_{model_format_str}"
            if Path(self._runs_directory / full_run_name).exists():
                raise FileExistsError(f"Run {full_run_name} already exists, please delete it first.")
            self._val_metrics = self._model.val(
                **self.config,
                project=self._runs_directory,
                data=dataset_config_path,
                format=model_format,
                name=full_run_name
            )

            if self._uploads_metrics:
                f1_curve = self._val_metrics.box.f1_curve[0]
                conf_curve = self._val_metrics.box.px
                best_idx = f1_curve.argmax()
                wandb.log({
                    "model_name": self.run_name,
                    "map50_95": self._val_metrics.box.map,
                    "map50": self._val_metrics.box.map50,
                    "precision": self._val_metrics.box.mp,
                    "recall": self._val_metrics.box.mr,
                    "best_f1": f1_curve[best_idx],
                    "best_conf": conf_curve[best_idx],
                })
                wandb.finish()
        else:
            raise ValueError("Model is not in validate or benchmark mode")

    def get_val_metrics(self):
        return self._val_metrics

    def get_config(self):
        return self.config

    def show_results(self):
        # Ensure the mode is correct
        if self._mode != self._mode:
            raise ValueError(f"Model is not in {self._mode} mode")

        if self._mode == "benchmark":
            return

        # Set the result directory based on the mode
        if self._mode == "train":
            result_dir = self._runs_directory / self.run_name
        else:
            result_dir = self._runs_directory / str(self.run_name + "_pytorch")

        if self._mode == "train":
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
        self.show_results()

    def show_validation_results(self):
        self.show_results()

    def _create_run_name(self, version, size):
        session_id = uuid.uuid4().hex[:6]
        return f"yolo{version}-{size}-{session_id}"

    def get_model_size(self):
        return self._model.model.yaml.get("scale")

    def get_model_version(self):
        return self._model.model.yaml.get("format")

    def get_selected_device(self):

        if self.config["device"] == "cpu":
            is_cuda = False
        elif self.config["device"] == "cuda":
            is_cuda = True
        elif isinstance(self.config["device"], list):
            is_cuda = True
        else:
            raise ValueError(f"Invalid device: {self.config['device']}")

        info_dict = {
            "cpu": get_cpu_info(),
            "cpu_count": os.cpu_count(),
            "gpu": get_gpu_info(index=0) if is_cuda else None,
            "gpu_count": torch.cuda.device_count() if is_cuda else None,
        }
        hardware_type = "GPU" if is_cuda else "CPU"
        name = info_dict["gpu"] if is_cuda else info_dict["cpu"]
        return hardware_type, name
