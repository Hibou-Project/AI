from src.managers.dataset.providers.base import BaseProvider
from src.logger import CustomLogger
from huggingface_hub import login
from huggingface_hub import HfApi
from datasets import load_dataset
from src.settings import SETTINGS
from pathlib import Path
from tqdm import tqdm

logger = CustomLogger("Hugging Face").get_logger()


class HuggingFaceProviders(BaseProvider):
    def __init__(self, name, hf_revision="main", sampling_ratio: float = 1.0):
        self._dataset_taget_path = None
        self.hf_revision = hf_revision
        self._dataset = None
        self.name = name
        self._sampling_ratio = sampling_ratio
        login(SETTINGS.TOKEN_HF)

        self._dataset_taget_path = Path(SETTINGS.DATASET_PATH) / ".downloads" / str("hf_" + self.name.replace("/", "_"))

    def download(self):
        self._dataset = load_dataset(self.name, revision=self.hf_revision)
        logger.info(f"Dataset downloaded to {self._dataset_taget_path}, starting export...")
        self._save()

    def get_dataset_sampling_ratio(self):
        return self._sampling_ratio

    def get_dataset_dir(self):
        return self._dataset_taget_path

    def _get_dataset_sha(self):
        api = HfApi()
        info = api.dataset_info(self.name)
        return info.sha

    def _get_export_marker_path(self):
        return Path(self._dataset_taget_path.parent, f".{self._dataset_taget_path.name.split('.')[0]}")

    def _write_marker(self):
        marker_path = self._get_export_marker_path()
        current_sha = self._get_dataset_sha()

        with open(marker_path, "w") as f:
            f.write(current_sha)

    def _has_been_downloaded(self) -> bool:
        current_sha = self._get_dataset_sha()
        marker_path = self._get_export_marker_path()

        if marker_path.exists():
            with open(marker_path, "r") as f:
                saved_sha = f.read().strip()
            if saved_sha == current_sha and not self._dataset_taget_path.exists():
                logger.warning("Dataset already exported, but files do not exist. Re-exporting.")
                return True
            elif saved_sha == current_sha:
                logger.info("Dataset already exported. Skipping")
                return True
        return False

    def _save(self):
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded.")

        # Check if the dataset has already been downloaded
        if self._has_been_downloaded():
            return

        self._dataset_taget_path.mkdir(parents=True, exist_ok=True)

        def export(ds):
            for idx, sample in enumerate(tqdm(ds, total=len(ds))):
                image = sample["image"]  # PIL.Image
                label = sample["raw_label"]  # YOLO format [[class, cx, cy, w, h], ...]
                img_name = sample["name"]
                txt_name = img_name.split(".")[0] + ".txt"

                image.save(self._dataset_taget_path / img_name, quality=100)

                # Save YOLO labels
                lbl_path = self._dataset_taget_path / txt_name
                with open(lbl_path, "w") as file:
                    file.write(label)

        export(self._dataset["train_validation_test"])
        self._write_marker()
