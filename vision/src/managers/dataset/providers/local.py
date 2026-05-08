from src.managers.dataset.providers.base import BaseProvider
from pathlib import Path


class LocalProvider(BaseProvider):
    def __init__(self, local_path: Path, sampling_ratio: float = 1.0):
        self._dataset_taget_path = local_path
        self.name = local_path.name
        self._sampling_ratio = sampling_ratio

    def download(self):
        pass

    def get_dataset_dir(self):
        return self._dataset_taget_path

    def get_dataset_sampling_ratio(self):
        return self._sampling_ratio
