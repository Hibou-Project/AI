from src.managers.dataset.providers.base import BaseProvider
from pathlib import Path


class LocalProvider(BaseProvider):
    def __init__(self, local_path: Path):
        self._dataset_taget_path = local_path
        self.name = local_path.name

    def download(self):
        pass

    def get_dataset_dir(self):
        return self._dataset_taget_path
