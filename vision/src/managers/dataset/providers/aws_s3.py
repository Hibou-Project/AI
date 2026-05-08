from src.managers.dataset.providers.base import BaseProvider
from src.logger import CustomLogger
from src.settings import SETTINGS
from pathlib import Path
from tqdm import tqdm

import logging
import boto3

logger = CustomLogger("AWS Provider").get_logger()

class AWSProvider(BaseProvider):
    def __init__(self, bucket: str, folder: str, region: str, sampling_ratio: float = 1.0):
        self.bucket = bucket
        self._folder = folder
        self.name = f"{bucket}_{folder}"
        self._sampling_ratio = sampling_ratio

        session = boto3.Session(
            aws_access_key_id=SETTINGS.TOKEN_AWS_KEY_ID,
            aws_secret_access_key=SETTINGS.TOKEN_AWS_SECRET_KEY,
            region_name=region
        )

        self._s3 = session.client("s3")

        self._dataset_target_path = (
                Path(SETTINGS.DATASET_PATH)
                / ".downloads"
                / f"aws_{self.name.replace('/', '_')}"
        )

    def get_dataset_dir(self):
        return self._dataset_target_path

    def get_dataset_sampling_ratio(self):
        return self._sampling_ratio

    def download(self):

        if self._dataset_target_path.exists():
            logging.info("Dataset already downloaded. Skipping")
            return

        paginator = self._s3.get_paginator("list_objects_v2")

        pages = list(paginator.paginate(
            Bucket=self.bucket,
            Prefix=self._folder
        ))

        total_files = 0

        for page in pages:
            if "Contents" in page:
                total_files += len(page["Contents"])

        pbar = tqdm(total=total_files, desc="Downloading dataset from AWS")

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]

                if key.endswith("/"):
                    continue

                rel_path = key.removeprefix(self._folder).lstrip("/")
                local_path = self._dataset_target_path / rel_path

                local_path.parent.mkdir(parents=True, exist_ok=True)

                self._s3.download_file(self.bucket, key, str(local_path))

                pbar.update(1)

        pbar.close()
