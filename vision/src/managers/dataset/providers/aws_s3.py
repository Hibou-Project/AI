from managers.dataset.providers.base import BaseProvider
from settings import SETTINGS
from pathlib import Path
import boto3


class AWSProvider(BaseProvider):
    def __init__(self, bucket: str, folder: str):
        self.name = self.bucket = bucket
        self._dataset_taget_path = None
        self.folder = folder
        self._s3 = boto3.client(
            "s3",
            aws_access_key_id="xxx",
            aws_secret_access_key="xxx",
            region_name="eu-central-1"
        )

    def get_dataset_dir(self):
        return self._dataset_taget_path

    def download(self):
        pass
        # response = self._s3.list_objects_v2(
        #     Bucket=self.bucket,
        #     Prefix=self.folder
        # )
        # for obj in response.get("Contents", []):
        #     s3_key = obj["Key"]
        #
        #     # Skip "folder" placeholders
        #     if s3_key.endswith("/"):
        #         continue
        #
        #     # Local file path
        #     # relative_path = os.path.relpath(s3_key, self.folder)
        #     local_file_path = Path(SETTINGS.DATASET_DIR, ".tmp")
        #
        #     # Create local subdirectories
        #     # os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        #     local_file_path.mkdir(parents=True, exist_ok=True)
        #
        #     print(f"Downloading {s3_key} -> {local_file_path}")
        #
        #     self._s3.download_file(
        #         self.bucket,
        #         s3_key,
        #         local_file_path
        #     )

    # def save(self, save_dir: str, image_transform: str):
    #     pass
