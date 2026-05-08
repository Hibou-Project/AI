from managers.dataset.providers.huggingface import HuggingFaceProviders
from managers.dataset.providers.local import LocalProvider
from managers.dataset.providers.aws_s3 import AWSProvider
from managers.dataset.utils.files import list_files
from src.logger import CustomLogger
from pathlib import Path
from tqdm import tqdm

import shutil
import random
import yaml
import re
import os

from utils.image import get_image_channels_from_filter

logger = CustomLogger("Dataset").get_logger()


class Dataset:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

    def __init__(
            self,
            providers,
            image_transform,
            save_dir,
            split_ratio,
            seed=53
    ):
        self.image_transform = image_transform
        self.sub_dataset = []
        self.save_dir = save_dir
        self._seed = seed
        self._split_ratio = split_ratio

        for key in providers.keys():
            for config in providers[key]:
                if key == "hf":
                    self.sub_dataset.append(
                        HuggingFaceProviders(
                            name=config["name"],
                            hf_revision=config["revision"])
                    )
                elif key == "aws_s3":
                    self.sub_dataset.append(
                        AWSProvider(
                            bucket=config["bucket"],
                            folder=config["folder"],
                            region=config["region"]
                        )
                    )
                elif key == "local":
                    self.sub_dataset.append(LocalProvider(
                        local_path=Path(config["path"])
                    ))
                else:
                    raise ValueError(f"Invalid provider: {key}")

        dataset_name = ""
        for sub_dataset in self.sub_dataset:
            dataset_name += sub_dataset.name + "_"
        dataset_name = dataset_name.replace("/", "_") + self.image_transform

        self._dataset_target_path = Path(self.save_dir, dataset_name)

    def download(self):
        for sub_dataset in self.sub_dataset:
            logger.info(f"Start downloading {sub_dataset.name}")
            sub_dataset.download()

    def get_dataset_dir(self):
        return self._dataset_target_path

    def _post_process(self):

        # Create an empty label if label is 1
        label_paths = list_files(
            directory=self._dataset_target_path,
            extensions=[".txt"],
            include_root_directory=True,
            recursive=False,
        )

        for label_path in label_paths:
            with open(label_path, "r") as f:
                content = f.read()

            content = [x for x in re.split(r"\s+", content) if x]

            if content and content[0] == "1":
                # create an empty file (overwrite existing content)
                open(label_path, "w").close()

    def merge(self):
        if not self._dataset_target_path.exists():
            self._dataset_target_path.mkdir(parents=True)
            logger.info(f"Create dataset directory: {self._dataset_target_path}")

        i = 0

        for sub_dataset in self.sub_dataset:
            dataset_dir = sub_dataset.get_dataset_dir()

            files = list_files(
                directory=dataset_dir,
                extensions=self.ALLOWED_EXTENSIONS,
                include_root_directory=True,
                recursive=False,
            )

            for file in tqdm(
                    files,
                    desc=f"Processing {dataset_dir}",
                    unit="file"
            ):
                root_directory = file.parent
                img_extension_name = file.suffix
                root_file_name = file.stem
                label_file_name = root_file_name + ".txt"

                new_img_file_name = f"{i}{img_extension_name}"
                new_label_file_name = f"{i}.txt"

                if not (dataset_dir / label_file_name).exists():
                    logger.warning(
                        f"Label file not found: {dataset_dir / label_file_name}, skipping"
                    )
                    continue

                # Copy image
                shutil.copy(file, self._dataset_target_path / new_img_file_name)

                # Copy label
                shutil.copy(
                    root_directory / label_file_name,
                    self._dataset_target_path / new_label_file_name
                )

                i += 1

        self._post_process()

    def split(self):
        train_ratio = self._split_ratio[0]
        val_ratio = self._split_ratio[1]

        pairs = []

        print(self._dataset_target_path)

        img_paths = list_files(
            directory=self._dataset_target_path,
            extensions=self.ALLOWED_EXTENSIONS,
            include_root_directory=True,
            recursive=False,
        )

        for img_path in img_paths:

            if img_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                continue

            label_path = self._dataset_target_path / f"{img_path.stem}.txt"

            if label_path.exists():
                pairs.append((img_path, label_path))

        print(f"Found {len(pairs)} pairs")

        random.seed(self._seed)
        random.shuffle(pairs)

        n = len(pairs)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        test_pairs = pairs[val_end:]

        def ds_split(split_name, split_pairs):

            target_dir = self._dataset_target_path / split_name

            for img_path, label_path in tqdm(
                    split_pairs,
                    desc=f"Copying {split_name}",
                    unit="pair"
            ):
                shutil.move(img_path, target_dir / img_path.name)
                shutil.move(label_path, target_dir / label_path.name)

        for split in ["train", "valid", "test"]:
            os.makedirs(f"{self._dataset_target_path}/{split}", exist_ok=True)

        ds_split("train", train_pairs)
        ds_split("valid", val_pairs)
        ds_split("test", test_pairs)

    def save_dataset_settings(self):
        path = self._dataset_target_path
        data_yaml = dict(
            train=f"train",
            val=f"valid",
            test=f"test",
            nc=1,
            channels=get_image_channels_from_filter(self.image_transform),
            names=['drone'],
        )
        data_config_path = Path(path, 'data.yaml')
        logger.info(f"Saving dataset settings to {data_config_path}")
        with open(data_config_path, 'w') as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=True)

    def get_config_path(self):
        return self._dataset_target_path / 'data.yaml'
