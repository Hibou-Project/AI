from .utils.image import apply_image_transformations, get_image_channels_from_filter
from datasets import load_dataset, concatenate_datasets, DatasetDict
from .utils.label import parse_label
from huggingface_hub import HfApi
from pathlib import Path
from tqdm import tqdm

import yaml
import cv2
import os


class Dataset:

    def __init__(
        self,
        hf_url: str = None,
        path: str = None,
        hf_revision: str = "main",
        save_dir: str = None,
        image_transform: str = "RGB",
        load_label_other: bool = False,
    ):
        self._dataset = None
        self._hf_url = hf_url
        self._path = path
        self._hf_revision = hf_revision
        self._root_dir = None
        self._dataset_full_path = None  # Used only for online datasets
        self.image_transform = image_transform
        self.load_label_other = load_label_other

        if self._hf_url is not None and self._path is not None:
            raise RuntimeError("Only one of hf_url or path can be specified.")
        self._is_online = self._hf_url is not None

        if self._is_online:
            if save_dir is None:
                raise RuntimeError("Missing argument. Save dataset directory not set.")
            self._root_dir = save_dir
            self._load_online_dataset()
            self._name = self._hf_url.split("/")[-1]
        else:
            self._load_local_dataset()
            self._name = Path(self._path).name

    def _load_online_dataset(self):
        self._dataset = load_dataset(self._hf_url, revision=self._hf_revision)

    def _load_local_dataset(self):
        self._dataset = self._path
        self._root_dir = Path(self._path)

    def _get_dataset_sha(self):
        api = HfApi()
        info = api.dataset_info(self._hf_url)
        return info.sha

    def _get_export_marker_path(self):
        return os.path.join(self._root_dir, ".export_sha")

    def split(self, seed: int, base_split="train_validation_test", label_column="class_id"):
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded.")
        train_ratio = [0.8, 0.6]  # [class 0, class 1]
        valid_ratio = [0.1, 0.2]
        test_ratio = [0.1, 0.2]

        for i in range(len(train_ratio)):
            assert train_ratio[i] + valid_ratio[i] + test_ratio[i] == 1.0

        train_parts = []
        valid_parts = []
        test_parts = []

        num_classes = len(train_ratio)

        for cls in range(num_classes):
            cls_ds = self._dataset[base_split].filter(
                lambda x: x[label_column] == cls
            )

            cls_ds = cls_ds.shuffle(seed=seed)

            n = len(cls_ds)
            n_train = int(n * train_ratio[cls])
            n_valid = int(n * valid_ratio[cls])

            train_parts.append(cls_ds.select(range(0, n_train)))
            valid_parts.append(cls_ds.select(range(n_train, n_train + n_valid)))
            test_parts.append(cls_ds.select(range(n_train + n_valid, n)))

        train_ds = concatenate_datasets(train_parts).shuffle(seed=seed)
        valid_ds = concatenate_datasets(valid_parts).shuffle(seed=seed)
        test_ds = concatenate_datasets(test_parts).shuffle(seed=seed)

        self._dataset = DatasetDict({
            "train": train_ds,
            "validation": valid_ds,
            "test": test_ds,
        })

    def export_to_yolo(self):
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded.")
        if self._root_dir is None:
            raise RuntimeError("Save dataset directory not set.")

        if not self._is_online:
            print("Skipping export. Dataset is not online.")
            return

        self._dataset_full_path = self._name + "_" + self.image_transform
        save_dir = Path(self._root_dir, self._dataset_full_path)

        current_sha = self._get_dataset_sha()
        marker_path = self._get_export_marker_path()

        if os.path.exists(marker_path):
            with open(marker_path, "r") as f:
                saved_sha = f.read().strip()
            if saved_sha == current_sha and not save_dir.exists():
                print("Dataset already exported, but files do not exist. Re-exporting.")
            elif saved_sha == current_sha:
                print("Dataset already exported. Skipping.")
                return

        # Create the basic folders structure
        os.makedirs(save_dir, exist_ok=True)
        for split in ["train", "valid", "test"]:
            os.makedirs(f"{save_dir}/{split}", exist_ok=True)

        def export(ds, split_name):
            for idx, sample in enumerate(tqdm(ds, total=len(ds))):
                image = sample["image"]  # PIL.Image
                label = sample["raw_label"]  # YOLO format [[class, cx, cy, w, h], ...]
                img_name = sample["name"]
                txt_name = img_name.split(".")[0] + ".txt"

                label_formated = parse_label(label)

                if len(label_formated) == 1 and self.load_label_other:
                    continue

                # Do not apply transformation to RGB images (Better performance)
                if self.image_transform == "RGB":
                    image.save(save_dir / split_name / img_name, quality=95)
                else:
                    # Apply transformation
                    cv2_img = apply_image_transformations(image, self.image_transform)

                    base_name = os.path.splitext(img_name)[0]
                    img_path = save_dir / split_name / f"{base_name}.tiff"
                    cv2.imwrite(img_path, cv2_img)

                # Save YOLO labels
                lbl_path = save_dir / split_name / txt_name
                with open(lbl_path, "w") as file:
                    if not self.load_label_other and int(label_formated[0]) == 1:
                        continue  # create an empty label file
                    else:
                        file.write(label)

        split_mapping = {"train": "train", "validation": "valid", "test": "test"}
        for hf_split, folder_name in split_mapping.items():
            export(self._dataset[hf_split], folder_name)
        with open(marker_path, "w") as f:
            f.write(current_sha)

    def save_dataset_settings(self):
        path = self._dataset_full_path if self._is_online else self._path
        data_yaml = dict(
            train=f"{path}/train",
            val=f"{path}/valid",
            test=f"{path}/test",
            nc=2 if self.load_label_other else 1,
            channels=get_image_channels_from_filter(self.image_transform),
            names=['drone', 'other'] if self.load_label_other else ['drone'],
        )
        data_config_path = Path(self._root_dir, 'data.yaml')
        with open(data_config_path, 'w') as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=True)

    def get_config_path(self):
        return Path(self._root_dir, 'data.yaml')

    def __getitem__(self):
        return self._dataset
