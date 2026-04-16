from datasets import load_dataset, Image, concatenate_datasets, DatasetDict
from .utils.image import apply_image_transformations
from .utils.label import parse_label
from pathlib import Path
from tqdm import tqdm

import cv2
import os


class Dataset:

    def __init__(self, hf_url: str = None, path: str = None, hf_revision: str = "main", save_dir: str = None):
        self.dataset = None
        self.hf_url = hf_url
        self.path = path
        self.hf_revision = hf_revision
        self.root_dir = None

        if self.hf_url is not None and self.path is not None:
            raise RuntimeError("Only one of hf_url or path can be specified.")
        self.is_online = self.hf_url is not None

        if self.is_online:
            if save_dir is None:
                raise RuntimeError("Missing argument. Save dataset directory not set.")
            self.root_dir = save_dir
            self._load_online_dataset()
        else:
            self._load_local_dataset()

    def _load_online_dataset(self):
        self.dataset = load_dataset(self.hf_url, revision=self.hf_revision)

    def _load_local_dataset(self):
        self.dataset = self.path
        self.root_dir = Path(self.path)

    def split(self, seed: int, base_split="train_validation_test", label_column="class_id"):
        if self.dataset is None:
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
            cls_ds = self.dataset[base_split].filter(
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

        self.dataset = DatasetDict({
            "train": train_ds,
            "validation": valid_ds,
            "test": test_ds,
        })

    def export_to_yolo(self, image_transform, load_label_other: bool):
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded.")
        if self.root_dir is None:
            raise RuntimeError("Save dataset directory not set.")

        for split in ["train", "valid", "test"]:
            os.makedirs(f"{self.root_dir}/{split}", exist_ok=True)

        def export(ds, split_name):
            for idx, sample in enumerate(tqdm(ds, total=len(ds))):
                image = sample["image"]  # PIL.Image
                label = sample["raw_label"]  # YOLO format [[class, cx, cy, w, h], ...]
                img_name = sample["name"]
                txt_name = img_name.split(".")[0] + ".txt"

                label_formated = parse_label(label)

                # Skip empty labels if required
                if len(label_formated) == 1 and load_label_other:
                    continue

                # Apply transformation
                cv2_img = apply_image_transformations(image, image_transform)

                base_name = os.path.splitext(img_name)[0]
                img_path = os.path.join(self.root_dir, split_name, f"{base_name}.tiff")
                cv2.imwrite(img_path, cv2_img)

                # Save YOLO labels
                lbl_path = os.path.join(self.root_dir, split_name, txt_name)
                with open(lbl_path, "w") as f:
                    if not load_label_other and int(label_formated[0]) == 1:
                        continue  # create an empty label file
                    else:
                        f.write(label)

        split_mapping = {"train": "train", "validation": "valid", "test": "test"}
        for hf_split, folder_name in split_mapping.items():
            export(self.dataset[hf_split], folder_name)

    def __getitem__(self):
        return self.dataset
