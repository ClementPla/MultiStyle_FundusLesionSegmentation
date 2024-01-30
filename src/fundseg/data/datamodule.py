import os
from collections import namedtuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from nntools.dataset.utils import ConcatDataset, concat_datasets_if_needed
from pytorch_lightning import LightningDataModule
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, WeightedRandomSampler

from fundseg.data.datasets import get_datasets_from_config, get_image_dataset, get_resize_composer
from fundseg.data.utils import ALL_CLASSES, ALL_DATASETS


class FundusSegmentationDatamodule(LightningDataModule):
    def __init__(
        self,
        data_config,
        img_size=(512, 512),
        crop_size=None,
        datasets=ALL_DATASETS,
        classes=ALL_CLASSES,
        valid_size=0.1,
        eval_batch_size=2,
        batch_size=64,
        num_workers="auto",
        use_cache=False,
        preprocess=False,
        persistent_workers=True,
        dataset_multiplicative_size=1,
        data_aug_ops="light",
        weightedSampler=False,
        return_tag=False,
        disable_cropping=False,
        disable_data_aug=False,
    ):
        super(FundusSegmentationDatamodule, self).__init__()
        self.crop_size = crop_size
        self.img_size = img_size
        self.data_config = data_config
        self.batch_size = batch_size
        if torch.cuda.is_available():
            self.batch_size = self.batch_size // torch.cuda.device_count()
        self.valid_size = valid_size
        self.datasets = datasets
        self.classes = classes
        self.eval_batch_size = eval_batch_size
        self.dataset_multiplicative_size = dataset_multiplicative_size
        self.preprocess = preprocess
        self.persistent_workers = persistent_workers
        self.data_aug_ops_type = data_aug_ops
        if self.preprocess:
            self._update_to_preprocess_folder()
        if num_workers == "auto":
            self.num_workers = os.cpu_count() // torch.cuda.device_count()
        else:
            self.num_workers = int(num_workers)
        self.train = self.val = self.test = None
        self.use_cache = use_cache
        self.weightedSampler = weightedSampler
        self.disable_cropping = disable_cropping
        self.disable_data_aug = disable_data_aug
        self.return_tag = return_tag

    def _update_to_preprocess_folder(self):
        for k, v in self.data_config["train"].items():
            if "url" in k and "img" in k:
                path = os.path.join(v, "preprocessed/")
                if os.path.exists(path):
                    self.data_config["train"][k] = path
                else:
                    print(f"Preprocessing folder {path} does not exists")

        for k, v in self.data_config["test"].items():
            if "url" in k and "img" in k:
                path = os.path.join(v, "preprocessed/")
                if os.path.exists(path):
                    self.data_config["test"][k] = path
                else:
                    print(f"Preprocessing folder {path} does not exists")

    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            datasets = get_datasets_from_config(
                root=self.data_config["root_path"],
                config=self.data_config["train"],
                use_cache=self.use_cache,
                sets=self.datasets,
                labels=self.classes,
                seed=1234,
                shape=self.img_size,
                split_ratio=self.valid_size,
                train_or_test="train",
            )
            train_sets = datasets["core"]
            train_sets = concat_datasets_if_needed(datasets["core"])
            if self.weightedSampler and len(datasets["core"]) > 1:
                weights = []
                for i, d in enumerate(train_sets):
                    d_samples = np.ones(len(d)) * i
                    weights.append(d_samples)
                weights = np.concatenate(weights)
                print("Computing samples weights")
                weights = compute_sample_weight("balanced", weights)
                self.distributer = WeightedRandomSampler(weights, len(weights), replacement=True)

                print("Using weighted sampler", np.unique(weights, return_counts=True))

            composer = train_sets.composer
            if not isinstance(composer, list):
                composer = [composer]
            for c in composer:
                if self.disable_data_aug:
                    ops = []
                else:
                    ops = [*self.data_aug_ops()]
                if self.crop_size is not None and not self.disable_cropping:
                    ops.append(A.RandomCrop(height=self.crop_size[0], width=self.crop_size[1]))
                c.add(A.Compose(ops + self.normalizing_ops(), additional_targets={"roi": "mask"}))

            self.train = train_sets
            self.train.return_tag = self.return_tag

            if len(datasets["split"]):
                val_sets = concat_datasets_if_needed(datasets["split"])
                composer = val_sets.composer
                if not isinstance(composer, list):
                    composer = [composer]
                for c in composer:
                    c.add(A.Compose(self.normalizing_ops(), additional_targets={"roi": "mask"}))
                self.val = val_sets
                self.val.use_cache = False
                self.val.return_tag = self.return_tag

        if stage == "test":
            datasets = get_datasets_from_config(
                root=self.data_config["root_path"],
                config=self.data_config["test"],
                sets=ALL_DATASETS,
                labels=self.classes,
                seed=1234,
                shape=self.img_size,
                split_ratio=0,
                train_or_test="test",
                use_cache=False,
            )["core"]
            for d in datasets:
                d.composer.add(A.Compose(self.normalizing_ops(), additional_targets={"roi": "mask"}))
                d.return_tag = self.return_tag
            self.test = datasets

    def normalizing_ops(self):
        return [A.Normalize(always_apply=True), ToTensorV2(transpose_mask=True)]

    def data_aug_ops(self):
        light_ops = (A.HorizontalFlip(p=0.5), A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT))

        medium_ops =  (*light_ops, A.RandomBrightnessContrast(p=0.5), A.VerticalFlip(p=0.5))

        heavy_ops =  (*medium_ops, A.RandomGamma(p=0.5), A.GaussianBlur(p=0.5))
        if self.data_aug_ops_type == "light":
            return light_ops
        elif self.data_aug_ops_type == "medium":
            return medium_ops
        elif self.data_aug_ops_type == "heavy":
            return heavy_ops
        else:
            raise ValueError(f"Invalid data augmentation ops {self.data_aug_ops_type}.")

    def train_dataloader(self, independant_dataloader=False):
        if independant_dataloader:
            if isinstance(self.train, ConcatDataset):
                dataloaders = []
                for d in self.train.datasets:
                    dataloader = DataLoader(
                        d,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.num_workers,
                        prefetch_factor=2,
                        persistent_workers=self.num_workers > 0 and self.persistent_workers,
                        pin_memory=True,
                    )
                    dataloaders.append(dataloader)
                return dataloaders
        return DataLoader(
            self.train,
            shuffle=True if self.weightedSampler is None else False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.distributer if self.weightedSampler else None,
            prefetch_factor=1,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            pin_memory=True,
        )

    def custom_dataloader(
        self,
        img_path=None,
        dataset=None,
        name="",
        force_shuffle=False,
        batch_size=None,
        return_indices=True,
    ):
        if dataset is None:
            dataset = get_image_dataset(img_path, input_shape=self.img_size)
            
        dataset.composer = self.simple_composer
        dataset.return_indices = return_indices
        DataID = namedtuple("Dataset", "value name")
        dataset.id = DataID(-1, name)
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=force_shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            pin_memory=True,
        )

    @property
    def simple_composer(self):
        final_ops = [A.Normalize(always_apply=True), ToTensorV2(transpose_mask=True)]
        composer = get_resize_composer(self.img_size)
        composer.add(A.Compose(final_ops, additional_targets={"roi": "mask"}))
        return composer

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=True,
        )

    def test_dataloader(self, force_shuffle=False):
        return [
            DataLoader(
                d,
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                shuffle=force_shuffle,
                persistent_workers=self.num_workers > 0 and self.persistent_workers,
                pin_memory=True,
            )
            for d in self.test
        ]
