from enum import Enum
from pathlib import Path
from typing import Dict, List

import cv2
from fundus_data_toolkit.data_aug import DAType
from fundus_data_toolkit.datamodules import SEG_PATHS, Task, register_paths
from fundus_data_toolkit.datamodules.segmentation import (
    DDRDataModule_s,
    FGADRDataModule_s,
    IDRiDDataModule_s,
    MESSIDORDataModule_s,
    RETLESDataModule_s,
)
from fundus_data_toolkit.datamodules.utils import merge_existing_datamodules
from tqdm import tqdm


class FundusDataset(str, Enum):
    IDRID: str = "IDRID"
    FGADR: str = "FGADR"
    MESSIDOR: str = "MESSIDOR"
    DDR: str = "DDR"
    RETLES: str = "RETLES"


ALL_DATASETS = [FundusDataset.IDRID, FundusDataset.FGADR, FundusDataset.MESSIDOR, FundusDataset.DDR, FundusDataset.RETLES]


def setup_data_toolkit(paths: Dict[str, Path]):
    register_paths(paths, Task.SEGMENTATION)


def setup_data_from_config(datasets: Dict[str, str]):
    paths = {}
    for dataset in datasets:
        paths[dataset] = Path(datasets[dataset])
        assert paths[dataset].exists(), f"Path {paths[dataset]} does not exist"
    setup_data_toolkit(paths)


def get_datamodule(datasets: List[str], training_sets: List[str], dataset_args, separate_test_test: bool = True):
    all_datamodules = []
    for d in datasets:
        match FundusDataset(d):
            case FundusDataset.IDRID:
                datamodule = IDRiDDataModule_s(
                    SEG_PATHS.IDRID, precise_autocrop=True, flag=cv2.IMREAD_COLOR, **dataset_args
                ).setup_all()

            case FundusDataset.FGADR:
                datamodule = FGADRDataModule_s(
                    SEG_PATHS.FGADR,
                    precise_autocrop=True,
                    flag=cv2.IMREAD_COLOR,
                    **dataset_args,
                ).setup_all()
            case FundusDataset.MESSIDOR:
                datamodule = MESSIDORDataModule_s(
                    SEG_PATHS.MESSIDOR,
                    precise_autocrop=True,
                    flag=cv2.IMREAD_COLOR,
                    **dataset_args,
                ).setup_all()
            case FundusDataset.DDR:
                datamodule = DDRDataModule_s(
                    SEG_PATHS.DDR,
                    precise_autocrop=True,
                    flag=cv2.IMREAD_COLOR,
                    **dataset_args,
                ).setup_all()
            case FundusDataset.RETLES:
                datamodule = RETLESDataModule_s(
                    SEG_PATHS.RETLES,
                    precise_autocrop=True,
                    flag=cv2.IMREAD_COLOR,
                    **dataset_args,
                ).setup_all()

        if datamodule.train is not None:
            datamodule.train.tag = FundusDataset(d)
        if datamodule.val is not None:
            datamodule.val.tag = FundusDataset(d)
        if datamodule.test is not None:
            datamodule.test.tag = FundusDataset(d)

        if d not in training_sets:
            datamodule.train = None
        all_datamodules.append(datamodule)

    return merge_existing_datamodules(all_datamodules, separate_test_sets=separate_test_test)


def get_datamodule_from_config(
    config: Dict[str, str], training_datasets: List[str], dataset_args, separate_test_test=True
):
    setup_data_from_config(config)
    available_datasets = [FundusDataset(d.upper()) for d in config]
    training_datasets = [FundusDataset(d.upper()) for d in training_datasets]
    datamodule = get_datamodule(
        available_datasets, training_datasets, dataset_args, separate_test_test=separate_test_test
    )

    print(f"Training on datasets: {training_datasets}")
    print(f"Training samples: {len(datamodule.train)}, {len(datamodule.train_dataloader())} batches")
    print(f"Validation samples: {len(datamodule.val)}, {len(datamodule.val_dataloader())} batches")
    print(f"Test samples: {len(datamodule.test)}")

    return datamodule


def precache_datamodule(config: Dict[str, str], dataset_args):
    datamodule = get_datamodule_from_config(config, dataset_args)
    train_dataloader = datamodule.train_dataloader()
    for _ in tqdm(train_dataloader, total=len(train_dataloader)):
        pass
    if datamodule.val:
        val_dataloader = datamodule.val_dataloader()
        for _ in tqdm(val_dataloader, total=len(val_dataloader)):
            pass
    if datamodule.test:
        test_dataloader = datamodule.test_dataloader()
        if not isinstance(test_dataloader, list):
            test_dataloader = [test_dataloader]

        for dl in test_dataloader:
            for _ in tqdm(dl, total=len(dl)):
                pass
