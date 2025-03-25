from pathlib import Path
from typing import Union

import torch

from fundseg.data.data_factory import FundusDataset


def map_dataset_to_integer(dataset: Union[FundusDataset, str]) -> int:
    match FundusDataset(dataset):
        case FundusDataset.IDRID:
            return 0
        case FundusDataset.MESSIDOR:
            return 1
        case FundusDataset.DDR:
            return 2
        case FundusDataset.FGADR:
            return 3
        case FundusDataset.RETLES:
            return 4
        case _:
            raise ValueError(f"Dataset {dataset} not recognized")


def map_integer_to_dataset(dataset: int) -> FundusDataset:
    match dataset:
        case 0:
            return FundusDataset.IDRID
        case 1:
            return FundusDataset.MESSIDOR
        case 2:
            return FundusDataset.DDR
        case 3:
            return FundusDataset.FGADR
        case 4:
            return FundusDataset.RETLES
        case _:
            raise ValueError(f"Dataset {dataset} not recognized")


def batch_dataset_to_integer(datasets: Union[list[FundusDataset], list[str]]) -> list[int]:
    return [map_dataset_to_integer(d) for d in datasets]


def batch_integer_to_dataset(datasets: Union[list[int], torch.Tensor]) -> list[FundusDataset]:
    return [map_integer_to_dataset(d) for d in datasets]


_all_datasets = [
    FundusDataset.IDRID,
    FundusDataset.MESSIDOR,
    FundusDataset.DDR,
    FundusDataset.FGADR,
    FundusDataset.RETLES,
]


def get_class_mapping(datasets: list[FundusDataset]) -> dict[int, int]:
    return {map_dataset_to_integer(d): i for i, d in enumerate(datasets)}


def trained_probe_path(position: int, encoder=True) -> str:
    import wandb

    root = Path("/home/clement/Documents/Projets/MultiStyle_FundusLesionSegmentation/checkpoints/probing")
    api = wandb.Api()
    runs = api.runs("liv4d-polytechnique/Probing-Lesions-Segmentation-Positions")
    for run in runs:
        is_encoder = run.config["feature_type"] == "ENCODER"
        if run.config["position"] == position and is_encoder == encoder:
            ckpt_folder = root / run.name
            ckpt_files = list(ckpt_folder.glob("epoch*"))
            return str(ckpt_files[0])
