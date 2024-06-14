from typing import Union

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


_all_datasets = [
    FundusDataset.IDRID,
    FundusDataset.MESSIDOR,
    FundusDataset.DDR,
    FundusDataset.FGADR,
    FundusDataset.RETLES,
]
