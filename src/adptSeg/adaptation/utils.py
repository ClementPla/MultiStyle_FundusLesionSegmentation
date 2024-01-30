import nntools.dataset as D
import numpy as np
import pandas as pd
import torch

# from captum.robust import PGD
from fundseg.data.datamodule import FundusSegmentationDatamodule
from fundseg.models.smp_model import SMPModel
from fundseg.utils.runs import ALL_DATASETS, models_path
from nntools.utils import Config

from adptSeg.adaptation.const import _all_datasets, trained_probe_path
from adptSeg.adaptation.encoder_wrapper import ModelEncoder
from adptSeg.adaptation.probe import ProbeModule


def get_probe_model_and_loss(
    model_type=ALL_DATASETS, probe_type=5, n_classes=5, as_regression=False, probe_datasets=_all_datasets
):
    model = SMPModel.load_from_checkpoint(models_path[model_type]).cuda()
    probe_position = 5
    encoder = ModelEncoder(model, encoding_position=probe_position).cuda()
    probe = ProbeModule.load_from_checkpoint(
        trained_probe_path[probe_type],
        encoder=encoder,
        weights=torch.ones(n_classes),
        as_regression=as_regression,
        n_classes=n_classes,
        datasets=probe_datasets,
    ).cuda()
    loss = probe.criterion
    return probe, model, loss


def model_from_checkpoint(dataset):
    model = SMPModel.load_from_checkpoint(models_path[dataset]).cuda()
    return model


def get_aptos_dataloader(batch_size, grade_filter=None):
    config = Config("configs/config.yaml")
    config_data = Config("configs/data_config.yaml")
    datamodule = FundusSegmentationDatamodule(config_data, **config["data"])
    datamodule.persistent_workers = True
    aptos_path = "/home/tmp/clpla/data/aptos/train/images/"
    if grade_filter:
        grade_dataset = pd.read_csv("/home/tmp/clpla/data/aptos/train.csv")
        dataset = D.ClassificationDataset(
            aptos_path,
            shape=datamodule.img_size,
            auto_pad=True,
            keep_size_ratio=True,
            label_dataframe=grade_dataset,
            gt_column="diagnosis",
            file_column="id_code",
        )
        dataset.subset(np.where(dataset.gts["diagnosis"] >= grade_filter))
        dataloader = datamodule.custom_dataloader(dataset=dataset, batch_size=batch_size, force_shuffle=False)
    else:
        dataloader = datamodule.custom_dataloader(aptos_path, batch_size=batch_size, force_shuffle=False)
    return dataloader
