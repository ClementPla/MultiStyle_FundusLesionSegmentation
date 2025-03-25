from copy import deepcopy

import torch
from nntools.utils import Config

from adptSeg.adaptation.const import _all_datasets, trained_probe_path
from adptSeg.adaptation.model_wrapper import ModelFeaturesExtractor
from adptSeg.adaptation.probe import ProbeModule
from fundseg.data.data_factory import ALL_DATASETS

# from captum.robust import PGD
from fundseg.utils.checkpoints import load_model_from_checkpoints


def get_probe_model_and_loss(
    model_type=ALL_DATASETS,
    probe_type=5,
    n_classes=5,
    as_regression=False,
    probe_datasets=_all_datasets,
    feature_type="encoder",
):
    model = load_model_from_checkpoints(
        train_datasets=model_type,
    ).cuda()
    encoder = ModelFeaturesExtractor(model, position=probe_type, feature_type=feature_type).cuda()
    probe = ProbeModule.load_from_checkpoint(
        trained_probe_path(probe_type, encoder=feature_type == "encoder"),
        featureExtractor=deepcopy(encoder),
        weights=torch.ones(n_classes),
        as_regression=as_regression,
        n_classes=n_classes,
        datasets=probe_datasets,
    ).cuda()
    loss = probe.criterion
    return probe, model, loss


def get_aptos_dataloader(batch_size, grade_filter=None):
    config = Config("configs/config.yaml")
    config["data"].pop("random_crop", None)
    config["data"]["batch_size"] = batch_size
    config["data"]["data_augmentation_type"] = None
    from fundus_data_toolkit.datamodules import CLASSIF_PATHS
    from fundus_data_toolkit.datamodules.classification import AptosDataModule

    datamodule = AptosDataModule(data_dir=CLASSIF_PATHS.APTOS, **config["data"])

    datamodule.setup_all()
    return datamodule.train_dataloader()
