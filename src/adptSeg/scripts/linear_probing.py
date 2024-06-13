import os
import sys
from argparse import ArgumentParser

import torch
from adptSeg.adaptation.model_wrapper import FeatureType, ModelFeaturesExtractor
from adptSeg.adaptation.probe import ProbeModule
from fundseg.data.data_factory import ALL_DATASETS, get_datamodule_from_config
from fundseg.utils.checkpoints import load_model_from_checkpoints
from nntools.utils import Config
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

import wandb

# This hack is needed to allow deserializing the model
sys.path.append("src/fundseg/")


@rank_zero_only
def init_wandb(name, hparams):
    wandb.init(project=name, config=hparams)


def train(feature_type, position):
    config_file = "configs/config.yaml"
    config = Config(config_file)
    config["data"]["batch_size"] = 96
    config["data"]["eval_batch_size"] = 16
    config["data"]["random_crop"] = None
    # config["data"]["use_cache"] = True
    model_name = "unet_seresnext50_32x4d"

    datasets = ALL_DATASETS

    model = load_model_from_checkpoints(train_datasets=datasets)

    fundus_datamodule = get_datamodule_from_config(
        config["datasets"], training_datasets=datasets, dataset_args=config["data"], separate_test_test=False
    )
    fundus_datamodule.train.return_tag = True
    fundus_datamodule.val.return_tag = True
    fundus_datamodule.test.return_tag = True

    fundus_datamodule.train.use_cache = True
    fundus_datamodule.val.use_cache = True
    fundus_datamodule.test.use_cache = False
    fundus_datamodule.train.init_cache()
    fundus_datamodule.val.init_cache()

    weights = []
    dataset = fundus_datamodule.train
    for d in dataset.datasets:
        weights.append(len(d))
    weights = torch.Tensor(weights)

    weights = weights.sum() / (weights * len(weights))

    hparams = {"position": position, "model_name": model_name, "feature_type": feature_type.name}

    featureExtractor = ModelFeaturesExtractor(model, position=position, feature_type=feature_type)
    logger = WandbLogger(
        project="Probing-Lesions-Segmentation-Positions",
        config=hparams,
    )

    probingModel = ProbeModule(featureExtractor, n_classes=len(datasets), weights=weights)

    if os.environ.get("LOCAL_RANK", None) is None:
        os.environ["WANDB_RUN_NAME"] = logger.experiment.name

    checkpoint = ModelCheckpoint(
        monitor="MulticlassAccuracy",
        mode="max",
        save_last=True,
        auto_insert_metric_name=True,
        save_top_k=1,
        dirpath=os.path.join("checkpoints", "probing", os.environ["WANDB_RUN_NAME"]),
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=[0, 1],
        max_epochs=50,
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="MulticlassAccuracy", mode="max", patience=10),
            LearningRateMonitor(),
        ],
        log_every_n_steps=30,
        logger=logger,
        check_val_every_n_epoch=1,
    )

    trainer.fit(
        probingModel,
        train_dataloaders=fundus_datamodule.train_dataloader(),
        val_dataloaders=fundus_datamodule.val_dataloader(),
    )

    trainer.test(probingModel, dataloaders=fundus_datamodule.test_dataloader(), ckpt_path="best")
    featureExtractor.delete_hook()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--feature_type",
        type=FeatureType,
        choices=[FeatureType.DECODER, FeatureType.ENCODER],
        default=FeatureType.DECODER,
    )
    parser.add_argument("--position", type=int, default=1)
    opts = parser.parse_args()
    feature_type = FeatureType(opts.feature_type)
    position = opts.position
    train(feature_type, position)
