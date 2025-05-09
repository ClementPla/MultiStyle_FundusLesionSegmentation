import argparse
import logging
import warnings

import torch
from nntools.utils import Config
from pytorch_lightning import Trainer, seed_everything

from fundseg.data.data_factory import ALL_DATASETS, get_datamodule_from_config
from fundseg.data.utils import ALL_CLASSES
from fundseg.models import get_model
from fundseg.utils.callbacks import get_callbacks
from fundseg.utils.logger import get_wandb_logger

warnings.filterwarnings("ignore")
import wandb

logging.basicConfig(level=logging.WARNING)
seed_everything(1234, workers=True)

torch.set_float32_matmul_precision("high")


def run_train_test(architecture, config, train_datasets):
    project_name = config["logger"]["project"]
    if not isinstance(train_datasets, list):
        train_datasets = [train_datasets]

    tags = train_datasets

    datamodule = get_datamodule_from_config(config["datasets"], train_datasets, config["data"])
    test_dataset_id = [d.id for d in datamodule.test]
    model = get_model(architecture, **config["model"], test_dataset_id=test_dataset_id)

    wandb_logger = get_wandb_logger(
        project_name=project_name,
        tracked_params=config.tracked_params,
        tags=tags,
        item_check_if_run_exists=("model/architecture", architecture),
    )
    callbacks = get_callbacks(
        config,
        classes=ALL_CLASSES,
        wandb_logger=wandb_logger,
    )
    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        strategy="ddp",
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())
    trainer.test(model, dataloaders=datamodule.test_dataloader())

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(prog="Segmentation Lesions in Fundus")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.003)
    parser.add_argument("--optimizer", type=str, help="Optimizer", default="adamw")
    parser.add_argument("--log_dice", type=bool, help="Use the log of dice loss", default=False)
    parser.add_argument("--dice_smooth", type=float, help="Smoothness constant for dice loss", default=0.4)
    parser.add_argument("--dataset", type=str, help="Train dataset", nargs="+")

    args = parser.parse_args()

    datasets = args.dataset
    if datasets == ["all"]:
        datasets = ALL_DATASETS
    config_file = "configs/config.yaml"
    config = Config(config_file)
    config["model"] = {}
    config["model"]["lr"] = float(args.lr)
    config["model"]["optimizer"] = args.optimizer
    config["model"]["smooth_dice"] = args.dice_smooth
    config["model"]["log_dice"] = args.log_dice

    model_name = args.model

    run_train_test(model_name, config, datasets)


if __name__ == "__main__":
    main()
