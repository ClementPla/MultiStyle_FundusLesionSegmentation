import argparse
import logging
import os

import torch
from fundseg.data.datamodule import (
    ALL_CLASSES,
    ALL_DATASETS,
    FundusSegmentationDatamodule,
)
from fundseg.data.utils import Dataset, get_dataset_from_name
from fundseg.models import get_model
from fundseg.utils.callbacks import get_callbacks
from fundseg.utils.logger import check_if_run_already_started, init_logger
from nntools.utils import Config
from pytorch_lightning import Trainer, seed_everything

import wandb

logging.basicConfig(level=logging.WARNING)
seed_everything(1234, workers=True)

torch.set_float32_matmul_precision('high')


def run_train_test(model, datamodule, config):
    tags = datamodule.datasets.name
    if not isinstance(tags, list):
        tags = [tags]
    
    # tags += ['Finetuned from ALL']
    wandb_logger = init_logger(
        config["logger"],
        tags=tags,
        **config.tracked_params,
        model_name=model.model_name,
        train_dataset=tags,
    )

    callbacks = get_callbacks(
        config,
        classes=ALL_CLASSES,
        wandb_logger=wandb_logger,
    )
    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        strategy="ddp_find_unused_parameters_true",
        # strategy="ddp",
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=datamodule)
    datamodule.setup("test")

    test_dataloaders = datamodule.test_dataloader()
    for dataloader in test_dataloaders:
        model.dataset_name = dataloader.dataset.id.name
        trainer.test(model=model, dataloaders=dataloader, ckpt_path="best")
    wandb_logger.experiment.finish()
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(prog="Segmentation Lesions in Fundus")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--data_aug_ops", type=str, help="Type of data augmentation")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--optimizer", type=str, help="Optimizer")
    parser.add_argument("--log_dice", type=bool, help="Use the log of dice loss")
    parser.add_argument("--dice_smooth", type=float, help="Smoothness constant for dice loss")
    parser.add_argument("--dataset", type=str, help="Train dataset")

    args = parser.parse_args()
    
    # if args.dataset == 'all':
    #     datasets = ALL_DATASETS
    # else:
    #     datasets = get_dataset_from_name(args.dataset)
    datasets = Dataset.IDRID | Dataset.RETINAL_LESIONS
    config_file = "configs/config.yaml"
    config = Config(config_file)
    config_data = Config("configs/data_config.yaml")
    
    config["data"]["data_aug_ops"] = args.data_aug_ops
    config['model'] = {}
    config["model"]["lr"] = float(args.lr)
    config["model"]["optimizer"] = args.optimizer
    config['model']['smooth_dice'] = args.dice_smooth
    config['model']['log_dice'] = args.log_dice
    
    model_name = args.model
    fundus_datamodule = FundusSegmentationDatamodule(
        datasets=datasets, data_config=config_data, **config["data"]
    )

    fundus_datamodule.setup("fit")

    
    fundus_datamodule.setup("test")
    
    model = get_model(model_name, **config["model"])
    
    run_train_test(model, fundus_datamodule, config)

    
if __name__ == "__main__":
    main()