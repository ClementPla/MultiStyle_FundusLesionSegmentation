import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.utilities import rank_zero_only

import wandb
from fundseg.data.utils import ALL_CLASSES


def onehot_to_label(predmap):
    if predmap.ndim == 3:
        return predmap
    elif predmap.ndim == 4:
        b, c, h, w = predmap.shape
        zero_class = torch.zeros(
            (b, 1, h, w),
            dtype=predmap.dtype,
            device=predmap.device,
            requires_grad=False,
        )
        out = torch.cat([zero_class, predmap], 1).long()
        return torch.argmax(out, 1)
    else:
        raise ValueError(f"The vector should either have 3 or 4 dimension, but got a shape of {predmap.shape}")


class LogPredictionSamplesCallback(Callback):
    def __init__(self, wandb_logger, classes, n_images=8, frequency=10):
        self.n_images = n_images
        self.wandb_logger = wandb_logger
        self.frequency = frequency
        self.classes = classes
        self.__call = 0

        super().__init__()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 1 and trainer.is_global_zero and (self.__call % self.frequency == 0):
            n = self.n_images
            x = batch["image"][:n].float()
            y = batch["mask"][:n]
            roi = batch["roi"][:n].unsqueeze(1)
            prob = pl_module.get_prob(outputs, roi)
            pred = pl_module.get_pred(prob)

            pred = onehot_to_label(pred)
            y = onehot_to_label(y)
            columns = ["image"]
            class_labels = {i: name for i, name in enumerate(pl_module.classes_legend)}

            data = [
                [
                    wandb.Image(
                        x_i,
                        masks={
                            "Prediction": {
                                "mask_data": p_i.cpu().numpy(),
                                "class_labels": class_labels,
                            },
                            "Groundtruth": {
                                "mask_data": y_i.cpu().numpy(),
                                "class_labels": class_labels,
                            },
                        },
                    )
                ]
                for x_i, y_i, p_i in list(zip(x, y, pred))
            ]
            self.wandb_logger.log_table(data=data, key=f"Validation Batch {batch_idx}", columns=columns)
        self.__call += 1


class DropoutSchedulerCallback(Callback):
    def __init__(self, warm_up=0.5, final_dropout=0.2):
        self.warm_up = warm_up
        self.dropout = final_dropout

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        t_max = trainer.max_epochs * self.warm_up
        t_current = trainer.current_epoch
        model = trainer.model

        if t_current >= t_max:
            p = self.dropout
        else:
            p = max(
                0,
                self.dropout * (1 - math.cos(math.pi * t_current / t_max)) / 2,
            )

        for module in model.modules():
            if isinstance(module, nn.Dropout2d):
                module.p = p
        if trainer.logger is not None:
            trainer.logger.log_metrics({"dropout": p}, step=trainer.global_step)


def get_callbacks(config, wandb_logger=None, classes=ALL_CLASSES, run_name=None):
    project_name = config["logger"]["project"]
    callbacks = []
    if wandb_logger:
        log_pred_callback = LogPredictionSamplesCallback(wandb_logger, n_images=8, classes=classes)
        callbacks.append(log_pred_callback)

    lr_monitor = LearningRateMonitor()
    callbacks.append(lr_monitor)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_mIoU",
        mode="max",
        save_last=True,
        auto_insert_metric_name=True,
        save_top_k=1,
        dirpath=os.path.join("checkpoints", project_name, os.environ["WANDB_RUN_NAME"]),
    )

    callbacks.append(checkpoint_callback)
    callbacks.append(DropoutSchedulerCallback())

    if config["trainer"].pop("early_stopping", False):
        early_stopping = EarlyStopping(monitor="val_mIoU", mode="max", patience=25)
        callbacks.append(early_stopping)

    return callbacks
