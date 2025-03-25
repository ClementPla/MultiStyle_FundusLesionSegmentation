from typing import Any

import torch
import torch.nn as nn
import torchmetrics
from huggingface_hub import PyTorchModelHubMixin
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import Metric

import wandb
from adptSeg.adaptation.const import _all_datasets, batch_dataset_to_integer

seed_everything(1234, workers=True)


class Probe(nn.Module):
    def __init__(self, n_channels, n_classes, as_regression=False) -> None:
        super().__init__()

        self.pre_linear = nn.Identity()
        if as_regression:
            self.probe = nn.Linear(n_channels, 1)
        else:
            self.probe = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        x = self.pre_linear(x)
        x = x.mean(dim=(2, 3))
        return self.probe(x)


class MyPredictions(Metric):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        super().compute()
        if isinstance(self.preds, list):
            preds = torch.cat(self.preds)
            targets = torch.cat(self.targets)
        else:
            preds = self.preds
            targets = self.targets
        return preds, targets


class WeightedCategoricalMSE(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, x, y):
        if self.weights is not None:
            weights = self.weights.to(x.device)
            weights = weights[y.long().squeeze()]
            return torch.mean(weights * (x - y.float()) ** 2)
        else:
            return torch.mean((x - y.float()) ** 2)


class ProbeModule(LightningModule, PyTorchModelHubMixin):
    def __init__(
        self, featureExtractor, lr=0.005, n_classes=5, weight_decay=0.00001, weights=None, as_regression=False
    ) -> None:
        super().__init__()

        self.featureExtractor = featureExtractor
        self.probe = Probe(featureExtractor.out_chans, n_classes, as_regression=as_regression)
        self.as_regression = as_regression
        self.n_classes = n_classes

        if self.as_regression:
            self.criterion = WeightedCategoricalMSE(weights=weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.lr = lr
        self.weight_decay = weight_decay
        kwargs = dict(task="multiclass", num_classes=n_classes)
        self.validation_metrics = torchmetrics.MetricCollection(
            torchmetrics.Accuracy(**kwargs),
            torchmetrics.Recall(**kwargs),
            torchmetrics.Precision(**kwargs),
            torchmetrics.Specificity(**kwargs),
        )
        self.myPreds = MyPredictions()

    def forward(self, batch):
        x = self.transfer_batch_to_device(batch, self.device, 0)
        return self.probe(self.featureExtractor(x))

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x = batch["image"]
        y = batch["tag"]
        y = torch.Tensor(batch_dataset_to_integer(y)).to(self.device).long()

        with torch.no_grad():
            self.featureExtractor.eval()
            logits = self.featureExtractor(x)
        logits = self.probe(logits)
        loss = self.criterion(logits, y)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["tag"]
        y = torch.Tensor(batch_dataset_to_integer(y)).to(self.device).long()
        logits = self.featureExtractor(x)
        logits = self.probe(logits)
        preds = self.get_preds(logits)
        self.validation_metrics.update(preds, target=y)
        self.log_dict(self.validation_metrics, on_epoch=True, sync_dist=True, batch_size=x.size(0))

    def get_preds(self, logits):
        if self.as_regression:
            return torch.round(logits).clamp(0, self.n_classes - 1).squeeze(1)
        else:
            return torch.argmax(logits, dim=1)

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["tag"]
        y = torch.Tensor(batch_dataset_to_integer(y)).to(self.device).long()
        logits = self.featureExtractor(x)
        logits = self.probe(logits)
        preds = self.get_preds(logits)
        self.myPreds.update(preds, target=y)

    def on_test_end(self) -> None:
        preds, targets = self.myPreds.compute()
        if self.trainer.is_global_zero:
            wandb.log(
                {
                    "conf_mat": wandb.plot.confusion_matrix(
                        preds=preds.cpu().numpy(),
                        y_true=targets.cpu().numpy(),
                        class_names=[k.name for k in _all_datasets],
                        title="Confusion Matrix",
                    )
                }
            )

    def configure_optimizers(self):
        params = self.probe.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
