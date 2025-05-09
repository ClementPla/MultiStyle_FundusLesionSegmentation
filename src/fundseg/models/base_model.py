from abc import abstractmethod
from typing import Any, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchseg.base.initialization as smp_init
from huggingface_hub import PyTorchModelHubMixin
from kornia.morphology import gradient
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchseg.losses import DiceLoss
from torchvision.utils import draw_segmentation_masks, make_grid

from fundseg.data.utils import ALL_CLASSES
from fundseg.models.utils.metric import AUCPrecisionRecallCurve


class BaseModel(LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        n_classes=5,
        classes=ALL_CLASSES,
        lr=0.001,
        log_dice=False,
        smooth_dice=0.225,
        test_dataset_id: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.task = "multiclass"
        self.n_classes = n_classes
        self.dice_loss = DiceLoss(mode=self.task, log_loss=log_dice, smooth=smooth_dice)
        self.classes = classes
        self.lr = lr
        self.weight_decay = 0.00001

        self.valid_metrics = torchmetrics.MetricCollection(
            {
                "val_IoU": torchmetrics.JaccardIndex(
                    task=self.task,
                    num_classes=self.n_classes,
                    num_labels=self.n_classes,
                    average=None,
                ),
                "val_mIoU": torchmetrics.JaccardIndex(
                    task=self.task,
                    num_classes=self.n_classes,
                    num_labels=self.n_classes,
                    average="macro",
                ),
            }
        )
        self.single_test_metrics = torchmetrics.MetricCollection(
            {
                "AUC Precision Recall": AUCPrecisionRecallCurve(
                    task=self.task,
                    num_classes=self.n_classes,
                    num_labels=self.n_classes,
                    validate_args=False,
                    thresholds=11,
                ),
                "mIoU": torchmetrics.JaccardIndex(
                    task=self.task,
                    num_labels=self.n_classes,
                    num_classes=self.n_classes,
                    average="macro",
                ),
            },
        )
        try:
            self.test_dataloader_ids = [t.replace("_test", "").replace("_split_1", "") for t in test_dataset_id]
            test_metrics = []

            for test_id in self.test_dataloader_ids:
                print(f"Test dataset: {test_id}")
                test_metrics.append(
                    torchmetrics.MetricCollection(
                        {
                            "AUC Precision Recall": AUCPrecisionRecallCurve(
                                task=self.task,
                                num_classes=self.n_classes,
                                num_labels=self.n_classes,
                                validate_args=False,
                                thresholds=11,
                            ),
                            "mIoU": torchmetrics.JaccardIndex(
                                task=self.task,
                                num_labels=self.n_classes,
                                num_classes=self.n_classes,
                                average="macro",
                            ),
                        },
                        postfix=f" - {test_id}",
                    )
                )
            self.test_metrics = nn.ModuleList(test_metrics)

            self.current_test_dataloader_idx = 0
        except TypeError:
            print("No test dataset provided")
        self.save_hyperparameters()

    def initialize(self):
        smp_init.initialize_decoder(self.model.decoder)
        smp_init.initialize_head(self.model.segmentation_head)

    @abstractmethod
    def get_loss(self, logits, mask):
        pass

    @property
    def classes_label(self):
        return ["Background", *self.classes]

    @property
    def classes_legend(self):
        return self.classes_label

    @property
    @abstractmethod
    def model_name(self):
        pass

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x = batch["image"]
        mask = batch["mask"].long()
        mask = torch.clamp(mask, 0, self.n_classes - 1)

        logits = self(x)

        loss = self.get_loss(logits, mask)

        loss = torch.nan_to_num(loss)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        mask = batch["mask"].long()
        mask = torch.clamp(mask, 0, self.n_classes - 1)
        roi = batch["roi"].unsqueeze(1)
        logits = self(x)
        loss = self.get_loss(logits, mask)
        self.log("val_loss", loss.to(self.device), on_epoch=True, on_step=False, sync_dist=True)
        output = self.get_prob(logits, roi)
        self.valid_metrics.update(output, mask)
        return output

    def get_prob(self, logits, roi=None, temperature=1.0):
        if roi is not None:
            if roi.ndim == 4:
                roi.squeeze_(1)
            for k in range(1, self.n_classes):
                logits[:, k][roi < 1] = -torch.inf
        return torch.softmax(logits * temperature, 1)

    def get_pred(self, prob):
        return torch.argmax(prob, 1)

    @torch.inference_mode()
    def inference_step(self, batch, temperature=1):
        self.eval()
        batch = self.transfer_batch_to_device(batch, self.device, 0)
        x = batch["image"]
        roi = batch["roi"].unsqueeze(1)
        logits = self(x)
        output = self.get_prob(logits, roi, temperature)
        return output

    def on_validation_epoch_end(self):
        score = self.setup_scores(self.valid_metrics)
        self.log_dict(score, sync_dist=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["image"]
        roi = batch["roi"].unsqueeze(1)
        y = batch["mask"].long()
        y = torch.clamp(y, 0, self.n_classes - 1)
        output = self(x)
        prob = self.get_prob(output, roi)
        self.test_metrics[dataloader_idx].update(prob, y)

    def on_test_batch_end(
        self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if dataloader_idx != self.current_test_dataloader_idx:
            score = self.setup_scores(self.test_metrics[self.current_test_dataloader_idx])
            self.log_dict(score, add_dataloader_idx=False)
            self.current_test_dataloader_idx = dataloader_idx

    def on_test_epoch_end(self):
        score = self.setup_scores(self.test_metrics[-1])
        self.log_dict(score, add_dataloader_idx=False, sync_dist=True)

    def setup_scores(self, metric):
        score = metric.compute()
        new_score = {}
        for k, v in score.items():
            if v.ndim > 0:
                classes = self.classes_label
                for i in range(len(classes)):
                    c_legend = classes[i]
                    if c_legend == "Background":
                        continue
                    score_name = f"{k} {classes[i]}"
                    new_score[score_name] = torch.nan_to_num(v[i]).to(self.device)
            else:
                new_score[k] = torch.nan_to_num(v).to(self.device)
        return new_score

    def get_grid_with_predicted_mask(
        self, batch, alpha=0.8, colors=None, ncol=8, padding=2, border_alpha=1.0, kernel_size=5
    ):
        prob = self.inference_step(batch)
        pred = self.get_pred(prob).unsqueeze(1)
        image = batch["image"].to(self.device)
        grid_image = make_grid(image, nrow=ncol, padding=padding, normalize=True, scale_each=True) * 255
        grid_pred = make_grid(pred, nrow=ncol, padding=padding)[0]
        grid_pred = F.one_hot(grid_pred, num_classes=self.n_classes).permute((2, 0, 1))

        kernel = torch.ones(kernel_size, kernel_size, device=self.device)
        border = gradient(grid_pred.unsqueeze(0), kernel).squeeze(0)
        border[0] = 0
        grid_pred[0] = 0
        draw = draw_segmentation_masks(
            grid_image.to(torch.uint8).cpu(),
            grid_pred.to(torch.bool).cpu(),
            alpha=alpha,
            colors=colors,
        )
        draw = draw_segmentation_masks(draw, border.to(torch.bool).cpu(), alpha=border_alpha, colors=colors)

        return draw

    def get_grid_with_gt_mask(self, batch, alpha=0.8, colors=None, ncol=8, padding=2, border_alpha=1.0, kernel_size=5):
        gt = batch["mask"].unsqueeze(1).to(self.device).long()
        gt = torch.clamp(gt, 0, self.n_classes - 1)
        image = batch["image"].to(self.device)
        grid_image = make_grid(image, nrow=ncol, padding=padding, normalize=True, scale_each=True) * 255
        grid_gt = make_grid(gt, nrow=ncol, padding=padding)[0]
        grid_gt = F.one_hot(grid_gt, num_classes=self.n_classes).permute((2, 0, 1))

        kernel = torch.ones(kernel_size, kernel_size, device=self.device)
        border = gradient(grid_gt.unsqueeze(0), kernel).squeeze(0)
        border[0] = 0
        grid_gt[0] = 0
        draw = draw_segmentation_masks(
            grid_image.to(torch.uint8).cpu(),
            grid_gt.to(torch.bool).cpu(),
            alpha=alpha,
            colors=colors,
        )
        draw = draw_segmentation_masks(draw, border.to(torch.bool).cpu(), alpha=border_alpha, colors=colors)
        return draw
