"""
Adaptation from "Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses"
by Rupprecht et al.
"""

from copy import deepcopy

import torch
import torch.nn as nn
from monai.losses import FocalLoss
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fundseg.models.smp_model import SMPModel


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class VoronoiCNN(SMPModel):
    def __init__(self, *args, n_heads=5, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_heads = n_heads
        self.eps = 0.05

        seg_head = self.model.segmentation_head
        self.multiples_segmentation_head = nn.ModuleList([deepcopy(seg_head) for _ in range(5)])
        self.multiples_segmentation_head.apply(initialize_head)  # Initialize weights differently for each head
        self.model.segmentation_head = nn.Identity()
        self.focal_loss = FocalLoss(reduction="none", to_onehot_y=True, use_softmax=True)

    def forward(self, x):
        prelogits = self.model(x)
        logits = [head(prelogits) for head in self.multiples_segmentation_head]
        return logits

    def get_loss(self, logits, mask):
        # We compute the loss for each head
        losses = []
        for head_logits in logits:
            loss = self.dice_loss(head_logits, mask.unsqueeze(1))

            losses.append(loss)
        losses = torch.stack(losses)
        losses_shape = losses.shape
        weights = torch.ones_like(losses) * self.eps / (self.n_heads - 1)
        min_idx = torch.argmin(losses, dim=0, keepdim=True)
        weights.scatter_(0, min_idx, 1 - self.eps)
        weights = weights.view(losses_shape)
        loss = (weights * losses).sum(dim=0)
        if loss.ndim > 1:
            loss = loss.sum(dim=(1)).mean()
        return loss

    def get_prob(self, logits, roi=None):
        preds = torch.stack(logits)
        preds = torch.mean(preds, dim=0)
        return super().get_prob(preds, roi)

    def configure_optimizers(self):
        params = list(self.model.parameters()) + list(self.multiples_segmentation_head.parameters())

        if self.optim == "adam":
            optimizer = Adam(params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-8)
        elif self.optim == "adamw":
            optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-8)
        elif self.optim == "sgd":
            optimizer = SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer {self.optim}")

        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-6)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


if __name__ == "__main__":
    from pytorch_lightning import seed_everything
