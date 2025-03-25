import torch.nn as nn
import torchseg
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fundseg.models.base_model import BaseModel


class SMPModel(BaseModel):
    def __init__(
        self,
        in_chans=3,
        n_classes=5,
        arch="unet",
        encoder="resnet34",
        pretrained=True,
        optimizer="adamw",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        encoder_weights = "imagenet" if pretrained else None
        self.arch = arch
        self.encoder_name = encoder
        self.model = torchseg.create_model(
            arch,
            encoder,
            in_channels=in_chans,
            classes=n_classes,
            encoder_weights=encoder_weights,
        )
        self.optim = optimizer
        self.prepare_dropout(0.2)

    @property
    def model_name(self):
        return f"{self.arch}-{self.encoder_name}"

    def initialize(self):
        self.model.initialize()

    def forward(self, x):
        return self.model(x)

    def get_loss(self, logits, mask):
        return self.dice_loss(logits, mask)

    def configure_optimizers(self):
        params = self.model.parameters()
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

    def prepare_dropout(self, initial_value=0.0):
        for k, v in list(self.model.named_modules()):
            if "drop" in k.split("."):
                parent_model = self.model
                for model_name in k.split(".")[:-1]:
                    parent_model = getattr(parent_model, model_name)
                setattr(parent_model, "drop", nn.Dropout2d(p=initial_value))
