"""
Implementation of the LesionNet model as proposed by the paper "Learn to Segment Retinal Lesions and Beyond" by Wei et al.
Include the implementation of the Dual Loss as described in the paper.
"""
import math
from typing import Literal

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from fundseg.models.base_model import BaseModel
from fundseg.models.utils.blocks import SequentialConvs
from fundseg.models.utils.upsampling import Conv1x1Upsampling


class LesionNet(BaseModel):
    def __init__(
        self,
        variant: Literal[2, 4, 16, 32],
        pretrained: bool = True,
        encoder="inception_v3",
        use_batchnorm=False,
        n_classes=5,
    ) -> None:
        super().__init__(n_classes=n_classes)
        self.arch = "LesionNet"
        self.use_batchnorm = use_batchnorm
        self.encoder_name = encoder
        self.encoder = timm.create_model(
            encoder, pretrained=pretrained, features_only=True
        )
        self.variant = variant

        features_infos = self.encoder.feature_info.channels()
        self.n_scales = 6 - int(math.log2(self.variant))  # Maximum scales is 6
        decoder_layers = []
        features_from_previous_scale = 0
        for i in range(self.n_scales):
            features_from_skipped_connection = features_infos[-i - 1]
            input_features = (
                features_from_previous_scale + features_from_skipped_connection
            )
            layer = [
                SequentialConvs(
                    input_features,
                    input_features,
                    n_convs=2,
                    batch_norm=self.use_batchnorm,
                )
            ]
            if i < self.n_scales - 1:
                layer.append(Conv1x1Upsampling(input_features, input_features // 2, 2))
                features_from_previous_scale = input_features // 2
            else:
                features_from_previous_scale = input_features
            decoder_layers.append(nn.Sequential(*layer))

        self.decoder = nn.ModuleList(decoder_layers)

        self.segmentation_head = nn.Conv2d(
            features_from_previous_scale,
            n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        print(
            f"LesionNet with variant {variant} and {n_classes} classes initialized. \
            Building decoder with {len(self.decoder)} upsampling layers."
        )
        self.lambda_loss = 0.8
        self.lr = 0.001
        self.decay_happened = False
        self.initialize()
        
    @property
    def model_name(self):
        return f"LesionNet {self.variant}-{self.encoder_name}"

    def forward(self, x):
        features = self.encoder(x)
        prev_decoder_f = None
        for i, decoder_layer in enumerate(self.decoder):
            encoder_f = features[-i - 1]
            if prev_decoder_f is not None:
                encoder_f = F.interpolate(
                    encoder_f,
                    size=prev_decoder_f.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                encoder_f = torch.cat([encoder_f, prev_decoder_f], dim=1)
            prev_decoder_f = decoder_layer(encoder_f)

        output = self.segmentation_head(prev_decoder_f)
        return F.interpolate(
            output, size=x.shape[2:], mode="bilinear", align_corners=False
        )

    def cls_dice(self, output, target):
        output = F.softmax(output, dim=1)[:, 1:]  # remove background class
        target = F.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2)
        target = target[:, 1:]
        output = torch.amax(output, dim=(2, 3))
        target = torch.amax(target, dim=(2, 3))
        numerator = 2 * torch.sum(output * target)
        denominator = torch.sum(output**2) + torch.sum(target**2)
        return 1 - (numerator / denominator)

    def get_loss(self, logits, mask):
        """
        Compute the Dual Loss as described in the paper.
        """
        loss = self.dice_loss(logits, mask)
        if not self.decay_happened:
            lightning_optimizer = self.optimizers()  # self = your model
            for param_group in lightning_optimizer.optimizer.param_groups:
                self.decay_happened = param_group["lr"] != self.lr
            if self.decay_happened:
                print("Switching to dual class loss")

        if self.decay_happened:
            loss = self.lambda_loss * loss + (1 - self.lambda_loss) * self.cls_dice(
                logits, mask
            )
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )
        # The original paper uses SGD. We got much more success with AdamW
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr,
                                    weight_decay=0.0001)


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=4, verbose=True
        )
        return [optimizer], [
            {
                "scheduler": scheduler,
                "frequency": self.trainer.check_val_every_n_epoch,
                "monitor": "val_loss",
            }
        ]


if __name__ == "__main__":
    model = LesionNet(16)

    x = torch.randn(1, 3, 512, 512)
    print(model(x).shape)
