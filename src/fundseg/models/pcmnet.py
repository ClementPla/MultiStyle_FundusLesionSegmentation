"""
Implementation of PCMNet as proposed by the paper "Progressive Multiscale Consistent Network 
for Multiclass Fundus Lesion Segmentation" by He et al.
Mostly a pytorch conversion of:
https://github.com/NKUhealong/PMCNet/blob/main/models.py
"""
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from fundseg.models.base_model import BaseModel
from fundseg.models.utils.blocks import ProgressiveFeatureFusion, SequentialConvs
from fundseg.models.utils.upsampling import Conv1x1Upsampling


class ProgressiveMultiscaleConsistentNetwork(BaseModel):
    def __init__(
        self,
        n_classes=5,
        encoder="efficientnet_b0",
        pretrained=True,
        n_scale=4,
        use_batchnorm=False,
        *args,
        **kwargs,
    ):
        super().__init__(n_classes=n_classes, *args, **kwargs)

        self.arch = "PMCNet"
        self.use_batchnorm = use_batchnorm
        self.encoder_name = encoder
        self.loss = nn.CrossEntropyLoss()
        self.encoder = timm.create_model(
            encoder, pretrained=pretrained, features_only=True,
            out_indices = np.arange(n_scale),
        )
        features_infos = self.encoder.feature_info.channels()
        encoder_ops = []
        decoder_ops = []
        self.n_scales = n_scale
        assert (
            len(features_infos) >= n_scale
        ), f"Number of scales ({n_scale}) must be smaller or equal to the number of features ({len(features_infos)})"
        
        features_infos = features_infos[:self.n_scales]
        # Encoder additional operations from PMCNet
        for i in range(n_scale):
            f = features_infos[-n_scale + i]
            if i == 0:
                encoder_ops.append(
                    ProgressiveFeatureFusion(f, None, features_infos[-n_scale + i + 1])
                )
            elif i == (n_scale - 1):
                encoder_ops.append(
                    ProgressiveFeatureFusion(f, features_infos[-n_scale + i - 1], None)
                )
            else:
                encoder_ops.append(
                    ProgressiveFeatureFusion(
                        f,
                        features_infos[-n_scale + i - 1],
                        features_infos[-n_scale + i + 1],
                    )
                )

        self.encoder_ops = nn.ModuleList(encoder_ops)
        # Decoder traditional blocks from UNet
        features_from_previous_scale = 0
        decoder_ops = []
        for i in range(n_scale):
            features_from_skipped_connection = features_infos[-i - 1]
            input_features = (
                features_from_previous_scale + features_from_skipped_connection
            )

            decoder = [
                SequentialConvs(input_features, input_features, n_convs=2, 
                                batch_norm=use_batchnorm),
            ]
            if i < (n_scale - 1):
                decoder.append(Conv1x1Upsampling(input_features, input_features, 2))

            decoder_ops.append(nn.Sequential(*decoder))
            features_from_previous_scale = input_features

        self.decoder = nn.ModuleList(decoder_ops)

        self.segmentation_head = nn.Conv2d(features_from_previous_scale, n_classes, 1)
        self.initialize()
        
    @property
    def model_name(self):
        return f"PCMNet-{self.encoder_name}"

    def forward(self, x):
        features = self.encoder(x)[:self.n_scales]
        
        updated_features = []
        for i, f in enumerate(features):
            if i == 0:
                prev_features = None
            else:
                prev_features = features[i - 1]
            if i == (self.n_scales - 1):
                next_features = None
            else:
                next_features = features[i + 1]
            updated_features.append(
                self.encoder_ops[i](
                    f, next_features=next_features, previous_features=prev_features
                )
            )

        prev_decoder_f = None
        for i, decoder_layer in enumerate(self.decoder):
            encoder_f = updated_features[-i - 1]
            if prev_decoder_f is not None:
                encoder_f = torch.cat([encoder_f, prev_decoder_f], dim=1)
            prev_decoder_f = decoder_layer(encoder_f)
        output = self.segmentation_head(prev_decoder_f)
        return F.interpolate(
            output, size=x.shape[2:], mode="bilinear", align_corners=False
        )

    def get_loss(self, output, target):
        return self.loss(output, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=self.trainer.max_epochs, power=0.9
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
