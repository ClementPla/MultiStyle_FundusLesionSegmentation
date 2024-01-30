import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from fundseg.models.base_model import BaseModel
from fundseg.models.utils.blocks import SequentialConvs
from fundseg.models.utils.losses import MultiLabelSoftBinaryCrossEntropy


class LSeg(BaseModel):
    def __init__(
        self,
        in_chans=3,
        n_classes=5,
        encoder="resnet34",
        use_batchnorm=False,
        scales=4,
        pretrained=True,
        *args,
        **kwargs,
    ):
        super(LSeg, self).__init__(*args, **kwargs)
        self.arch = "LSeg"
        self.encoder_name = encoder
        self.encoder = timm.create_model(
            encoder,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=np.arange(scales),
        )

        features_infos = self.encoder.feature_info.channels()
        self.use_batchnorm = use_batchnorm
        self.n_scales = scales
        self.n_classes = n_classes
        self.decoder = nn.ModuleList()
        self.deep_supervised = True
        for f_in in features_infos:
            # This is the side feature extraction
            self.decoder.append(
                SequentialConvs(f_in, n_classes, kernel_size=1, 
                                padding=0, bias=True, 
                                batch_norm=self.use_batchnorm,
                                n_convs=1),
            )

        self.segmentation_head = nn.ModuleList([SequentialConvs(in_channels=self.n_scales,
                            out_channels=1, 
                            kernel_size=1, padding=0, 
                            activation=nn.Identity, n_convs=1) 
                                                for i in range(self.n_classes)])
        
        # We do not obtain convergence with this loss. It is likely because it is built 
        # for multilabel and not multiclass    
        # self.loss = MultiLabelSoftBinaryCrossEntropy(
        #     weighted=True, mcb=True, from_logits=False, hp_lambda=15
        # )
        
        self.loss = self.dice_loss
        self.initialize()

    @property
    def model_name(self):
        return f"LSeg-{self.encoder_name}"

    def forward(self, x):
        features = self.encoder(x)
        out = []
        for s, f in enumerate(features):
            o = self.decoder[s](f)
            o = F.interpolate(input=o, size=x.shape[2:], mode="bilinear")
            out.append(o)
        concatChannels = []
        for c in range(self.n_classes):
            concatChannel = torch.cat(
                [torch.unsqueeze(tens[:, c], 1) for tens in out], 1
            )
            concatChannels.append(self.segmentation_head[c](concatChannel))
        fuse = torch.cat(concatChannels, 1)
        if self.training and self.deep_supervised:
            return fuse, *out
        else:
            return fuse

    def get_loss(self, output, target):
        if not isinstance(output, tuple):
            output = (output,)
        if isinstance(self.loss, MultiLabelSoftBinaryCrossEntropy):
            return self.weightedMultiLabelLoss(output, target)
        else:
            loss = 0
            for o in output:
                loss += self.loss(o, target)
            
            return loss
                
            
    def weightedMultiLabelLoss(self, output, target):
        loss = 0
        target = F.one_hot(target.squeeze(1), num_classes=self.n_classes)
        target = target.permute(0, 3, 1, 2)
        for o in output:
            # multiclass -> multilabel
            o = torch.softmax(o, 1)
            loss = loss + self.loss(
                o[:, 1:], target[:, 1:]
            )  # We skip the backgroud class
        return loss

    def get_prob(self, logits, roi=None):
        if isinstance(logits, tuple):
            logits = logits[0]
        return super().get_prob(logits, roi)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-8, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=self.trainer.estimated_stepping_batches, power=0.1
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
