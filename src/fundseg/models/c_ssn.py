import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torchseg
from adptSeg.adaptation.const import batch_dataset_to_integer
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fundseg.models.base_model import BaseModel
from fundseg.models.utils.probs_style import ReshapedDistribution, StochasticSegmentationNetworkLossMCIntegral, TaskMode


class CSNNStyleModel(BaseModel):
    def __init__(
        self,
        in_chans=3,
        n_classes=5,
        arch="unet",
        encoder="resnet34",
        pretrained=True,
        optimizer="rmsprop",
        num_styles=5,
        epsilon=1e-5,
        rank: int = 10,
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

        self.model.segmentation_head = nn.Identity()
        self.number_styles = num_styles
        self.mean_l = nn.Conv2d(16, n_classes, kernel_size=1)
        self.log_cov_diag_l = nn.Conv2d(16, n_classes, kernel_size=1)
        self.cov_factor_l = nn.Conv2d(16, n_classes * rank, kernel_size=1)
        self.style_encoder = nn.Conv2d(5, 16, kernel_size=1)
        self.end_encoder = nn.Conv2d(16 * 2, 16, kernel_size=1)
        self.optim = optimizer
        self.epsilon = epsilon
        self.rank = rank
        task_mode = TaskMode.MULTICLASS if n_classes > 2 else TaskMode.BINARY
        self.loss_function = StochasticSegmentationNetworkLossMCIntegral(num_mc_samples=20, mode=task_mode)
        self.distribution = None

    @property
    def model_name(self):
        return f"{self.arch}-{self.encoder_name}"

    def initialize(self):
        self.model.initialize()

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        Tile means Fliese in Deutsch
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            self.device
        )
        return torch.index_select(a, dim, order_index)

    def forward_train(self, x, style):
        logits = self.forward(x, style)
        b = logits.shape[0]
        # tensor size num_classesxHxW
        event_shape = (self.n_classes,) + logits.shape[2:]
        mean = self.mean_l(logits)
        b, c, h, w = mean.shape
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((b, -1))
        cov_diag = cov_diag.view((b, -1))

        cov_factor = self.cov_factor_l(logits)
        cov_factor = cov_factor.view((b, self.rank, self.n_classes, -1))
        cov_factor = cov_factor.flatten(2, 3)
        cov_factor = cov_factor.transpose(1, 2)

        try:
            base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
        except:
            print("Covariance became not invertible using independent normals for this batch!")
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        self.distribution = ReshapedDistribution(
            base_distribution=base_distribution,
            new_event_shape=event_shape,
            validate_args=False,
        )
        return mean.view((b, self.n_classes, -1))

    def forward(self, x, style):
        logits = self.model(x)
        b, c, h, w = logits.shape
        style = torch.nn.functional.one_hot(style, num_classes=self.number_styles)

        style = torch.unsqueeze(style, 2)  # give the tensor another dimension
        style = self.tile(style, 2, h)
        style = torch.unsqueeze(style, 3)  # give the tensor another dimension
        style = self.tile(style, 3, w)
        style = style.float()
        style = self.style_encoder(style)
        logits = torch.cat((logits, style), dim=1)
        logits = self.end_encoder(logits)
        return logits

    def get_loss(self, logits, mask):
        if self.distribution is None:
            return torch.tensor(0.0)
        else:
            return self.loss_function(logits, mask, self.distribution)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        mask = batch["mask"].long()
        style = batch["tag"]
        style = batch_dataset_to_integer(style)
        style = torch.tensor(style, device=self.device)

        logits = self.forward_train(x, style)
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

    def get_prob(self, logits, roi=None):
        if roi is not None:
            if roi.ndim == 4:
                roi.squeeze_(1)
            for k in range(1, self.n_classes):
                logits[:, k][roi < 1] = -torch.inf
        return torch.softmax(logits, 1)

    def get_pred(self, prob):
        return torch.argmax(prob, 1)

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        b, c, h, w = x.shape
        mask = batch["mask"].long()
        style = batch["tag"]
        style = batch_dataset_to_integer(style)
        style = torch.tensor(style, device=self.device)
        roi = batch["roi"].unsqueeze(1)
        logits = self.forward_train(x, style)
        logits = logits.view((b, self.n_classes, h, w))
        output = self.get_prob(logits, roi).to(self.device)
        self.valid_metrics.update(output, mask)

        return output

    @torch.inference_mode()
    def inference_step(self, batch):
        self.eval()
        batch = self.transfer_batch_to_device(batch, self.device, 0)
        x = batch["image"]
        b, c, h, w = x.shape
        roi = batch["roi"].unsqueeze(1)
        style = batch["tag"]
        style = batch_dataset_to_integer(style)
        style = torch.tensor(style, device=self.device)
        logits = self.forward_train(x, style)
        logits = logits.view((b, self.n_classes, h, w))
        output = self.get_prob(logits, roi)
        return output

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["image"]
        b, c, h, w = x.shape
        roi = batch["roi"].unsqueeze(1)
        y = batch["mask"].long()
        style = batch["tag"]
        style = batch_dataset_to_integer(style)
        style = torch.tensor(style, device=self.device)

        logits = self.forward_train(x, style)
        logits = logits.view((b, self.n_classes, h, w))

        prob = self.get_prob(logits, roi)
        self.test_metrics[dataloader_idx].update(prob, y)

    def configure_optimizers(self):
        params = self.parameters()
        if self.optim == "adam":
            optimizer = Adam(params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-8)
        elif self.optim == "adamw":
            optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-8)
        elif self.optim == "sgd":
            optimizer = SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optim == "rmsprop":
            optimizer = torch.optim.RMSprop(params, lr=self.lr, weight_decay=self.weight_decay, alpha=0.9, momentum=0.6)
        else:
            raise ValueError(f"Invalid optimizer {self.optim}")

        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-6)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


if __name__ == "__main__":
    model = CSNNStyleModel(test_dataset_id=["MESSIDOR", "IDRID"])
    print(model)
