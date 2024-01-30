import torch.nn as nn
from fundseg.models.smp_model import SMPModel


class ModelEncoder(nn.Module):
    def __init__(self, model:SMPModel, encoding_position=-1):
        super().__init__()
        self.model = model.model
        self.model.eval()
        self.model.requires_grad_(False)
        self.encoding_position = encoding_position
        self.out_chans = self.model.encoder.out_channels[encoding_position]
    def forward(self, x):
        self.eval()
        features = self.model.encoder(x)
        return features[self.encoding_position]