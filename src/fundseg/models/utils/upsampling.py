import math

import torch.nn as nn
import torch.nn.functional as F


class Conv1x1Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor=2, mode="bilinear"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.mode = mode
        self.upsample_factor = upsample_factor

    def forward(self, x):
        x = F.interpolate(
            self.conv(x), scale_factor=self.upsample_factor, mode=self.mode
        )
        return x


def upsample_to_next_power_of_two(x):
    _, _, h, w = x.shape
    next_power_2 = 2 ** int(math.log2(h + 1))
    return F.interpolate(x, size=(next_power_2, next_power_2), mode="bilinear")
