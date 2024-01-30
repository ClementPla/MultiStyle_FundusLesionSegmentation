import torch
import torch.nn as nn


class SequentialConvs(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        activation=nn.ReLU,
        n_convs=2,
        batch_norm=False,
        groups=1,
    ):
        super().__init__()

        convs = []
        for i in range(n_convs):
            if batch_norm:
                norm = nn.BatchNorm2d(out_channels)
            else:
                norm = nn.Identity()
            convs += [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                ),
                norm,
                activation(),
            ]
            in_channels = out_channels

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class DynamicAttentionBlock(nn.Module):
    def __init__(self, input_features:int):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(input_features, input_features), nn.Sigmoid())

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(
                input_features, input_features, kernel_size=1, stride=1, padding=0
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        channel_wise_attn = self.fc(x.mean(dim=(2, 3))).unsqueeze(2).unsqueeze(3) * x
        point_wise_attn = self.conv1x1(x) * x
        spatial_wise_attn = torch.sigmoid(x.mean(dim=1, keepdim=True)) * x

        return channel_wise_attn + point_wise_attn + spatial_wise_attn


class ProgressiveFeatureFusion(nn.Module):
    def __init__(
        self, current_features, previous_features=None, next_features=None
    ) -> None:
        super().__init__()

        if next_features:
            self.next_op = nn.Sequential(
                nn.Conv2d(
                    next_features, current_features, kernel_size=1, stride=1, padding=0
                ),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            )

        if previous_features:
            self.previous_op = nn.Sequential(
                nn.Conv2d(
                    previous_features,
                    previous_features,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    groups=previous_features,
                ),
                nn.Conv2d(
                    previous_features,
                    current_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )

        concat_features = (
            current_features * 3
            if next_features and previous_features
            else current_features * 2
        )
        self.dab = DynamicAttentionBlock(concat_features)

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                concat_features, current_features, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(),
        )

    def forward(self, current_features, previous_features=None, next_features=None):
        if previous_features is not None:
            f1 = current_features * self.previous_op(previous_features)
            if next_features is None:
                f1 = torch.cat(
                    [f1, current_features], dim=1
                )  # This differs from the paper, but corresponds to the code in the official repo
                return self.last_conv(self.dab(f1))

        if next_features is not None:
            f2 = current_features * self.next_op(next_features)
            if previous_features is None:
                f2 = torch.cat(
                    [f2, current_features], dim=1
                )  # This differs from the paper, but corresponds to the code in the official repo
                return self.last_conv(self.dab(f2))

        return self.last_conv(self.dab(torch.cat([f1, f2, current_features], dim=1)))
