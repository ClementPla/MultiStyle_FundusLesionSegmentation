import math
from typing import Tuple

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F


class StochasticSegmentationNetworkLossMCIntegral(nn.Module):
    def __init__(self, num_mc_samples: int = 1):
        super().__init__()
        self.num_mc_samples = num_mc_samples

    @staticmethod
    def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean

    def forward(self, logits, target, distribution, **kwargs):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        logit_sample = self.fixed_re_parametrization_trick(
            distribution, self.num_mc_samples
        )
        target = target.unsqueeze(1)
        target = target.expand((self.num_mc_samples,) + target.shape)
        flat_size = self.num_mc_samples * batch_size
        logit_sample = logit_sample.view((flat_size, num_classes, -1))
        target = target.reshape((flat_size, -1))
        target = target.unsqueeze(1)
        log_prob = -F.binary_cross_entropy_with_logits(
            logit_sample, target, reduction="none"
        ).view((self.num_mc_samples, batch_size, -1))

        loglikelihood = torch.mean(
            torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0)
            - math.log(self.num_mc_samples)
        )
        loss = -loglikelihood

        return loss


class ReshapedDistribution(td.Distribution):
    def __init__(
        self,
        base_distribution: td.Distribution,
        new_event_shape: Tuple[int, ...],
        validate_args=None,
    ):
        super().__init__(
            batch_shape=base_distribution.batch_shape,
            event_shape=new_event_shape,
            validate_args=validate_args,
        )
        self.base_distribution = base_distribution
        self.new_shape = base_distribution.batch_shape + new_event_shape

    @property
    def support(self):
        return self.base_distribution.support

    @property
    def arg_constraints(self):
        return self.base_distribution.arg_constraints()

    @property
    def mean(self):
        return self.base_distribution.mean.view(self.new_shape)

    @property
    def variance(self):
        return self.base_distribution.variance.view(self.new_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_distribution.rsample(sample_shape).view(
            sample_shape + self.new_shape
        )

    def log_prob(self, value):
        return self.base_distribution.log_prob(value.view(self.batch_shape + (-1,)))

    def entropy(self):
        return self.base_distribution.entropy()