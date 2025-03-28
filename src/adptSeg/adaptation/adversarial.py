from copy import deepcopy
from typing import Callable, Optional, Union

import torch


def _clip(inputs: torch.Tensor, outputs: torch.Tensor, radius, norm: str = "Linf"):
    diff = outputs - inputs
    if norm == "Linf":
        return inputs + torch.clamp(diff, -radius, radius)
    elif norm == "L2":
        return inputs + torch.renorm(diff, 2, 0, radius)
    else:
        raise AssertionError(f"Norm constraint must be L2 or Linf, got {norm}")


class PGD:
    def __init__(
        self,
        forward_func: Callable,
        loss_func: Callable,
        lower_bound: float = float("-inf"),
        upper_bound: float = float("inf"),
        norm="Linf",
        sign_grad: bool = True,
    ):
        super().__init__()
        self.loss_func = loss_func
        self.forward_func = forward_func
        self.forward_func.eval()
        self.bound = lambda x: torch.clamp(x, lower_bound, upper_bound)
        self.norm = norm
        self.zero_thresh = 1e-7
        self.sign_grad = sign_grad

    def get_single_step_perturbation(self, x: torch.Tensor, grads: torch.Tensor, epsilon: float):
        if self.sign_grad:
            return x - epsilon * (torch.abs(grads) > self.zero_thresh) * torch.sign(grads)

        else:
            return torch.where(
                torch.abs(grads) > self.zero_thresh,
                x - epsilon * grads,
                x,
            )

    def get_gradients(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        interpolation: Optional[float] = None,
        as_regression: bool = False,
        interpolation_mode: str = "loss",
    ):
        pred = self.forward_func(x)

        if interpolation is not None:
            if interpolation_mode == "loss":
                if as_regression:
                    target = target[0] * (1 - interpolation) + target[1] * interpolation
                    output = self.loss_func(pred, target)
                else:
                    l1 = self.loss_func(pred, target[0])
                    l2 = self.loss_func(pred, target[1])
                    l1 = l1 / l1.item()
                    l2 = l2 / l2.item()
                    # l1 = torch.sigmoid(l1)
                    # l2 = torch.sigmoid(l2)
                    output = l1 * (1 - interpolation) + l2 * interpolation
            else:
                output = self.loss_func(pred, target)
        else:
            output = self.loss_func(pred, target)
        output = output.unsqueeze(0)
        with torch.autograd.set_grad_enabled(True):
            grads = torch.autograd.grad(torch.unbind(output), x, allow_unused=True)[0]

        # grads = grads[0] / torch.norm(grads[0], p=torch.inf, keepdim=True)

        return grads

    def perturb(
        self,
        x: torch.Tensor,
        target: Union[torch.Tensor, int, list[int, torch.Tensor]],
        step_size: float,
        radius: float,
        step_num: int,
        interpolation: Optional[float] = None,
        interpolation_mode: str = "loss",  # "loss" or "input"
        as_regression: bool = False,
        **kwargs,
    ):
        x.requires_grad_(True)
        # Create a copy of the input tensor
        x = x.clone()
        xmin = x.reshape(x.shape[0], -1).min(1, keepdim=True)[0].view(x.shape[0], 1, 1, 1)
        xmax = x.reshape(x.shape[0], -1).max(1, keepdim=True)[0].view(x.shape[0], 1, 1, 1)
        b = x.shape[0]

        if interpolation is not None:
            if interpolation_mode == "input":
                x = torch.cat([x, x], dim=0)

            assert isinstance(target, list)
            targets = []
            for i, t in enumerate(target):
                targets.append(torch.tensor([t]).to(x.device).expand(b))
            target = targets
            if interpolation_mode == "input":
                target = torch.cat(target, dim=0)
        elif isinstance(target, int):
            target = torch.tensor([target]).to(x.device).expand(b)

        perturbed_input = x
        for i in range(step_num):
            gradients = self.get_gradients(
                perturbed_input,
                target,
                interpolation=interpolation,
                as_regression=as_regression,
                interpolation_mode=interpolation_mode,
            )
            perturbed_input = self.get_single_step_perturbation(perturbed_input, gradients, epsilon=step_size)
            perturbed_input = _clip(x, perturbed_input, radius, norm=self.norm)
            perturbed_input = self.bound(perturbed_input)  # .detach()
            perturbed_input.requires_grad_(True)
        if interpolation is not None and interpolation_mode == "input":
            perturbed_input_1 = perturbed_input[:b]
            perturbed_input_2 = perturbed_input[b:]
            perturbed_input_1 = torch.clamp(perturbed_input_1, xmin, xmax)
            perturbed_input_2 = torch.clamp(perturbed_input_2, xmin, xmax)
            perturbed_input = perturbed_input_1 * (1.0 - interpolation) + perturbed_input_2 * interpolation

        perturbed_input = torch.clamp(perturbed_input, xmin, xmax)
        return perturbed_input
