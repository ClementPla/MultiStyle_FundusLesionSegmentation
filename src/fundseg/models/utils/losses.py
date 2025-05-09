import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelSoftBinaryCrossEntropy(nn.Module):
    def __init__(
        self,
        smooth: float = 0,
        weighted: bool = True,
        mcb: bool = True,
        hp_lambda: int = 10,
        mode: str = "multilabel",
        epsilon: float = 1e-6,
        from_logits=True,
    ):
        super(MultiLabelSoftBinaryCrossEntropy, self).__init__()
        self.smooth_factor = smooth
        self.logits = from_logits
        assert mode in ["binary", "multilabel"], "mode argument must be either multilabel or binary, got %s" % mode
        if from_logits:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none" if weighted else "mean")

        else:
            self.criterion = nn.BCELoss(reduction="none" if weighted else "mean")

        self.weighted = weighted
        self.hp_lambda = hp_lambda
        self.MCB = mcb
        self.epsilon = epsilon
        self.type_loss = mode

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        b, c, h, w = y_true.shape
        y_true = y_true.float()

        if self.smooth_factor:
            smooth = torch.rand_like(y_true) * self.smooth_factor
            soft_targets = y_true * (1 - smooth) + (1 - y_true) * smooth
        else:
            soft_targets = y_true

        ce_loss = self.criterion(y_pred, soft_targets)
        if not self.weighted:
            return ce_loss.mean()
        if not self.MCB:
            N = h * w
            weights = y_true.sum(dim=(2, 3), keepdim=True) / N
            betas = 1 - weights
            ce_loss = y_true * ce_loss * betas + (1 - y_true) * ce_loss * weights
            ce_loss = ce_loss.sum() / (b * N)

        else:
            y_pos_count = y_true.sum(dim=(0, 2, 3), keepdim=False)
            mcb_loss = 0
            for i, TP in enumerate(y_pos_count):
                """
                i is the class
                k is the number of positive elements per class
                """

                class_loss = ce_loss[:, i]
                y_true_positive = y_true[:, i]
                y_true_negative = 1 - y_true_positive

                TN = b * h * w - TP
                pos_loss = y_true_positive * class_loss
                neg_loss = (y_true_negative * class_loss).flatten(1, 2)

                topk = max(min(TP * self.hp_lambda, TN) / b, self.hp_lambda)

                ik = torch.topk(neg_loss, k=int(topk), dim=1, sorted=False).values
                # We can't use a "topk" per image on the batch, so we take an batch-wise value
                # (limitation of the topk function)
                beta_k = ik.shape[1] / (TP / b + ik.shape[1])
                # For the same reason, beta_k is batch-wise, not image-wise.
                # The original paper defines a single beta instead of beta_k; the rational of this choice is unclear.
                mcb_loss += (ik * (1 - beta_k)).mean()  # Negative loss
                mcb_loss += beta_k * pos_loss.sum() / (TP + self.epsilon)
            ce_loss = mcb_loss

        return ce_loss


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (alpha - ones).mul(torch.digamma(alpha) - torch.digamma(sum_alpha)).sum(dim=1, keepdim=True)
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device))
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device))
    return loss


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device))
    return loss
