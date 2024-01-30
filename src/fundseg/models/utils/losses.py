import torch
import torch.nn as nn


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
