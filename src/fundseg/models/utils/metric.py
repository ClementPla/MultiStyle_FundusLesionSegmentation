from typing import Any, List, Literal, Optional, Union

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)


class MulticlassAUCPrecisionRecallCurve(MulticlassPrecisionRecallCurve):
    def compute(self) -> Union[Tensor, List[Tensor]]:
        precision, recall, threshold = super().compute()
        aucs = []
        for pr, re in zip(precision, recall):
            argsort = torch.argsort(re)
            aucs.append(torch.trapezoid(pr[argsort], re[argsort]))
        return torch.Tensor(aucs)


class BinaryAUCPrecisionRecallCurve(BinaryPrecisionRecallCurve):
    def compute(self):
        precision, recall, threshold = super().compute()
        argsort = torch.argsort(recall)
        return torch.trapezoid(precision[argsort], recall[argsort])


class MultilabelAUCPrecisionRecallCurve(MultilabelPrecisionRecallCurve):
    def compute(self):
        precision, recall, threshold = super().compute()
        aucs = []
        for pr, re in zip(precision, recall):
            argsort = torch.argsort(re)
            aucs.append(torch.trapezoid(pr[argsort], re[argsort]))
        return torch.Tensor(aucs)


class AUCPrecisionRecallCurve:
    def __new__(
        cls,
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        kwargs.update(dict(thresholds=thresholds, ignore_index=ignore_index, validate_args=validate_args))
        if task == "binary":
            return BinaryAUCPrecisionRecallCurve(**kwargs)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            return MulticlassAUCPrecisionRecallCurve(num_classes, **kwargs)
        if task == "multilabel":
            assert isinstance(num_labels, int)
            return MultilabelAUCPrecisionRecallCurve(num_labels, **kwargs)
        raise ValueError(
            f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
        )



