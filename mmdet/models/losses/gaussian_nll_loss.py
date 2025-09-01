# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weighted_loss


# @weighted_loss
# def gaussian_nll_loss(pred: Tensor, target: Tensor) -> Tensor:
#     """GaussianNLL loss.

#     Args:
#         pred (Tensor): The prediction.
#         target (Tensor): The learning target of the prediction.

#     Returns:
#         Tensor: Calculated loss
#     """
#     if target.numel() == 0:
#         return pred.sum() * 0

#     pred, var = pred
#     assert pred.shape[0] == target.shape[0]
#     # 防止梯度爆炸哦
#     var = torch.clamp(var, min=1e-6)

#     if len(target.shape) == 2:
#         # box
#         gnll_loss = 0.5 * torch.log(2 * torch.pi * var ** 2) \
#             + ((target - pred) ** 2) / (2 * var ** 2)
#     elif len(target.shape) == 1:
#         # class
#         # Convert labels to one-hot encoding
#         labels_one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
#         gnll_loss = 0.5 * torch.log(2 * torch.pi * var ** 2) \
#             + ((labels_one_hot - pred) ** 2) / (2 * var ** 2)
#         gnll_loss = gnll_loss.mean(dim=-1) + 1e-3 * (var ** 2).mean(dim=-1)
#     # 还可以引入L2正则化 torch.mean(var ** 2)，前面可以用个较小的系数
#     loss = gnll_loss# + l1_loss
#     return loss

@weighted_loss
def gaussian_nll_loss(pred: Tensor, target: Tensor) -> Tensor:
    """GaussianNLL loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    pred, var = pred
    assert pred.shape[0] == target.shape[0]
    # 防止梯度爆炸哦
    var = torch.clamp(var, min=1e-6, max=10.)

    if len(target.shape) == 2:
        # box
        loss = 0.5 * torch.log(2 * torch.pi * var) \
            + ((target - pred) ** 2) / (2 * var)
        loss = loss + 1e-3 * var ** 2   # 引入正则化项，维持稳定

    elif len(target.shape) == 1:
        # class
        # # Convert labels to one-hot encoding
        # labels_one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        # loss = 0.5 * torch.log(2 * torch.pi * var) \
        #     + ((labels_one_hot - pred) ** 2) / (2 * var)
        # loss = loss.mean(dim=-1)# + 1e-3 * (var).mean(dim=-1)

        # Gather only the predictions and variances corresponding to the correct labels
        pred_for_target = pred.gather(1, target.unsqueeze(1)).squeeze(1)  # Shape: (N,)
        var_for_target = var.gather(1, target.unsqueeze(1)).squeeze(1)  # Shape: (N,)

        # Compute Gaussian NLL loss for the correct labels
        loss = 0.5 * torch.log(2 * torch.pi * var_for_target) + ((pred_for_target - 1) ** 2) / (2 * var_for_target)
        # loss = loss.mean(dim=-1)
        loss = loss + 1e-3 * (var ** 2).mean(dim=-1)   # 引入正则化项，维持稳定

    # 还可以引入L2正则化 torch.mean(var)，前面可以用个较小的系数
    return loss


@MODELS.register_module()
class GaussianNLLLoss(nn.Module):
    """GaussianNLL loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                var: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * gaussian_nll_loss(
            (pred, var), target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox