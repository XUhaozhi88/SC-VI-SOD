# Author: Xu Haozhi.
# Time:   2025.05.15

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weighted_loss


@weighted_loss
def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-6,
) -> Tensor:

    # Check var size
    # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    if var.size() != input.size():

        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if input.size()[:-1] == var.size():
            var = torch.unsqueeze(var, -1)

        # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
        # This is also a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
        elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:  # Heteroscedastic case
            pass

        # If none of the above pass, then the size of var is incorrect.
        else:
            raise ValueError("var is of incorrect size")

    # Entries of var must be non-negative
    # if torch.any(var < 0):
    #     raise ValueError("var has negative entry/entries")

    # Clamp for stability
    # var = var.clone()
    # with torch.no_grad():
    #     var.clamp_(min=eps)

    # Calculate the loss
    # loss = 0.5 * (torch.log(var) + (input - target)**2 / var)
    loss = 0.5 * (torch.exp(-1 * var) * (input - target)**2 + var)
    if full:
        loss += 0.5 * math.log(2 * math.pi)
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
                 full: bool = False,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.full = full
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                var: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            var (Tensor): The variance of the prediction.
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
            return (pred * weight).sum() + (var * weight).sum()  # important, 如果没有的话会导致var_branches没有参与梯度计算
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # loss_bbox = self.loss_weight * \
        #     torch.nn.functional.gaussian_nll_loss(pred, target, var, reduction=reduction)
        loss_bbox = self.loss_weight * gaussian_nll_loss(
            pred, target, weight, var=var, full=self.full, eps=self.eps, 
            reduction=reduction, avg_factor=avg_factor)
        return loss_bbox