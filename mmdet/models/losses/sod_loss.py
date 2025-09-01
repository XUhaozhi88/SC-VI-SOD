import torch

from torch import nn
from torch import Tensor

from mmdet.structures.bbox import bbox_overlaps
from .utils import weighted_loss

def sigmoid(x, k):
    return 1/(1+torch.exp(-(1/k) * x))

def xyxy2xyw2h2(bbox):
    _bbox = bbox.clone()
    _bbox[:,0] = bbox[:,0]
    _bbox[:,1] = bbox[:,1]
    _bbox[:,2] = (bbox[:,2] - bbox[:,0])/2
    _bbox[:,3] = (bbox[:,3] - bbox[:,1])/2
    return _bbox

@weighted_loss
def safit_loss(pred: Tensor, 
               target: Tensor, 
               eps: float = 1e-7, 
               C: int = 32,
               reduction=None, 
               avg_factor=None) -> Tensor:
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # avoid fp16 overflow
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    # IOU
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    
    # NWD
    pred_xyhw = xyxy2xyw2h2(pred)
    target_xyhw = xyxy2xyw2h2(target)
    nwd = torch.exp(-torch.sqrt(torch.sum((pred_xyhw - target_xyhw)*(pred_xyhw - target_xyhw),dim=1))/C)
 
    # SAFit
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    area = ag
    safit = sigmoid(torch.sqrt(area)-C, C) * ious+\
            (1-sigmoid(torch.sqrt(area)-C, C)) * nwd
    
    if fp16:
        safit = safit.to(torch.float16)

    loss = 1 - safit
    return loss

@weighted_loss
def nwd_loss(pred: Tensor, 
               target: Tensor, 
               eps: float = 1e-7, 
               C: int = 32,
               reduction=None, 
               avg_factor=None) -> Tensor:
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    pred, target = xyxy2xyw2h2(pred), xyxy2xyw2h2(target)
    nwd = torch.exp(-torch.norm((pred-target), p=2, dim=1)/C)
    loss = 1 - nwd.clamp(min=-1.0, max=1.0)
    return loss

class SAFitLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(SAFitLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * safit_loss(
            pred,
            target,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

class NWDLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0, C=32):
        super(NWDLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.C = C

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * nwd_loss(
            pred,
            target,
            eps=self.eps,
            C = self.C,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss