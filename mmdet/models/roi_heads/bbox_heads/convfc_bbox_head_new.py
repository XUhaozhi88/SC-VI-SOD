# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union
import csv
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from .bbox_head import BBoxHead
from mmdet.models.utils import empty_instances
from mmdet.models.layers import multiclass_nms
from mmdet.models.losses import accuracy
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.utils import ConfigType

from .convfc_bbox_head import ConvFCBBoxHead


@MODELS.register_module()
class ConvFCBBoxHeadNew(ConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 *args,
                 loss_cls_var: ConfigType = None,
                 loss_bbox_var: ConfigType = None,
                 mode='mean',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = mode
        # for gaussian nll loss
        if mode == 'mean_var':
            self.loss_cls_var = MODELS.build(loss_cls_var) if loss_cls_var else None # diff
            self.loss_bbox_var = MODELS.build(loss_bbox_var) if loss_bbox_var else None
        
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=self.cls_last_dim, out_features=cls_channels)
            self.fc_cls = MODELS.build(cls_predictor_cfg_)
            
            # for gaussian nll loss
            if mode == 'mean_var':
                self.var_cls = MODELS.build(cls_predictor_cfg_)  # diff
                nn.init.constant_(self.var_cls.bias, 0.0)
                nn.init.normal_(self.var_cls.weight, mean=0.0, std=0.01)

        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else \
                box_dim * self.num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                reg_predictor_cfg_.update(
                    in_features=self.reg_last_dim, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)

            # for gaussian nll loss
            if mode == 'mean_var':
                self.var_reg = MODELS.build(reg_predictor_cfg_)  # diff
                nn.init.constant_(self.var_reg.bias, 0.0)
                nn.init.normal_(self.var_reg.weight, mean=0.0, std=0.01)

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # classify branch
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        # for gaussian nll loss
        if self.mode == 'mean_var':
            cls_var = self.var_cls(x_cls) if self.with_cls else None   # diff
            # cls_var = torch.sigmoid(cls_var)
            cls_var = torch.log(1. + torch.exp(cls_var))
            if self.training:
                cls_score = torch.cat((cls_score, cls_var), dim=-1)

        # box regression branch
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        # for gaussian nll loss
        if self.mode == 'mean_var':
            bbox_var = self.var_reg(x_reg) if self.with_reg else None   # diff
            # bbox_var = torch.sigmoid(bbox_var)
            bbox_var = torch.log(1. + torch.exp(bbox_var))
            if self.training:
                bbox_pred = bbox_pred.view(bbox_pred.size(0), self.num_classes, self.bbox_coder.encode_size)
                bbox_var = bbox_var.view(bbox_pred.size(0), self.num_classes, self.bbox_coder.encode_size)
                bbox_pred = torch.cat((bbox_pred, bbox_var), dim=-1)
                bbox_pred = bbox_pred.view(bbox_pred.size(0), self.num_classes * self.bbox_coder.encode_size * 2)
        
        return cls_score, bbox_pred
    
    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:

        losses = dict()

        if cls_score is not None:
            # for gaussian nll loss
            if self.mode == 'mean_var':  # diff
                num_classes = cls_score.shape[-1] // 2
                cls_var = cls_score[:, num_classes:]
                cls_score = cls_score[:, :num_classes]

            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                if self.mode == 'mean':
                    loss_cls_ = self.loss_cls(
                        cls_score,
                        labels,
                        label_weights,
                        avg_factor=avg_factor,
                        reduction_override=reduction_override)
                # for gaussian nll loss
                elif self.mode == 'mean_var':
                    loss_cls_ = self.loss_cls(
                        cls_score,
                        cls_var,    # diff
                        labels,
                        label_weights,
                        avg_factor=avg_factor,
                        reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                
                # for gaussian nll loss
                if self.mode == 'mean_var':  # diff
                    pos_bbox_pred_var = pos_bbox_pred[:, 4:]
                    pos_bbox_pred = pos_bbox_pred[:, :4]

                if self.mode == 'mean':
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
                # for gaussian nll loss
                elif self.mode == 'mean_var':
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        pos_bbox_pred_var,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses
    
    def loss_old(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:

        losses = dict()

        if cls_score is not None:   # diff
            if self.loss_cls_var:
                num_classes = cls_score.shape[-1] // 2
                cls_var = cls_score[:, num_classes:]
                cls_score = cls_score[:, :num_classes]

            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)

                if self.loss_cls_var:   # diff
                    bg_class_ind = self.num_classes
                    # 0~self.num_classes-1 are FG, self.num_classes is BG
                    pos_inds = (labels >= 0) & (labels < bg_class_ind)
                    loss_cls_var = self.loss_cls_var(
                        pred=cls_score[pos_inds.type(torch.bool), 
                                       labels[pos_inds.type(torch.bool)]],
                        var=cls_var[pos_inds.type(torch.bool), 
                                    labels[pos_inds.type(torch.bool)]],
                        target=labels[pos_inds.type(torch.bool)],
                        weight=label_weights[pos_inds.type(torch.bool)],
                        avg_factor=pos_inds.type(torch.float32).sum(),
                        reduction_override=reduction_override)
                    if isinstance(loss_cls_var, dict):
                        losses.update(loss_cls_var)
                    else:
                        losses['loss_cls_guassian-nll'] = loss_cls_var

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), self.num_classes,
                        -1)[pos_inds.type(torch.bool),
                            labels[pos_inds.type(torch.bool)]]
                
                if self.loss_bbox_var:  # diff
                    pos_bbox_pred_var = pos_bbox_pred[:, 4:]
                    pos_bbox_pred = pos_bbox_pred[:, :4]
                
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                
                if self.loss_bbox_var:
                    losses['loss_bbox_guassian-nll'] = self.loss_bbox_var(  # diff
                        pred=pos_bbox_pred,
                        var=pos_bbox_pred_var,
                        target=bbox_targets[pos_inds.type(torch.bool)],
                        weight=bbox_weights[pos_inds.type(torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses


@MODELS.register_module()
class Shared2FCBBoxHeadNew(ConvFCBBoxHeadNew):

    def __init__(self, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)        
        
        self.train_log_file = '/workspace/mmdetection/results/rgbt_tiny-RGB/0.2train-0.5val-12e/faster_rcnn/bs8-auto_scale_lr/anchor_scale_8-box-cls-gnllloss/train.json'
        self.val_log_file = '/workspace/mmdetection/results/rgbt_tiny-RGB/0.2train-0.5val-12e/faster_rcnn/bs8-auto_scale_lr/anchor_scale_8-box-cls-gnllloss/val.json'

        # Create a new file if it doesn't exist
        if not os.path.exists(self.train_log_file):
            with open(self.train_log_file, "w") as f_train:
                json.dump([], f_train)  # Initialize an empty JSON array
        if not os.path.exists(self.val_log_file):
            with open(self.val_log_file, "w") as f_val:
                json.dump([], f_val)  # Initialize an empty JSON array

        self.train_first_entry = not os.path.exists(self.train_log_file)  # Check if the file exists
        self.val_first_entry = not os.path.exists(self.val_log_file)

        # Open the file in append mode
        self.train_file = open(self.train_log_file, "a")
        self.val_file = open(self.val_log_file, "a")

        self.train_iteration = 0
        self.val_iteration = 0
    
    def CSVLogger(self, cls_pred, cls_var, box_pred, box_var):

        def pred_var(preds, variances):
            # 计算统计量
            var_min, var_max, var_mean, var_std, \
                mean_min, mean_max, mean_mean, mean_std = [], [], [], [], [], [], [], []

            for i in range(preds.shape[1]):
                pred = preds[:, i]
                variance = variances[:, i]
                var_min.append(variance.min().cpu().item())
                var_max.append(variance.max().cpu().item())
                var_mean.append(variance.mean().cpu().item())
                var_std.append(variance.std().cpu().item())
                mean_min.append(pred.min().cpu().item())
                mean_max.append(pred.max().cpu().item())
                mean_mean.append(pred.mean().cpu().item())
                mean_std.append(pred.std().cpu().item())
            return {"Var_Min": var_min, "Var_Max": var_max, "Var_Mean": var_mean, "Var_Std": var_std,
                   "Mean_Min": mean_min, "Mean_Max": mean_max, "Mean_Mean": mean_mean, "Mean_Std": mean_std}

        if self.training:
            if self.train_iteration % 100 == 0:
                # Load existing data
                with open(self.train_log_file, "r") as f:
                    data = json.load(f)
                iteration = self.train_iteration
                # Append new data
                data.append({
                    "Iteration": iteration,
                    "Class": pred_var(cls_pred, cls_var),
                    "Box": pred_var(box_pred, box_var)})                
                # Write updated data back to file
                with open(self.train_log_file, "w") as f:
                    json.dump(data, f, indent=4)
            
            self.train_iteration += 1
        else:      
            if self.val_iteration % 200 == 0:      
                with open(self.val_log_file, "r") as f:
                    data = json.load(f)
                iteration = self.val_iteration
                # Append new data
                data.append({
                    "Iteration": iteration,
                    "Class": pred_var(cls_pred, cls_var),
                    "Box": pred_var(box_pred, box_var)})
                # Write updated data back to file
                with open(self.val_log_file, "w") as f:
                    json.dump(data, f, indent=4)

            self.val_iteration += 1
        
    def forward_new(self, x: Tuple[Tensor]) -> tuple:
        def round4(k):
            return torch.round(k * 10000) / 10000

        cls_score, bbox_pred = super().forward(x)
        cls_pred, cls_var = torch.chunk(cls_score, 2, dim=1)
        box_pred, box_var = torch.chunk(bbox_pred, 2, dim=1)

        self.CSVLogger(round4(cls_pred.detach()), round4(cls_var.detach()), 
                       round4(box_pred.detach()), round4(box_var.detach()))

        return cls_score, bbox_pred