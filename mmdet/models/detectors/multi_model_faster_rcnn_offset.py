# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import copy
from typing import Tuple

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from .faster_rcnn import FasterRCNN

import cv2
import numpy
import os
@MODELS.register_module()
class MultiModelFasterRCNN_OFFSET(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 *args,
                 mod_list=["img", "ir_img"],
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod_list = mod_list
        backbone = nn.ModuleList()
        if self.with_neck: neck = nn.ModuleList()
        self.fusion_module = nn.ModuleList()
        for _ in range(len(self.mod_list)):
            backbone.append(copy.deepcopy(self.backbone))
            if self.with_neck: neck.append(copy.deepcopy(self.neck))
        self.backbone = backbone
        if self.with_neck: self.neck = neck

        for _ in range(self.neck[0].num_outs):
            # fusion_module = nn.Sequential(
            #     nn.Conv2d(in_channels=256 * len(self.mod_list), out_channels=256, kernel_size=1),
            #     nn.ReLU())
            # self.fusion_module.append(fusion_module)
            self.fusion_module.append(nn.Conv2d(in_channels=256 * len(self.mod_list), out_channels=256, kernel_size=1))

        self.offset_module = nn.Conv2d(256, 2, 3, 1, 1)
        self.offset_loss = MODELS.build(dict(type='L1Loss', loss_weight=1.0))
        
    def forward(self,
                inputs: torch.Tensor,
                data_samples=None,
                mode: str = 'tensor',
                **kwargs):
        extra_inputs = kwargs.get('extra_inputs', None)

        if extra_inputs is None:
            super().forward(inputs, data_samples, mode)
        else:
            if mode == 'loss':
                return self.loss(inputs, data_samples, extra_inputs)
            elif mode == 'predict':
                return self.predict(inputs, data_samples, extra_inputs)
            elif mode == 'tensor':
                return self._forward(inputs, data_samples)
            else:
                raise RuntimeError(f'Invalid mode "{mode}". '
                                'Only supports loss, predict and tensor mode')

    def extract_feat(self, batch_inputs: Tensor,
                     batch_extra_inputs: Tensor) -> Tuple[Tensor]:
        xs = []
        for i, backbone in enumerate(self.backbone):
            # 选择输入：第一个模态使用 batch_inputs，其余模态使用 batch_extra_inputs
            inputs = batch_inputs if i == 0 else batch_extra_inputs[i - 1]

            # 提取 backbone 特征
            x = backbone(inputs)            
            if self.with_neck: 
                x = self.neck[i](x) 

            # 偏移 其他模态到 RGB
            if i == 0:
                raw_x = [x1.clone() for x1 in x]
                x = tuple([self.apply_offset_to_feature_map(x1, x1) for x1 in x])
                
                # 计算损失，这里的思路目前是只监督RGB模态，其他模态我还没有想好怎么计算损失                
                if self.training:
                    raw_x = [raw_x1.reshape(raw_x1.shape[0], raw_x1.shape[1], -1) for raw_x1 in raw_x]
                    raw_x = torch.cat(raw_x, dim=-1)
                    new_x = [x1.clone() for x1 in x]
                    new_x = [new_x1.reshape(new_x1.shape[0], new_x1.shape[1], -1) for new_x1 in new_x]
                    new_x = torch.cat(new_x, dim=-1)
                    loss = self.offset_loss(raw_x, new_x)
            else:
                x = tuple([self.apply_offset_to_feature_map(x1, xs[0][i]) for i, x1 in enumerate(x)])
            
            xs.append(x)
        
        # 融合每个特征层，并使用 fusion_module 处理
        out = [fusion_module(torch.cat([x[i] for x in xs], dim=1)) 
               for i, fusion_module in enumerate(self.fusion_module)]
        
        if self.training:
            return out, {'loss_offset': loss}
        else:
            return out

    def apply_offset_to_feature_map(self, batch_extra_inputs: Tensor, batch_inputs: Tensor) -> torch.Tensor:

        offset = self.offset_module(batch_extra_inputs - batch_inputs)
        B, _, H, W = batch_extra_inputs.shape
        
        # 生成标准网格 (H, W) -> (grid_y, grid_x)
        # 注意：grid_sample 需要 [-1, 1] 范围内的归一化坐标
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1)  # 形状为 (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1).to(batch_extra_inputs.device)  # 形状为 (B, H, W, 2)
        
        # 将偏移量从 (B, 2, H, W) 转换为 (B, H, W, 2)，与网格匹配
        offset = offset.permute(0, 2, 3, 1)  # (B, H, W, 2)
        
        # 生成新的采样网格
        new_grid = grid + offset
        
        # 使用 grid_sample 对特征图进行采样
        # align_corners=True 确保采样时保持坐标对齐
        warped_feature_map = F.grid_sample(batch_extra_inputs, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped_feature_map

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             batch_extra_inputs: Tensor) -> dict:        

        x, offset_loss = self.extract_feat(batch_inputs, batch_extra_inputs)

        losses = dict()
        losses.update(offset_loss)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                batch_extra_inputs: Tensor,
                rescale: bool = True) -> SampleList:

        assert self.with_bbox, 'Bbox head must be implemented.'
        
        x = self.extract_feat(batch_inputs, batch_extra_inputs)        

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

