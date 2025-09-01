# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import torch
from torch import Tensor, nn

from typing import List

from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .faster_rcnn import FasterRCNN

import cv2
import numpy


@MODELS.register_module()
class FasterRCNNScale(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 *args,
                 scale_thresholds: List = [],
                 loss_scale=dict(type='L1Loss', loss_weight=1.0),
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # 生成attention图，和真实目标的attention图做损失
        self.scale_thresholds = scale_thresholds
        self.scale_conv = nn.ModuleList()
        for _ in range(len(scale_thresholds) + 1):
            scale_conv = nn.Sequential(     
                nn.Conv2d(256, 32, 1, 1),
                nn.ReLU(),                
                nn.Conv2d(32, 1, 1, 1))
            self.scale_conv.append(scale_conv)

        # atten loss
        self.loss_scale = MODELS.build(loss_scale)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super(FasterRCNNScale, self).init_weights()

    def scale_loss(self, x: Tensor, batch_data_samples: list, scale) -> dict:

        def generate_2d_gaussian(shape, center, sigma, dtype, device):
            """
            生成一个二维高斯分布，shape 为高斯图像的尺寸，center 为高斯中心，sigma 为标准差。
            """
            h, w = shape
            y = torch.arange(0, h, dtype=dtype, device=device)
            x = torch.arange(0, w, dtype=dtype, device=device)
            y, x = torch.meshgrid(y, x, indexing='ij')
            
            # 计算二维高斯分布
            gaussian = torch.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma ** 2))
            return gaussian

        xs = [scale_conv(x).squeeze(1) for scale_conv in self.scale_conv]  # 不同尺度的特征图
        attens = [torch.zeros_like(x) for x in xs]  # 初始化注意力图

        for id, data_sample in enumerate(batch_data_samples):
            bboxes = data_sample.gt_instances.bboxes  # 已经经过缩放
            metainfo = data_sample.metainfo
            img_shape = metainfo['img_shape']

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)

                # 转换 bbox 坐标到对应的特征图尺度
                x1, y1, x2, y2 = int(x1 / scale) - 1, int(y1 / scale) - 1, int(x2 / scale) + 1, int(y2 / scale) + 1
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, img_shape[0] // scale), min(y2, img_shape[1] // scale)

                # 选择不同尺度的特征图
                if len(self.scale_thresholds) == 0:
                    scale_idx = 0
                else:
                    # 对比bbox area属于哪一个尺度，常见的是小、中、大目标
                    scale_idx = len(self.scale_thresholds) + 1
                    for scale_idx, scale_threshold in enumerate(self.scale_thresholds):
                        if area <= scale_threshold * scale_threshold:
                            break

                if x1 > x2 or y1 > y2: continue
                
                # 计算中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # 生成二维高斯分布
                gaussian = generate_2d_gaussian(
                    shape=(x2 - x1, y2 - y1), 
                    center=(center_x, center_y), 
                    sigma=(center_x + center_y) / 2, 
                    dtype=xs[0].dtype, device=xs[0].device)

                # 将高斯分布添加到对应的区域
                attens[scale_idx][id, x1:x2, y1:y2] += gaussian

        # atten0 = attens[0].sigmoid().detach().cpu().numpy()        
        # denominator = atten0.max() - atten0.min() # Normalize atten0 to [0, 1] while handling division by zero
        # atten0 = numpy.zeros_like(atten0) if denominator == 0 else (atten0 - atten0.min()) / denominator
        # atten0 = numpy.nan_to_num(atten0, nan=0.0, posinf=255.0, neginf=0.0)    # Remove NaN or infinity values
        # atten0 = (atten0 * 255).astype(numpy.uint8)
        # if len(atten0.shape) != 2: atten0 = atten0[0]
        # atten0_colored = cv2.applyColorMap(atten0, cv2.COLORMAP_JET)
        # atten0_colored = cv2.resize(atten0_colored, (640, 512))
        # cv2.imwrite(os.path.join("D:/Files/Code/Python/Scale_IoU/mmdetection/results/atten0/", os.path.basename(metainfo['img_path'])), atten0_colored)
        
        # 计算损失
        attens = torch.stack(attens)
        xs = torch.stack(xs)
        loss = self.loss_scale(attens.sigmoid(), xs.sigmoid())    # 到时候可以考虑筛选有目标的特征层

        return {'loss_scale': loss}

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()
        loss_scale = self.scale_loss(x[-1], batch_data_samples, scale=64)
        # loss_scale = self.scale_loss(x[-2], batch_data_samples, scale=32)
        losses.update(loss_scale)

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