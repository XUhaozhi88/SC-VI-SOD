# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector

import torch
import torch.nn as nn

@MODELS.register_module()
class RetinaNet_New(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.num_scale = 3   # 1, 3
        self.scale_conv = nn.ModuleList()
        for _ in range(self.num_scale):
            scale_conv = nn.Sequential(     
                nn.Conv2d(256, 32, 1, 1),
                nn.ReLU(),                
                nn.Conv2d(32, 1, 1, 1))
            self.scale_conv.append(scale_conv)

    def scale_loss(self, x, batch_data_samples: list, scale) -> dict:

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
                if self.num_scale == 1:
                    scale_idx = 0
                elif self.num_scale == 3:
                    if area <= 1024:
                        scale_idx = 0
                        # sigma = 1.0  # 小目标，较小的扩散
                    elif area <= 9216:
                        scale_idx = 1
                        # sigma = 2.0  # 中等目标，较大的扩散
                    else:
                        scale_idx = 2
                        # sigma = 3.0  # 大目标，较大的扩散

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
        # cv2.imwrite(os.path.join("D:/Files/Code/Python/Scale_IoU/mmdetection/results/", os.path.basename(metainfo['img_path'])), atten0_colored)
        
        # 计算损失
        loss = 0.1
        for atten, x in zip(attens, xs):
            loss += torch.mean(torch.abs(atten.sigmoid() - x.sigmoid()))    # 到时候可以考虑筛选有目标的特征层

        return {'scale_loss': loss}

    def loss(self, batch_inputs,
             batch_data_samples):
        x = self.extract_feat(batch_inputs)
        losses = dict()
        scale_loss = self.scale_loss(x[-1], batch_data_samples, scale=128)
        losses.update(scale_loss)
        losses.update(self.bbox_head.loss(x, batch_data_samples))
        return losses
