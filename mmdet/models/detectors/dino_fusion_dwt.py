# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union
import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.init import normal_
from mmengine.runner.amp import autocast
from mmengine.model import BaseModel

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType
from .dino import DINO

from .dino_fusion_dwt_utils import SingleScaleFusion


@MODELS.register_module()
class DINO_Fusion_DWT(DINO):

    def __init__(self, *args, 
                 mod_list=["img", "ir_img"], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod_list = mod_list  
        
        # 初始化结构
        backbone = nn.ModuleList()
        if self.with_neck: neck = nn.ModuleList()
        for _ in range(len(self.mod_list)):
            backbone.append(copy.deepcopy(self.backbone))
            if self.with_neck: neck.append(copy.deepcopy(self.neck))
        self.backbone = backbone
        if self.with_neck: self.neck = neck

        # 融合
        # self.fusion = Fusion(channels=256, reduction=4, num_layer=4)
        self.fusions = nn.ModuleList(SingleScaleFusion(in_channels=256, mid_channels=64) 
                                    for _ in range(4))
        # self.fusions = Dual_Fusion(with_cp=True, batch_size=batch_size)
        # self.fusions = Dual_Fusion_New()#with_cp=True)        
        
    def extract_feat(self, batch_inputs: Tensor,
                    batch_extra_inputs: Tensor) -> Tuple[Tensor]:
        xs = []
        for i, backbone in enumerate(self.backbone):
            # 选择输入：第一个模态使用 batch_inputs，其余模态使用 batch_extra_inputs
            inputs = batch_inputs if i == 0 else batch_extra_inputs[i - 1]

            # 提取 backbone 特征
            x = backbone(inputs)            
            if self.with_neck: 
                neck = self.neck[i]
                x = neck(x)  
            xs.append(x)

            # if i == 0: 
            #     out = list(x)
            # else:
            #     for level in range(len(out)):
            #         out[level] += x[level]
        
        ## 简单的sum融合
        # out = [torch.sum(torch.stack([x[i] for x in xs], dim=0), dim=0)
        #         for i in range(len(xs[0]))]
        # DWT融合
        out = [fusion(rgb_x, ir_x) for rgb_x, ir_x, fusion in zip(xs[0], xs[1], self.fusions)]
        return tuple(out)   # fusion
        # return self.fusions(*xs)
    
    def extract_feat1(self, batch_inputs: Tensor,
                    batch_extra_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs, batch_extra_inputs[0])            
        if self.with_neck: 
            x = self.neck(x)
        return x
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,             
             batch_extra_inputs: Tensor,
             batch_extra_data_samples: SampleList) -> Union[dict, list]:
        
        img_feats = self.extract_feat(batch_inputs, batch_extra_inputs)       

        for img_feat in img_feats:
            if torch.isnan(img_feat).any(): # search nan
                raise AssertionError
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs,
                batch_data_samples, batch_extra_inputs, rescale: bool = True):

        # image feature extraction
        img_feats = self.extract_feat(batch_inputs, batch_extra_inputs)

        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def forward(self,
                inputs: torch.Tensor,
                data_samples=None,
                mode: str = 'tensor',
                **kwargs):
        extra_inputs = kwargs.get('extra_inputs', None)
        extra_data_samples = kwargs.get('extra_data_samples', None)

        if extra_inputs is None:
            super().forward(inputs, data_samples, mode)
        else:
            if mode == 'loss':
                return self.loss(inputs, data_samples, extra_inputs, extra_data_samples)
            elif mode == 'predict':
                return self.predict(inputs, data_samples, extra_inputs)
            elif mode == 'tensor':
                return self._forward(inputs, data_samples)
            else:
                raise RuntimeError(f'Invalid mode "{mode}". '
                                'Only supports loss, predict and tensor mode')