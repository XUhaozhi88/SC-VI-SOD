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


@MODELS.register_module()
class DINO_Fusion_Simple(DINO):

    def __init__(self, *args, 
                 mod_list=["img", "ir_img"], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod_list = mod_list  
        
        # 初始化结构
        # backbone = nn.ModuleList()
        # if self.with_neck: neck = nn.ModuleList()
        # for _ in range(len(self.mod_list)):
        #     backbone.append(copy.deepcopy(self.backbone))
        #     # if self.with_neck: neck.append(copy.deepcopy(self.neck))
        # self.backbone = backbone
        # if self.with_neck: self.neck = neck
    
    def extract_feat_old(self, batch_inputs: Tensor,
                    batch_extra_inputs: Tensor) -> Tuple[Tensor]:
        x_V = self.backbone[0](batch_inputs)
        # x_I = self.backbone[1](batch_extra_inputs)
        x_I = self.backbone[1](batch_extra_inputs[0])
        x = [x_V1 + x_I1 for x_V1, x_I1 in zip(x_V, x_I)]
        if self.with_neck: 
            x = self.neck(x)
        return x
    
    def extract_feat(self, batch_inputs: Tensor,
                    batch_extra_inputs) -> Tuple[Tensor]:
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

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples = None, 
            batch_extra_inputs = None):        
        img_feats = self.extract_feat(batch_inputs, batch_extra_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward(self,
                inputs: torch.Tensor,
                data_samples=None,
                mode: str = 'tensor',
                **kwargs):
        extra_inputs = kwargs.get('extra_inputs', None)
        extra_data_samples = kwargs.get('extra_data_samples', None)

        if mode == 'tensor':
            extra_inputs = inputs.clone()
            # extra_data_samples = copy.deepcopy(data_samples)

        if extra_inputs is None:
            super().forward(inputs, data_samples, mode)
        else:
            if mode == 'loss':
                return self.loss(inputs, data_samples, extra_inputs, extra_data_samples)
            elif mode == 'predict':
                return self.predict(inputs, data_samples, extra_inputs)
            elif mode == 'tensor':
                return self._forward(inputs, data_samples, extra_inputs)
            else:
                raise RuntimeError(f'Invalid mode "{mode}". '
                                'Only supports loss, predict and tensor mode')