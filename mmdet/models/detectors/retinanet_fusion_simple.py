# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .retinanet import RetinaNet

import torch
from torch import Tensor, nn
import copy
from typing import Dict, Optional, Tuple, Union


from mmdet.registry import MODELS
from mmdet.structures import SampleList


@MODELS.register_module()
class RetinaNet_Fusion_Simple(RetinaNet):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self, *args, 
                 mod_list=["img", "ir_img"], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod_list = mod_list  
        
        # 初始化结构
        backbone = nn.ModuleList()
        # if self.with_neck: neck = nn.ModuleList()
        for _ in range(len(self.mod_list)):
            backbone.append(copy.deepcopy(self.backbone))
            # if self.with_neck: neck.append(copy.deepcopy(self.neck))
        self.backbone = backbone
        # if self.with_neck: self.neck = neck

        # 融合
        # self.fusion = Fusion(channels=256, reduction=4, num_layer=4)
        # self.fusions = nn.ModuleList(SingleScaleFusion(channels=256, reduction=4) 
        #                             for _ in range(4))
        # self.fusions = Dual_Fusion(with_cp=True, batch_size=batch_size)
        # self.fusions = Dual_Fusion_New()#with_cp=True)  

    def extract_feat(self, batch_inputs: Tensor,
                    batch_extra_inputs: Tensor) -> Tuple[Tensor]:
        x_V = self.backbone[0](batch_inputs)
        x_I = self.backbone[1](batch_extra_inputs[0])
        x = [x_V1 + x_I1 for x_V1, x_I1 in zip(x_V, x_I)]
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

        losses = self.bbox_head.loss(
            img_feats, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs,
                batch_data_samples, batch_extra_inputs, rescale: bool = True):

        # image feature extraction
        img_feats = self.extract_feat(batch_inputs, batch_extra_inputs)
        results_list = self.bbox_head.predict(
            img_feats,
            batch_data_samples=batch_data_samples, 
            rescale=rescale)
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
