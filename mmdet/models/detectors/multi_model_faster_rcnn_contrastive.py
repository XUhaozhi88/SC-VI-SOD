# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn, Tensor
from torch.nn import functional as F

import copy
from typing import Tuple, List

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from .faster_rcnn import FasterRCNN

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, mid_channel=None, out_channel=None) -> None:
        super().__init__()
        mid_channel = in_channel if mid_channel is None else mid_channel
        out_channel = in_channel if out_channel is None else out_channel

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 计算权重
        self.weight = nn.Sequential(
            nn.Linear(in_channel, mid_channel, bias=False),
            nn.ReLU(),
            nn.Linear(mid_channel, out_channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        B, C, _, _ = x.shape
        weight = self.gap(x).view(B, C)
        weight = self.weight(weight)
        return x * weight.unsqueeze(-1).unsqueeze(-1)

@MODELS.register_module()
class MultiModelFasterRCNN_Contrastive(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 *args,
                 mod_list=["img", "ir_img"],
                 con_mode=['easy', 'hard'],
                 is_channel_align=False,
                 fusion_mode='sum',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # init super params
        self.mod_list = mod_list
        self.con_mode = con_mode
        self.is_channel_align = is_channel_align
        self.fusion_mode = fusion_mode

        # 初始化结构
        backbone = nn.ModuleList()
        if self.with_neck: neck = nn.ModuleList()
        for _ in range(len(self.mod_list)):
            backbone.append(copy.deepcopy(self.backbone))
            if self.with_neck: neck.append(copy.deepcopy(self.neck))
        self.backbone = backbone
        if self.with_neck: self.neck = neck
        
        # 通道对齐
        if self.is_channel_align:
            self.channel_alignment = nn.ModuleList()
            for _ in range(5):
                self.channel_alignment.append(ChannelAttention(in_channel=256, mid_channel=64))
            self.init_weights_other(self.channel_alignment)
        
        # 多尺度特征映射对齐
        self.project_module = nn.ModuleList()
        for _ in range(5):
            project_module = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU())
            # project_module = nn.Sequential(
            #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            #     nn.ReLU(),
            #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=256),
            #     nn.ReLU())
            self.project_module.append(project_module)
        self.init_weights_other(self.project_module)

        # 多尺度多模态融合
        if 'conv' in self.fusion_mode:
            self.fusion_module = nn.ModuleList()
            for _ in range(5):
                if self.fusion_mode == 'conv':
                    fusion_module = nn.Sequential(
                        nn.Conv2d(in_channels=256 * len(self.mod_list), out_channels=256, kernel_size=1),
                        nn.ReLU())
                elif self.fusion_mode == 'group_conv':
                    fusion_module = nn.Sequential(
                        nn.Conv2d(in_channels=256 * len(self.mod_list), out_channels=256, kernel_size=1, groups=256, bias=False),   # group conv    而且这里感觉有一种逐层加权的感觉，是否要把bias=False，参考Channel Attention
                        nn.ReLU())
                self.fusion_module.append(fusion_module)
            self.init_weights_other(self.fusion_module)

    def init_weights_other(self, module) -> None:
        """Initialize weights for other components."""
        for submodule in module.modules():
            if isinstance(submodule, nn.Conv2d):
                # 对卷积层使用 Kaiming 初始化
                nn.init.kaiming_normal_(submodule.weight, mode='fan_out', nonlinearity='relu')
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.Linear):
                # 对线性层使用 Xavier 初始化
                nn.init.xavier_normal_(submodule.weight)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.BatchNorm2d):
                # 对 BatchNorm 层的 weight 设置为1，bias 设置为0
                nn.init.constant_(submodule.weight, 1)
                nn.init.constant_(submodule.bias, 0)

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

    def contrast_loss_single(self, main_feats, pos_feats, neg_feats, pos_margin=0.2, neg_margin=0.2):
        def contrastive_loss(cos_distance, label, margin):
            """
            :param cos_distance: 余弦距离 (batch_size,)
            :param label: 样本对的标签 (batch_size,) 1表示正样本，0表示负样本
            :param margin: 负样本的边界阈值(0.2~1.0)
            :return: 对比损失
            """
            loss_pos = label * cos_distance**2
            loss_neg = (1 - label) * torch.max(torch.zeros_like(cos_distance), margin - cos_distance)**2
            loss = torch.mean(loss_pos + loss_neg)
            return loss
        
        loss = 0
        # for main_feat, pos_feat, neg_feat \
        #     in zip(main_feats, pos_feats, neg_feats):
        for project_module, main_feat, pos_feat, neg_feat \
            in zip(self.project_module, main_feats, pos_feats, neg_feats):
            main_feat = project_module(main_feat)
            pos_feat = project_module(pos_feat)
            neg_feat = project_module(neg_feat)
            
            # Ensure the features are in the correct shape: (B, C, H, W)
            B, C, _, _ = main_feat.shape
            main_feat = main_feat.view(B, C, -1).permute(0, 2, 1)  # Shape: (B, H * W, C)
            pos_feat = pos_feat.view(B, C, -1).permute(0, 2, 1)
            neg_feat = neg_feat.view(B, C, -1).permute(0, 2, 1)        

            # Compute cosine similarity pixel-wise
            pos_cos_sim = F.cosine_similarity(main_feat, pos_feat, dim=1)  # Similarity along channel dimension (C)
            neg_cos_sim = F.cosine_similarity(main_feat, neg_feat, dim=1)

            # Compute loss per pixel (H * W)
            pos_loss = contrastive_loss(pos_cos_sim, label=1, margin=pos_margin)  # Contrastive loss for positive pairs
            neg_loss = contrastive_loss(neg_cos_sim, label=0, margin=neg_margin)  # Contrastive loss for negative pairs

            # Aggregate losses across all pixels
            loss += (pos_loss + neg_loss) #* 0.1
            # loss.append((pos_loss + neg_loss)) #* 0.1
        return loss

    def contrast_loss(self, main_feats, pos_feats, neg_feats_dict):        
        losses_dict = {}
        for con_mode, neg_feats in neg_feats_dict.items():            
            if con_mode == 'easy':
                loss = self.contrast_loss_single(main_feats, pos_feats, neg_feats, neg_margin=1.0)
            elif con_mode == 'hard':                
                loss = self.contrast_loss_single(main_feats, pos_feats, neg_feats, neg_margin=0.2)
            else:
                raise AssertionError
            losses_dict[f'{con_mode}_contrast_loss'] = loss

        return losses_dict

    def extract_feat_CA_before(self, batch_inputs: Tensor,
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
            # # 特征映射            
            # x = [project_module(x1) for x1, project_module in zip(x, self.project_module)]       
            # 通道对齐            
            if self.is_channel_align:
                x = [channel_alignment(x1) for x1, channel_alignment in zip(x, self.channel_alignment)]       
            xs.append(x)

        # 这里就是用于对比学习的ir图像特征
        if self.training:
            con_x = {}
            contrast_inputs = batch_extra_inputs[i]
            assert len(contrast_inputs) == len(self.con_mode)
            for con_mode, contrast_input in zip(self.con_mode, contrast_inputs):
                x = backbone(contrast_input)       
                if self.with_neck: x = neck(x)    
                # # 特征映射            
                # x = [project_module(x1) for x1, project_module in zip(x, self.project_module)]
                # 通道对齐                
                if self.is_channel_align:
                    x = [channel_alignment(x1) for x1, channel_alignment in zip(x, self.channel_alignment)]
                con_x[con_mode] = x
                
        # 融合每个特征层，并使用 fusion_module 处理
        if self.fusion_mode == 'sum':
            # sum
            out = [torch.sum(torch.stack([x[i] for x in xs], dim=0), dim=0)
                for i in range(len(xs[0]))]
        elif self.fusion_mode == 'conv':
            # conv      
            out = [fusion_module(torch.cat([x[i] for x in xs], dim=1)) 
                for i, fusion_module in enumerate(self.fusion_module)]
        elif self.fusion_mode == 'group_conv':          
            # group conv
            # only fuse every mod's feature maps in per channel
            out = []    
            for i, fusion_module in enumerate(self.fusion_module):
                x1 = torch.stack([x[i] for x in xs], dim=2) # num_mod * [B, C, H, W] -> [B, C, num_mod, H, W]
                B, _, _, H, W = x1.shape
                out.append(fusion_module(x1.view(B, -1, H, W))) 

        if self.training:
            return out, (*xs, con_x)
        else:
            return out
        
    def extract_feat(self, batch_inputs: Tensor,
                     batch_extra_inputs: Tensor) -> Tuple[Tensor]:
        xs = []
        con_xs = []
        for i, backbone in enumerate(self.backbone):
            # 选择输入：第一个模态使用 batch_inputs，其余模态使用 batch_extra_inputs
            inputs = batch_inputs if i == 0 else batch_extra_inputs[i - 1]

            # 提取 backbone 特征
            x = backbone(inputs)            
            if self.with_neck: 
                neck = self.neck[i]
                x = neck(x)         
            # # 特征映射            
            # x = [project_module(x1) for x1, project_module in zip(x, self.project_module)]    
            con_xs.append(x)    # before channel attention
            # 通道对齐            
            if self.is_channel_align:
                x = [channel_alignment(x1) for x1, channel_alignment in zip(x, self.channel_alignment)]       
            xs.append(x)

        # 这里就是用于对比学习的ir图像特征
        if self.training:
            con_x = {}
            contrast_inputs = batch_extra_inputs[i]
            assert len(contrast_inputs) == len(self.con_mode)
            for con_mode, contrast_input in zip(self.con_mode, contrast_inputs):
                x = backbone(contrast_input)       
                if self.with_neck: x = neck(x)    
                # # 特征映射            
                # x = [project_module(x1) for x1, project_module in zip(x, self.project_module)]
                # 通道对齐                
                if self.is_channel_align:
                    x = [channel_alignment(x1) for x1, channel_alignment in zip(x, self.channel_alignment)]
                con_x[con_mode] = x
                
        # 融合每个特征层，并使用 fusion_module 处理
        if self.fusion_mode == 'sum':
            # sum
            out = [torch.sum(torch.stack([x[i] for x in xs], dim=0), dim=0)
                for i in range(len(xs[0]))]
        elif self.fusion_mode == 'conv':
            # conv      
            out = [fusion_module(torch.cat([x[i] for x in xs], dim=1)) 
                for i, fusion_module in enumerate(self.fusion_module)]
        elif self.fusion_mode == 'group_conv':          
            # group conv
            # only fuse every mod's feature maps in per channel
            out = []    
            for i, fusion_module in enumerate(self.fusion_module):
                x1 = torch.stack([x[i] for x in xs], dim=2) # num_mod * [B, C, H, W] -> [B, C, num_mod, H, W]
                B, _, _, H, W = x1.shape
                out.append(fusion_module(x1.view(B, -1, H, W))) 

        if self.training:
            return out, (*con_xs, con_x)
        else:
            return out
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             batch_extra_inputs: Tensor) -> dict:
        
        x, contrast_input = self.extract_feat(batch_inputs, batch_extra_inputs)
        contrast_loss = self.contrast_loss(*contrast_input)

        losses = dict()
        losses.update(contrast_loss)

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

