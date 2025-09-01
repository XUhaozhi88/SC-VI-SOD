# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList, BaseModule
from torch import Tensor

from mmdet.models.utils.vlfuse_helper import (SingleScaleBiAttentionBlock, 
                                              BiAttentionBlock, BiMultiHeadAttention)
from mmdet.utils import ConfigType, OptConfigType
from .deformable_detr_layers import (DeformableDetrTransformerDecoderLayer,
                                     DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer)
from .detr_layers import DetrTransformerEncoderLayer
from .dino_layers import DinoTransformerDecoder
from .utils import MLP, get_text_sine_pos_embed

from .detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer)
from .grounding_dino_layers_fusion import SingleScaleAttentionBlock

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


# /workspace/mmdetection/mmdet/models/utils/vlfuse_helper.py
from ...utils.vlfuse_helper import BiAttentionBlock, BiMultiHeadAttention, permute_and_flatten
from mmcv.cnn.bricks import DropPath
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
from .new_ops import MultiScaleDeformableAttention_Local


class SmallAttentionBlock(DetrTransformerDecoderLayer):
    def __init__(self,
                self_attn_cfg: OptConfigType = dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                rgb_self_attn_cfg=None,
                ir_self_attn_cfg=None,
                cross_attn_cfg: OptConfigType = dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                num_cross_atten=1,
                ffn_cfg: OptConfigType = dict(
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True)),                
                 receptive_field_sizes: list = None,
                norm_cfg: OptConfigType = dict(type='LN'),
                init_cfg: OptConfigType = None) -> None:

        super(DetrTransformerDecoderLayer, self).__init__(init_cfg=init_cfg)
        self.receptive_field_sizes = receptive_field_sizes

        self.self_attn_cfg = self_attn_cfg
        self.rgb_self_attn_cfg = rgb_self_attn_cfg
        self.ir_self_attn_cfg = ir_self_attn_cfg
        assert (rgb_self_attn_cfg is None) == (ir_self_attn_cfg is None)
        
        self.cross_attn_cfg = cross_attn_cfg
        self.num_cross_atten = num_cross_atten
        assert num_cross_atten in [1, 2], \
            '1 means one fusion layer for rgb and ir, 2 means respectively fusion'

        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        num_layer = 1
        if self.rgb_self_attn_cfg is not None: num_layer += 1
        if self.ir_self_attn_cfg is not None: num_layer += 1
        self_attn_list = [
            MultiScaleDeformableAttention(**self.self_attn_cfg)
            for _ in range(num_layer)]
        cross_attn_list = [
            # MultiheadAttention(**self.cross_attn_cfg) 
            # MultiScaleDeformableAttention_Local(**self.cross_attn_cfg) 
            MultiScaleDeformableAttention(**self.cross_attn_cfg) 
            for _ in range(self.num_cross_atten)]
        self.self_attn = ModuleList(self_attn_list)
        self.cross_attn = ModuleList(cross_attn_list)

        ffn_list = [
            FFN(**self.ffn_cfg) for _ in range(num_layer)]
        self.ffn = ModuleList(ffn_list)

        self.embed_dims = self.self_attn[0].embed_dims
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        norms_list = ModuleList(norms_list)
        norms_list_list = [
            norms_list for _ in range(num_layer)]
        self.norms = ModuleList(norms_list_list)       
    
    def forward(self,
                # fusion
                query: Tensor, query_pos: Tensor, key_padding_mask: Tensor, reference_points: Tensor, 
                spatial_shapes: Tensor, level_start_index: Tensor, valid_ratios: Tensor, 
                # rgb
                rgb_query: Tensor, rgb_query_pos: Tensor, rgb_key_padding_mask: Tensor, rgb_reference_points: Tensor, 
                rgb_spatial_shapes: Tensor, rgb_level_start_index: Tensor, rgb_valid_ratios: Tensor, 
                # ir
                ir_query: Tensor, ir_query_pos: Tensor, ir_key_padding_mask: Tensor, ir_reference_points: Tensor, 
                ir_spatial_shapes: Tensor, ir_level_start_index: Tensor, ir_valid_ratios: Tensor,                 
                self_attn_mask: Tensor = None, cross_attn_mask: Tensor = None) -> Tensor:
        self.id, self.rgb_id, self.ir_id = 0, 1, 2

        # deformable self attention
        query = self.self_attn[self.id](
            query=query,
            # key=query,
            value=query,
            identity=query,
            query_pos=query_pos,
            # key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            # attn_mask=self_attn_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        query = self.norms[self.id][0](query)

        if self.rgb_self_attn_cfg is not None:
            rgb_query = self.self_attn[self.rgb_id](
                query=rgb_query,
                # key=rgb_query,
                value=rgb_query,
                identity=rgb_query,
                query_pos=rgb_query_pos,
                # key_pos=rgb_query_pos,
                key_padding_mask=rgb_key_padding_mask,
                reference_points=rgb_reference_points,
                # attn_mask=self_attn_mask,
                spatial_shapes=rgb_spatial_shapes,
                level_start_index=rgb_level_start_index)
            rgb_query = self.norms[self.rgb_id][0](rgb_query)

        if self.ir_self_attn_cfg is not None:
            ir_query = self.self_attn[self.ir_id](
                query=ir_query,
                # key=ir_query,
                value=ir_query,
                identity=rgb_query,
                query_pos=ir_query_pos,
                # key_pos=ir_query_pos,
                key_padding_mask=ir_key_padding_mask,
                reference_points=ir_reference_points,
                # attn_mask=self_attn_mask,
                spatial_shapes=ir_spatial_shapes,
                level_start_index=ir_level_start_index)
            ir_query = self.norms[self.ir_id][0](ir_query)
        
        # cross attention
        cross_attn_name = self.cross_attn[self.rgb_id - 1].__class__.__name__
        if cross_attn_name == 'MultiScaleDeformableAttention_Local':
            query = self.local_multiscale_deformableattention(
                # fusion
                query=query, query_pos=query_pos, level_start_index=level_start_index,  
                # rgb
                rgb_query=rgb_query, rgb_key_padding_mask=rgb_key_padding_mask, 
                rgb_reference_points=rgb_reference_points, rgb_spatial_shapes=rgb_spatial_shapes,
                rgb_level_start_index=rgb_level_start_index, rgb_valid_ratios=rgb_valid_ratios,
                # ir 
                ir_query=ir_query, ir_key_padding_mask=ir_key_padding_mask, 
                ir_reference_points=ir_reference_points, ir_spatial_shapes=ir_spatial_shapes,
                ir_level_start_index=ir_level_start_index, ir_valid_ratios=ir_valid_ratios,
            )
        elif cross_attn_name == 'MultiScaleDeformableAttention':
            query = self.multiscale_deformableattention(
                # fusion
                query=query, query_pos=query_pos,
                # rgb
                rgb_query=rgb_query, rgb_key_padding_mask=rgb_key_padding_mask, 
                rgb_reference_points=rgb_reference_points, rgb_spatial_shapes=rgb_spatial_shapes,
                rgb_level_start_index=rgb_level_start_index,
                # ir 
                ir_query=ir_query, ir_key_padding_mask=ir_key_padding_mask, 
                ir_reference_points=ir_reference_points, ir_spatial_shapes=ir_spatial_shapes,
                ir_level_start_index=ir_level_start_index,
            )
        elif cross_attn_name == 'MultiheadAttention':
            query = self.local_multihead_attention(
                query=query, query_pos=query_pos, spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,    # fusion
                rgb_query=rgb_query, rgb_query_pos=rgb_query_pos,   # rgb            
                ir_query=ir_query, ir_query_pos=ir_query_pos, # ir
                )            
        
        query = self.norms[self.id][1](query)
        query = self.ffn[self.id](query)
        query = self.norms[self.id][2](query)        
        if self.rgb_self_attn_cfg is not None:
            rgb_query = self.norms[self.rgb_id][1](rgb_query)
            rgb_query = self.ffn[self.rgb_id](rgb_query)
            rgb_query = self.norms[self.rgb_id][2](rgb_query)    
        if self.ir_self_attn_cfg is not None:
            ir_query = self.norms[self.ir_id][1](ir_query)
            ir_query = self.ffn[self.ir_id](ir_query)
            ir_query = self.norms[self.ir_id][2](ir_query)  
        return query, rgb_query, ir_query
    
    def local_multiscale_deformableattention(self,
                # fusion
                query: Tensor, query_pos: Tensor, level_start_index: Tensor,
                # rgb
                rgb_query: Tensor, rgb_key_padding_mask: Tensor, rgb_reference_points: Tensor, 
                rgb_spatial_shapes: Tensor, rgb_level_start_index: Tensor, rgb_valid_ratios: Tensor,
                # ir
                ir_query: Tensor, ir_key_padding_mask: Tensor, ir_reference_points: Tensor, 
                ir_spatial_shapes: Tensor, ir_level_start_index: Tensor, ir_valid_ratios: Tensor
                ) -> Tensor:
        assert self.num_cross_atten == 2, 'local multi-scale deformable attention must be respectively calculated.'
        query = self.cross_attn[self.rgb_id - 1](
                query=query,
                # key=rgb_query,
                value=rgb_query,
                identity=query,
                query_pos=query_pos,
                key_padding_mask=rgb_key_padding_mask,
                reference_points=rgb_reference_points,  # 用哪个reference_points
                valid_ratios=rgb_valid_ratios,          # 和reference_points一致，但是感觉是对应value
                spatial_shapes=rgb_spatial_shapes,      # 对应num_value
                query_level_start_index=level_start_index,  # 对应num_query
                value_level_start_index=rgb_level_start_index,   # 对应num_value
                receptive_field_sizes=self.receptive_field_sizes
                )    
        query = self.cross_attn[self.ir_id - 1](
                query=query,
                # key=ir_query,
                value=ir_query,
                identity=query,
                query_pos=query_pos,
                key_padding_mask=ir_key_padding_mask,
                reference_points=ir_reference_points,
                valid_ratios=ir_valid_ratios,          # 和reference_points一致，但是感觉是对应value
                spatial_shapes=ir_spatial_shapes,      # 对应num_value
                query_level_start_index=level_start_index,  # 对应num_query
                value_level_start_index=ir_level_start_index,   # 对应num_value
                receptive_field_sizes=self.receptive_field_sizes
                )
        return query
    
    def multiscale_deformableattention(self,
                # fusion
                query: Tensor, query_pos: Tensor,
                # rgb
                rgb_query: Tensor, rgb_key_padding_mask: Tensor, rgb_reference_points: Tensor, 
                rgb_spatial_shapes: Tensor, rgb_level_start_index: Tensor,
                # ir
                ir_query: Tensor, ir_key_padding_mask: Tensor, ir_reference_points: Tensor, 
                ir_spatial_shapes: Tensor, ir_level_start_index: Tensor
                ) -> Tensor:
        assert self.num_cross_atten == 2, 'multi-scale deformable attention must be respectively calculated.'
        query = self.cross_attn[self.rgb_id - 1](
                query=query,
                value=rgb_query,
                identity=query,
                query_pos=query_pos,
                key_padding_mask=rgb_key_padding_mask,
                reference_points=rgb_reference_points,  # 用哪个reference_points
                spatial_shapes=rgb_spatial_shapes,
                level_start_index=rgb_level_start_index)
        query = self.cross_attn[self.ir_id - 1](
                query=query,
                value=ir_query,
                identity=query,
                query_pos=query_pos,
                key_padding_mask=ir_key_padding_mask,
                reference_points=ir_reference_points,
                spatial_shapes=ir_spatial_shapes,
                level_start_index=ir_level_start_index)
        return query
        
    def local_multihead_attention(self,            
            query: Tensor, query_pos: Tensor, spatial_shapes: Tensor, level_start_index: Tensor,    # fusion
            rgb_query: Tensor, rgb_query_pos: Tensor,   # rgb            
            ir_query: Tensor, ir_query_pos: Tensor, # ir
            cross_attn_mask: Tensor = None):
        
        # partial spatial cross attention (for small)
        bs, _, dim = query.shape
        output_query = []
        for i in range(len(level_start_index)):
            h, w = spatial_shapes[i]
            ks = self.receptive_field_sizes[i]
            index1 = level_start_index[i]
            index2 = level_start_index[i + 1] if i + 1 < len(level_start_index) else query.shape[1]

            # 逐层处理 query 和 query_pos
            layer_query = query[:, index1:index2, :].reshape(bs*h*w, 1, dim)
            layer_query_pos = query_pos[:, index1:index2, :].reshape(bs*h*w, 1, dim)

            # 直接处理 rgb 和 ir 的 unfold 操作，减少临时存储
            rgb_layer_query = rgb_query[:, index1:index2, :].reshape(bs, h, w, dim).permute(0, 3, 1, 2)
            rgb_layer_query_pos = rgb_query_pos[:, index1:index2, :].reshape(bs, h, w, dim).permute(0, 3, 1, 2)
            ir_layer_query = ir_query[:, index1:index2, :].reshape(bs, h, w, dim).permute(0, 3, 1, 2)
            ir_layer_query_pos = ir_query_pos[:, index1:index2, :].reshape(bs, h, w, dim).permute(0, 3, 1, 2)

            # 使用 unfold 提前展平，不需要分别创建中间变量
            rgb_layer_query = F.unfold(rgb_layer_query, kernel_size=ks, padding=(ks - 1) // 2, stride=1)
            rgb_layer_query_pos = F.unfold(rgb_layer_query_pos, kernel_size=ks, padding=(ks - 1) // 2, stride=1)
            ir_layer_query = F.unfold(ir_layer_query, kernel_size=ks, padding=(ks - 1) // 2, stride=1)
            ir_layer_query_pos = F.unfold(ir_layer_query_pos, kernel_size=ks, padding=(ks - 1) // 2, stride=1)

            # 展平 unfolded 结果
            rgb_layer_query = rgb_layer_query.view(bs, dim, ks*ks, h*w).transpose(1, 3).reshape(bs*h*w, ks*ks, dim)
            rgb_layer_query_pos = rgb_layer_query_pos.view(bs, dim, ks*ks, h*w).transpose(1, 3).reshape(bs*h*w, ks*ks, dim)
            ir_layer_query = ir_layer_query.view(bs, dim, ks*ks, h*w).transpose(1, 3).reshape(bs*h*w, ks*ks, dim)
            ir_layer_query_pos = ir_layer_query_pos.view(bs, dim, ks*ks, h*w).transpose(1, 3).reshape(bs*h*w, ks*ks, dim)

            layer_key_padding_mask=None #if key_padding_mask is None else key_padding_mask[:, index1:index2].reshape(bs*h*w, 1).repeat(1, ks*ks)
            
            # if key_padding_mask is None:
            #     rgb_layer_key_padding_mask = None
            #     ir_layer_key_padding_mask = None
            # else:
            #     rgb_layer_key_padding_mask = rgb_key_padding_mask[:, index1:index2].reshape(bs, h, w, 1).permute(0, 3, 1, 2).to(torch.float)
            #     ir_layer_key_padding_mask = ir_key_padding_mask[:, index1:index2].reshape(bs, h, w, 1).permute(0, 3, 1, 2).to(torch.float)

            #     rgb_layer_key_padding_mask = F.unfold(rgb_layer_key_padding_mask, kernel_size=ks, padding=(ks - 1) // 2, stride=1)
            #     ir_layer_key_padding_mask = F.unfold(ir_layer_key_padding_mask, kernel_size=ks, padding=(ks - 1) // 2, stride=1)

            #     rgb_layer_key_padding_mask = rgb_layer_key_padding_mask.transpose(1, 2).reshape(bs*h*w, ks*ks).to(torch.bool)
            #     ir_layer_key_padding_mask = ir_layer_key_padding_mask.transpose(1, 2).reshape(bs*h*w, ks*ks).to(torch.bool)
        
            if self.num_cross_atten == 1:
                # 一起融合
                layer_query = self.cross_attn[self.id](
                    query=layer_query,
                    key=torch.cat((rgb_layer_query, ir_layer_query), dim=1),
                    value=torch.cat((rgb_layer_query, ir_layer_query), dim=1),
                    identity=layer_query,
                    query_pos=layer_query_pos,
                    key_pos=torch.cat((rgb_layer_query_pos, ir_layer_query_pos), dim=1),
                    attn_mask=cross_attn_mask,
                    key_padding_mask=layer_key_padding_mask)
                    # key_padding_mask=key_padding_mask if key_padding_mask is None else \
                    #     torch.cat((rgb_layer_key_padding_mask, ir_layer_key_padding_mask), dim=1))
            elif self.num_cross_atten == 2:
                # 分开融合
                layer_query = self.cross_attn[self.rgb_id - 1](
                    query=layer_query,
                    key=rgb_layer_query,
                    value=rgb_layer_query,
                    identity=layer_query,
                    query_pos=layer_query_pos,
                    key_pos=rgb_layer_query_pos,
                    attn_mask=cross_attn_mask,
                    key_padding_mask=layer_key_padding_mask)
                    # key_padding_mask=rgb_layer_key_padding_mask)
                layer_query = self.cross_attn[self.ir_id - 1](
                    query=layer_query,
                    key=ir_layer_query,
                    value=ir_layer_query,
                    identity=layer_query,
                    query_pos=layer_query_pos,
                    key_pos=ir_layer_query_pos,
                    attn_mask=cross_attn_mask,
                    key_padding_mask=layer_key_padding_mask)
                    # key_padding_mask=ir_layer_key_padding_mask)
            output_query.append(layer_query.view(bs, h*w, dim))
        query = torch.cat(output_query, dim=1)
        return query
    
    

class DeformableDetrTransformerCrossLayer(DetrTransformerDecoderLayer):

    def __init__(self,
                cross_attn_cfg: OptConfigType = dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg: OptConfigType = dict(
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                norm_cfg: OptConfigType = dict(type='LN'),
                init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.cross_attn_cfg = cross_attn_cfg
        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.cross_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)
    
    def forward(self,
            query: Tensor,
            key: Tensor = None,
            value: Tensor = None,
            query_pos: Tensor = None,
            key_pos: Tensor = None,
            cross_attn_mask: Tensor = None,
            key_padding_mask: Tensor = None,    # from value
            spatial_shapes: Tensor = None,      # from value
            level_start_index: Tensor = None,   # from value
            reference_points: Tensor = None,    # from value
            ) -> Tensor:

        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points)
        
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query


class DinoTransformerEncoder_FusionNew(DeformableDetrTransformerEncoder):

    def __init__(self,
                 layer_cfg,
                mode='loss',
                extra_return=None,
                **kwargs) -> None:
        self.mode = mode
        self.extra_return = extra_return      
        if 'rgb' in extra_return:
            layer_cfg['rgb_self_attn_cfg'] = layer_cfg['self_attn_cfg']
        if 'ir' in extra_return:
            layer_cfg['ir_self_attn_cfg'] = layer_cfg['self_attn_cfg']
        super().__init__(layer_cfg=layer_cfg, **kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""  
        self.layers = ModuleList([            
            (checkpoint_wrapper(SmallAttentionBlock(**self.layer_cfg))
            if self.mode != 'tensor' else SmallAttentionBlock(**self.layer_cfg))
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self,
                # fusion
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                # rgb
                rgb_query: Tensor,
                rgb_query_pos: Tensor,
                rgb_key_padding_mask: Tensor,
                rgb_spatial_shapes: Tensor,
                rgb_level_start_index: Tensor,
                rgb_valid_ratios: Tensor,
                # ir
                ir_query: Tensor,
                ir_query_pos: Tensor,
                ir_key_padding_mask: Tensor,
                ir_spatial_shapes: Tensor,
                ir_level_start_index: Tensor,
                ir_valid_ratios: Tensor):
        output = query
        rgb_output = rgb_query
        ir_output = ir_query

        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        rgb_reference_points = self.get_encoder_reference_points(
            rgb_spatial_shapes, rgb_valid_ratios, device=query.device)
        ir_reference_points = self.get_encoder_reference_points(
            ir_spatial_shapes, ir_valid_ratios, device=query.device)

        # main process
        for layer in self.layers:            
            output, rgb_output, ir_output = layer(
                # fusion
                query=output,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                # rgb
                rgb_query=rgb_output,
                rgb_query_pos=rgb_query_pos,
                rgb_key_padding_mask=rgb_key_padding_mask,
                rgb_reference_points=rgb_reference_points,
                rgb_spatial_shapes=rgb_spatial_shapes,
                rgb_level_start_index=rgb_level_start_index,
                rgb_valid_ratios=rgb_valid_ratios,
                # ir
                ir_query=ir_output,
                ir_query_pos=ir_query_pos,
                ir_key_padding_mask=ir_key_padding_mask,
                ir_reference_points=ir_reference_points,
                ir_spatial_shapes=ir_spatial_shapes,
                ir_level_start_index=ir_level_start_index,
                ir_valid_ratios=ir_valid_ratios)  

        return_dict = dict()        
        return_dict['fusion'] = output
        if 'rgb' in self.extra_return:
            return_dict['rgb'] = rgb_output
        if 'ir' in self.extra_return:
            return_dict['ir'] = ir_output
        return return_dict


class DinoTransformerEncoder_Fusion(DeformableDetrTransformerEncoder):

    def __init__(self, 
                num_rgb_layers=6,
                num_ir_layers=6,
                fusion_layer_cfg: ConfigType = None,
                rgb_layer_cfg: ConfigType = None, 
                ir_layer_cfg: ConfigType = None,
                mode='loss',
                extra_return=None,
                **kwargs) -> None:
        self.num_rgb_layers = num_rgb_layers
        self.num_ir_layers = num_ir_layers

        self.rgb_layer_cfg = rgb_layer_cfg
        self.ir_layer_cfg = ir_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.mode = mode
        self.extra_return = extra_return
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""        
        self.layers = ModuleList([            
            (checkpoint_wrapper(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
            if self.mode != 'tensor' else DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
            for _ in range(self.num_layers)
        ])
        self.rgb_layers = None if self.rgb_layer_cfg is None else \
            ModuleList([
                (checkpoint_wrapper(DeformableDetrTransformerEncoderLayer(**self.rgb_layer_cfg))
                if self.mode != 'tensor' else DeformableDetrTransformerEncoderLayer(**self.rgb_layer_cfg))
                for _ in range(self.num_rgb_layers)
            ])
        self.ir_layers = None if self.ir_layer_cfg is None else \
            ModuleList([
                (checkpoint_wrapper(DeformableDetrTransformerEncoderLayer(**self.ir_layer_cfg))            
                if self.mode != 'tensor' else DeformableDetrTransformerEncoderLayer(**self.ir_layer_cfg))
                for _ in range(self.num_ir_layers)
            ])
        self.fusion_layers = ModuleList([
            # checkpoint_wrapper(MultiheadAttention(**self.fusion_layer_cfg))
            # checkpoint_wrapper(SingleScaleBiAttentionBlock(**self.fusion_layer_cfg))
            # checkpoint_wrapper(SingleScaleAttentionBlock(**self.fusion_layer_cfg))
            # checkpoint_wrapper(SmallAttentionBlock(**self.fusion_layer_cfg))
            (checkpoint_wrapper(SmallAttentionBlock(**self.fusion_layer_cfg)) 
            if self.mode != 'tensor' else SmallAttentionBlock(**self.fusion_layer_cfg))
            for _ in range(self.num_layers)
            # (checkpoint_wrapper(DeformableDetrTransformerCrossLayer(**self.fusion_layer_cfg)) 
            # if self.mode != 'tensor' else DeformableDetrTransformerCrossLayer(**self.fusion_layer_cfg))
            # for _ in range(self.num_layers * 2)
        ])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self,
                # fusion
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                # rgb
                rgb_query: Tensor,
                rgb_query_pos: Tensor,
                rgb_key_padding_mask: Tensor,
                rgb_spatial_shapes: Tensor,
                rgb_level_start_index: Tensor,
                rgb_valid_ratios: Tensor,
                # ir
                ir_query: Tensor,
                ir_query_pos: Tensor,
                ir_key_padding_mask: Tensor,
                ir_spatial_shapes: Tensor,
                ir_level_start_index: Tensor,
                ir_valid_ratios: Tensor):
        output = query
        rgb_output = rgb_query
        ir_output = ir_query

        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        rgb_reference_points = self.get_encoder_reference_points(
            rgb_spatial_shapes, rgb_valid_ratios, device=query.device)
        ir_reference_points = self.get_encoder_reference_points(
            ir_spatial_shapes, ir_valid_ratios, device=query.device)

        # main process
        for layer_id, layer in enumerate(self.layers):            
            if self.fusion_layers:
                fusion_layer_name = self.fusion_layers[0].__class__.__name__
                if fusion_layer_name == 'MultiheadAttention':
                    output = self.fusion_layers[layer_id * 2](
                        query=output,
                        key=rgb_output,
                        value=rgb_output,
                        query_pos=query_pos,
                        key_pos=rgb_query_pos,
                        attn_mask=None,
                        key_padding_mask=rgb_key_padding_mask)
                    output = self.fusion_layers[layer_id * 2 + 1](
                        query=output,
                        key=ir_output,
                        value=ir_output,
                        query_pos=query_pos,
                        key_pos=ir_query_pos,
                        attn_mask=None,
                        key_padding_mask=key_padding_mask)
                elif fusion_layer_name == 'SingleScaleBiAttentionBlock':
                    output, rgb_output = self.fusion_layers[layer_id * 2](
                        visual_feature=output,
                        lang_feature=rgb_output,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=rgb_key_padding_mask)
                    output, ir_output = self.fusion_layers[layer_id * 2 + 1](
                        visual_feature=output,
                        lang_feature=ir_output,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=ir_key_padding_mask)
                elif fusion_layer_name == 'SingleScaleAttentionBlock':
                    output = self.fusion_layers[layer_id * 2](
                        master_feature=output,
                        slave_feature=rgb_output,
                        attention_mask=rgb_key_padding_mask)
                    output = self.fusion_layers[layer_id * 2 + 1](
                        master_feature=output,
                        slave_feature=ir_output,
                        attention_mask=ir_key_padding_mask)
                elif fusion_layer_name == 'DeformableDetrTransformerCrossLayer':
                    output = self.fusion_layers[layer_id * 2](
                        query=output,
                        key=rgb_output,
                        value=rgb_output,
                        query_pos=query_pos,
                        key_pos=rgb_query_pos,
                        reference_points=rgb_reference_points,  
                        spatial_shapes=rgb_spatial_shapes,  
                        level_start_index=rgb_level_start_index,
                        key_padding_mask=rgb_key_padding_mask)
                    output = self.fusion_layers[layer_id * 2 + 1](
                        query=output,
                        key=ir_output,
                        value=ir_output,
                        query_pos=query_pos,
                        key_pos=ir_query_pos,
                        reference_points=ir_reference_points,
                        spatial_shapes=ir_spatial_shapes,
                        level_start_index=ir_level_start_index,
                        key_padding_mask=ir_key_padding_mask)
                else:
                    raise AssertionError
                
            if (self.rgb_layers is not None) and (layer_id < self.num_rgb_layers):   
                rgb_output = self.rgb_layers[layer_id](
                    query=rgb_output,
                    query_pos=rgb_query_pos,
                    reference_points=rgb_reference_points,
                    spatial_shapes=rgb_spatial_shapes,
                    level_start_index=rgb_level_start_index,
                    key_padding_mask=rgb_key_padding_mask)
            if (self.ir_layers is not None) and (layer_id < self.num_ir_layers):   
                ir_output = self.ir_layers[layer_id](
                    query=ir_output,
                    query_pos=ir_query_pos,
                    reference_points=ir_reference_points,
                    spatial_shapes=ir_spatial_shapes,
                    level_start_index=ir_level_start_index,
                    key_padding_mask=ir_key_padding_mask)
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)    

        return_dict = dict()        
        return_dict['fusion'] = output
        if 'rgb' in self.extra_return:
            return_dict['rgb'] = rgb_output
        if 'ir' in self.extra_return:
            return_dict['ir'] = ir_output
        return return_dict

    
class DinoTransformerDecoder_New(DinoTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""
    def __init__(self, mode = 'loss', **kwargs) -> None:
        self.mode = mode
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            (checkpoint_wrapper(DeformableDetrTransformerDecoderLayer(**self.layer_cfg)) 
            if self.mode != 'tensor' else DeformableDetrTransformerDecoderLayer(**self.layer_cfg))
            for _ in range(self.num_layers)
        ])

        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')        
        
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)