# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
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

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


# /workspace/mmdetection/mmdet/models/utils/vlfuse_helper.py
from ...utils.vlfuse_helper import BiAttentionBlock, BiMultiHeadAttention, permute_and_flatten
from mmcv.cnn.bricks import DropPath
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
MAX_CLAMP_VALUE = 50000

class MultiHeadAttention_New(nn.Module):

    def __init__(self,
                master_dim: int,
                slave_dim: int,
                embed_dim: int,
                num_heads: int,
                dropout: float = 0.1):
        super(MultiHeadAttention_New, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.master_dim = master_dim
        self.slave_dim = slave_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), 'embed_dim must be divisible by num_heads ' \
           f'(got `embed_dim`: {self.embed_dim} ' \
           f'and `num_heads`: {self.num_heads}).'
        self.scale = self.head_dim**(-0.5)
        self.dropout = dropout

        self.query_proj = nn.Linear(self.master_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.slave_dim, self.embed_dim)
        self.values_proj = nn.Linear(self.slave_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.master_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()
    
    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        self.query_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.key_proj.weight)
        self.key_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_proj.weight)
        self.values_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(
        self,
        master: Tensor,
        slave: Tensor,
        attention_mask: Optional[Tensor] = None,    # 代表的slave的mask
    ) -> Tuple[Tensor, Tensor]:
        bsz, tgt_len, _ = master.size()

        query_states = self.query_proj(master) * self.scale # (bsz, tgt_len, cv) -> (bsz, tgt_len, embed_dim)
        key_states = self._shape(self.key_proj(slave), -1, bsz)    # (bsz, nseq, cl) -> (bsz, nseq, embed_dim) -> (bsz, n_head, nseq, head_dim)
        value_states = self._shape(self.values_proj(slave), -1, bsz) # (bsz, nseq, cl) -> (bsz, nseq, embed_dim) -> (bsz, n_head, nseq, head_dim)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)   # (bsz, tgt_len, embed_dim) -> (bsz, n_head, tgt_len, head_dim) -> (bsz * n_head, tgt_len, head_dim)
        key_states = key_states.view(*proj_shape)   # (bsz, n_head, nseq, head_dim) -> (bsz * n_head, nseq, head_dim)
        value_states = value_states.view(*proj_shape)   # (bsz, n_head, nseq, head_dim) -> (bsz * n_head, nseq, head_dim)

        src_len = key_states.size(1)    # nseq
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # (bsz * n_head, tgt_len, nseq)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f'Attention weights should be of '
                f'size {(bsz * self.num_heads, tgt_len, src_len)}, '
                f'but is {attn_weights.size()}')

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            # Do not increase -50000, data type half has quite limited range
            attn_weights = torch.clamp(attn_weights, min=-MAX_CLAMP_VALUE)
        if self.clamp_max_for_overflow:
            # Do not increase 50000, data type half has quite limited range
            attn_weights = torch.clamp(attn_weights, max=MAX_CLAMP_VALUE)
        
        if attention_mask is not None:
            assert (attention_mask.dim() == 2)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError('Attention mask should be of '
                                 f'size {(bsz, 1, tgt_len, src_len)}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                                             src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                                             src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)    # (bsz * n_head, tgt_len, nseq)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training) # (bsz * n_head, tgt_len, nseq)
        attn_output = torch.bmm(attn_probs, value_states) # (bsz * n_head, tgt_len, head_dim)
        
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                '`attn_output_v` should be of '
                f'size {(bsz, self.num_heads, tgt_len, self.head_dim)}, '
                f'but is {attn_output.size()}')

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len,
                                           self.head_dim)   # (bsz * n_head, tgt_len, head_dim) -> (bsz, n_head, tgt_len, head_dim)
        attn_output = attn_output.transpose(1, 2)   # (bsz, n_head, tgt_len, head_dim) -> (bsz, tgt_len, n_head, head_dim)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim) # (bsz, tgt_len, n_head, head_dim) -> (bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output

class AttentionBlock(nn.Module):
    def __init__(self,
                master_dim: int,
                slave_dim: int,
                embed_dim: int,
                num_heads: int,
                dropout: float = 0.1,
                drop_path: float = .0,
                init_values: float = 1e-4):
        super(AttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_m = nn.LayerNorm(master_dim)
        self.layer_norm_s = nn.LayerNorm(slave_dim)
        self.attn = MultiHeadAttention_New(
            master_dim=master_dim,
            slave_dim=slave_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout)

        # add layer scale for training stability
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_m = nn.Parameter(
            init_values * torch.ones(master_dim), requires_grad=True)

    def forward(self,
                master_features: Tensor,
                slave_features: Tensor,
                master_level_start_index,
                slave_level_start_index,
                attention_mask=None):   # 代表的slave的mask
        assert master_level_start_index.shape == slave_level_start_index.shape
        
        new_master_features, new_slave_feature = [], []
        for i in range(master_level_start_index.shape[0]):
            master_index1 = master_level_start_index[i]
            slave_index1 = slave_level_start_index[i]
            master_index2 = master_level_start_index[i + 1] if i+1 < len(master_level_start_index) else master_features.shape[1]
            slave_index2 = slave_level_start_index[i + 1] if i+1 < len(slave_level_start_index) else slave_features.shape[1]

            # 逐层融合
            master_feature = master_features[:, master_index1:master_index2, :]
            slave_feature = slave_features[:, slave_index1:slave_index2, :]

            master_feature, slave_feature = self.single_attention_call(
                master_feature,
                slave_feature,
                attention_mask) 
            new_master_features.append(master_feature)
            new_slave_feature.append(slave_feature)

        return torch.cat(new_master_features, dim=1), torch.cat(new_slave_feature, dim=1)
    
    def single_attention_call(
        self,
        master_feature: Tensor,
        slave_feature: Tensor,
        attention_mask: Optional[Tensor] = None,    # 代表的slave的mask
    ) -> Tuple[Tensor, Tensor]:
        master_feature = self.layer_norm_m(master_feature)
        slave_feature = self.layer_norm_s(slave_feature)
        delta_m = self.attn(
            master=slave_feature,
            slave=slave_feature,
            attention_mask=attention_mask)
        master_feature = master_feature + self.drop_path(self.gamma_m * delta_m)
        return master_feature
    

class SingleScaleAttentionBlock(AttentionBlock):

    def forward(self,
                master_feature: Tensor,
                slave_feature: Tensor,
                attention_mask=None):   # 代表的slave的mask
        master_feature = self.single_attention_call(
            master_feature, # 这个是不是可以用conv1d减少query的数量，采用Unet那种思想，也很节约内存
            slave_feature,  # 这个是不是可以用conv1d减少query的数量
            attention_mask=attention_mask)
        return master_feature

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


class GroundingDinoTransformerEncoder_Fusion(DeformableDetrTransformerEncoder):

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
            (checkpoint_wrapper(DeformableDetrTransformerCrossLayer(**self.fusion_layer_cfg)) 
            if self.mode != 'tensor' else DeformableDetrTransformerCrossLayer(**self.fusion_layer_cfg))
            for _ in range(self.num_layers * 2)
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