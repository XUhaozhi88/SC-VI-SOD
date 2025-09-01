# Author: Xu Haozhi.
# Time:   2025.05.13

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.utils import OptConfigType
from .deformable_detr_layers import DeformableDetrTransformerEncoder

from .detr_layers import DetrTransformerDecoderLayer
from .new_ops import MultiScaleDeformableAttention_Local

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class ParallelAttentionBlock(DetrTransformerDecoderLayer):
    def __init__(self,
                 extra_return=None,
                self_attn_cfg: OptConfigType = dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
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
                    act_cfg=dict(type='ReLU', inplace=True)),                
                 receptive_field_sizes: list = None,
                norm_cfg: OptConfigType = dict(type='LN'),
                init_cfg: OptConfigType = None) -> None:

        super(DetrTransformerDecoderLayer, self).__init__(init_cfg=init_cfg)
        self.receptive_field_sizes = receptive_field_sizes
        self.extra_return = extra_return
        self.self_attn_cfg = self_attn_cfg        
        self.cross_attn_cfg = cross_attn_cfg

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
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        # self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
        # self.cross_attn = MultiScaleDeformableAttention_Local(**self.cross_attn_cfg) 
        # self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.cross_attn = torch.nn.Identity()
        self.ffn = FFN(**self.ffn_cfg)
        self.norms = ModuleList([
                build_norm_layer(self.norm_cfg, self.embed_dims)[1]
                for _ in range(3)])        

        if 'ir' in self.extra_return:
            self.ir_self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
            # self.ir_cross_attn = MultiheadAttention(**self.cross_attn_cfg)
            # self.ir_cross_attn = MultiScaleDeformableAttention_Local(**self.cross_attn_cfg) 
            # self.ir_cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
            self.ir_cross_attn = torch.nn.Identity()
            self.ir_ffn = FFN(**self.ffn_cfg)
            self.ir_norms = ModuleList([
                build_norm_layer(self.norm_cfg, self.embed_dims)[1]
                for _ in range(3)])        
    
    def forward(self,
                # rgb
                query: Tensor, query_pos: Tensor, reference_points: Tensor, 
                # ir
                ir_query: Tensor, ir_query_pos: Tensor, ir_reference_points: Tensor, 
                # all
                key_padding_mask: Tensor, spatial_shapes: Tensor, level_start_index: Tensor, valid_ratios: Tensor,              
                self_attn_mask: Tensor = None, cross_attn_mask: Tensor = None) -> Tensor:

        # deformable self attention
        query = self.self_attn(
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
        query = self.norms[0](query)

        if 'ir' in self.extra_return:
            ir_query = self.ir_self_attn(
                query=ir_query,
                # key=ir_query,
                value=ir_query,
                identity=ir_query,
                query_pos=ir_query_pos,
                # key_pos=ir_query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=ir_reference_points,
                # attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)
            ir_query = self.ir_norms[0](ir_query)
        
        # cross attention
        cross_attn_name = self.cross_attn.__class__.__name__
        if cross_attn_name == 'MultiScaleDeformableAttention_Local':
            query = self.local_multiscale_deformableattention(
                # rgb
                query=query, query_pos=query_pos, reference_points=reference_points, 
                # ir
                ir_query=ir_query, ir_query_pos=ir_query_pos, ir_reference_points=ir_reference_points, 
                # all
                key_padding_mask=key_padding_mask, spatial_shapes=spatial_shapes,
                level_start_index=level_start_index, valid_ratios=valid_ratios)
        elif cross_attn_name == 'MultiScaleDeformableAttention':
            query, ir_query = self.multiscale_deformableattention(
                # rgb
                query=query, query_pos=query_pos, reference_points=reference_points, 
                # ir
                ir_query=ir_query, ir_query_pos=ir_query_pos, ir_reference_points=ir_reference_points, 
                # all
                key_padding_mask=key_padding_mask, spatial_shapes=spatial_shapes, level_start_index=level_start_index)     
        else:
            query = self.cross_attn(query)
            query = self.norms[1](query)
            ir_query = self.ir_cross_attn(ir_query)
            ir_query = self.ir_norms[1](ir_query)
        
        query = self.ffn(query)
        query = self.norms[2](query)
        if 'ir' in self.extra_return:
            ir_query = self.ir_ffn(ir_query)
            ir_query = self.ir_norms[2](ir_query)
        else:
            ir_query = None
        return query, ir_query
    
    def local_multiscale_deformableattention(self,
                # rgb
                query: Tensor, query_pos: Tensor, reference_points: Tensor, 
                # ir
                ir_query: Tensor, ir_query_pos: Tensor, ir_reference_points: Tensor,
                # all
                key_padding_mask: Tensor, spatial_shapes: Tensor, level_start_index: Tensor, valid_ratios: Tensor
                ) -> Tensor:
        query = self.cross_attn(
                # rgb
                query=query,
                identity=query,
                query_pos=query_pos,
                query_level_start_index=level_start_index,  # 对应num_query
                # ir
                value=ir_query,
                key_padding_mask=key_padding_mask,
                reference_points=ir_reference_points,
                valid_ratios=valid_ratios,          # 和reference_points一致，但是感觉是对应value
                spatial_shapes=spatial_shapes,      # 对应num_value
                value_level_start_index=level_start_index,   # 对应num_value
                # all
                receptive_field_sizes=self.receptive_field_sizes)
        query = self.norms[1](query)
        ir_query = self.ir_cross_attn(
                # ir
                query=ir_query,
                identity=ir_query,
                query_pos=ir_query_pos,
                query_level_start_index=level_start_index,  # 对应num_query
                # rgb
                value=query,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points,
                valid_ratios=valid_ratios,          # 和reference_points一致，但是感觉是对应value
                spatial_shapes=spatial_shapes,      # 对应num_value
                value_level_start_index=level_start_index,   # 对应num_value
                # all
                receptive_field_sizes=self.receptive_field_sizes)
        ir_query = self.ir_norms[1](ir_query)
        return query, ir_query
    
    def multiscale_deformableattention(self,
                # rgb
                query: Tensor, query_pos: Tensor, reference_points: Tensor, 
                # ir
                ir_query: Tensor, ir_query_pos: Tensor, ir_reference_points: Tensor,
                # all
                key_padding_mask: Tensor, spatial_shapes: Tensor, level_start_index: Tensor
                ) -> Tensor:
        query = self.cross_attn(
                # rgb
                query=query,
                identity=query,
                query_pos=query_pos,
                # ir
                value=ir_query,
                key_padding_mask=key_padding_mask,
                reference_points=ir_reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)
        query = self.norms[1](query)
        ir_query = self.ir_cross_attn(
                # ir
                query=ir_query,
                identity=ir_query,
                query_pos=ir_query_pos,
                # rgb
                value=query,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)
        ir_query = self.ir_norms[1](ir_query)
        return query, ir_query
        
    
    
class DinoTransformerEncoder_Parallel(DeformableDetrTransformerEncoder):

    def __init__(self,
                 layer_cfg,
                mode='tensor',
                extra_return=None,
                **kwargs) -> None:
        self.mode = mode
        self.extra_return = extra_return 
        super().__init__(layer_cfg=layer_cfg, **kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""  
        self.layers = ModuleList([            
            (checkpoint_wrapper(ParallelAttentionBlock(extra_return=self.extra_return, **self.layer_cfg))
            if self.mode == 'loss' else \
                ParallelAttentionBlock(extra_return=self.extra_return, **self.layer_cfg))
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    def forward(self,
                # rgb
                query: Tensor,
                query_pos: Tensor,
                # ir
                ir_query: Tensor,
                ir_query_pos: Tensor,
                # all
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor):
        
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        ir_reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        
        for layer in self.layers:
            query, ir_query = layer(
                # rgb
                query=query,
                query_pos=query_pos,
                reference_points=reference_points,
                # ir
                ir_query=ir_query,
                ir_query_pos=ir_query_pos,
                ir_reference_points=ir_reference_points,
                # all
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios)
            
        return_dict = dict()        
        return_dict['rgb'] = query
        if 'ir' in self.extra_return:
            return_dict['ir'] = ir_query
        return return_dict