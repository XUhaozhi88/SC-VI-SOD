# Author: Xu Haozhi.
# Time:   2025.05.12

import torch
import torch.nn as nn
from mmengine.model import ModuleList
from torch import Tensor
from typing import Tuple

from .deformable_detr_layers import (DeformableDetrTransformerEncoder,
    DeformableDetrTransformerEncoderLayer, DeformableDetrTransformerDecoderLayer)
from .dino_layers import DinoTransformerDecoder
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

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
            (checkpoint_wrapper(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
            if self.mode == 'loss' else DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
            for _ in range(self.num_layers)
        ])
        if 'ir' in self.extra_return:
            self.ir_layers = ModuleList([            
                (checkpoint_wrapper(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
                if self.mode == 'loss' else DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
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
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points)
        if 'ir' in self.extra_return:
            for layer in self.ir_layers:
                ir_query = layer(
                    query=ir_query,
                    query_pos=ir_query_pos,
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=ir_reference_points)
            
        return_dict = dict()        
        return_dict['rgb'] = query
        if 'ir' in self.extra_return:
            return_dict['ir'] = ir_query
        return return_dict

class DinoTransformerDecoder_Parallel(DinoTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""
    def __init__(self, mode = 'tensor', 
                extra_return=None, **kwargs) -> None:
        self.mode = mode
        self.extra_return = extra_return 
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            (checkpoint_wrapper(DeformableDetrTransformerDecoderLayer(**self.layer_cfg)) 
            if self.mode == 'loss' else DeformableDetrTransformerDecoderLayer(**self.layer_cfg))
            for _ in range(self.num_layers)
        ])
        if 'ir' in self.extra_return:
            self.ir_layers = ModuleList([            
                (checkpoint_wrapper(DeformableDetrTransformerDecoderLayer(**self.layer_cfg))
                if self.mode == 'loss' else DeformableDetrTransformerDecoderLayer(**self.layer_cfg))
                for _ in range(self.num_layers)
            ])

        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')        
        
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, ir_value: Tensor, key_padding_mask: Tensor,
            self_attn_mask: Tensor, reference_points: Tensor,
            spatial_shapes: Tensor, level_start_index: Tensor,
            valid_ratios: Tensor, reg_branches: nn.ModuleList,
            **kwargs) -> Tuple[Tensor]:
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)
            if self.extra_return:
                query = self.ir_layers[lid](
                    query,
                    query_pos=query_pos,
                    value=ir_value,
                    key_padding_mask=key_padding_mask,
                    self_attn_mask=self_attn_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    reference_points=reference_points_input,
                    **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points