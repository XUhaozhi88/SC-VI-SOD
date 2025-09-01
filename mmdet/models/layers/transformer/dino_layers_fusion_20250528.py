# Author: Xu Haozhi.
# Time:   2025.05.12

import torch
import torch.nn as nn
from mmengine.model import ModuleList
from torch import Tensor
from typing import Tuple

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from .deformable_detr_layers import (DeformableDetrTransformerEncoder,
    DeformableDetrTransformerEncoderLayer, DeformableDetrTransformerDecoderLayer)
from .dino_layers import DinoTransformerDecoder
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid
from .new_ops import MultiScaleDeformableAttention_MultiModal
from .new_ops_20250529 import (MultiScaleDeformableAttention_MultiModal1, 
    MultiScaleDeformableAttention_MultiModal2, MultiScaleDeformableAttention_MultiModal3, 
    MultiScaleDeformableAttention_MultiModal4)
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

class DeformableDetrTransformerDecoderLayer_MultiModal(DeformableDetrTransformerDecoderLayer):
    """Decoder layer of Deformable DETR."""

    def __init__(self,
                 num_modals=1,
                 **kwargs) -> None:
        self.num_modals = num_modals
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        # self.cross_attn = MultiScaleDeformableAttention_MultiModal(
        self.cross_attn = MultiScaleDeformableAttention_MultiModal1(
        # self.cross_attn = MultiScaleDeformableAttention_MultiModal2(
        # self.cross_attn = MultiScaleDeformableAttention_MultiModal3(
        # self.cross_attn = MultiScaleDeformableAttention_MultiModal4(
            **self.cross_attn_cfg, num_modals=self.num_modals)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

class DinoTransformerDecoder_MultiModal(DinoTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""
    def __init__(self, mode = 'tensor', 
                extra_return=None, **kwargs) -> None:
        self.mode = mode
        self.extra_return = extra_return 
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        num_modals = 1 + len(self.extra_return)
        self.layers = ModuleList([
            (checkpoint_wrapper(DeformableDetrTransformerDecoderLayer_MultiModal(
                **self.layer_cfg, num_modals=num_modals)) 
            if self.mode == 'loss' else DeformableDetrTransformerDecoderLayer_MultiModal(
                **self.layer_cfg, num_modals=num_modals))
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
                ir_value=ir_value,
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