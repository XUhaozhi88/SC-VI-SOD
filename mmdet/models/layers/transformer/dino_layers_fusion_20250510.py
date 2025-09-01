# Author: Xu Haozhi.
# Time:   2025.05.10

from torch import Tensor
from mmengine.model import ModuleList

from .deformable_detr_layers import (DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer)

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
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                # ir
                ir_query: Tensor,
                ir_query_pos: Tensor,
                ir_key_padding_mask: Tensor,
                ir_spatial_shapes: Tensor,
                ir_level_start_index: Tensor,
                ir_valid_ratios: Tensor):
        
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        ir_reference_points = self.get_encoder_reference_points(
            ir_spatial_shapes, ir_valid_ratios, device=query.device)
        
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
                    key_padding_mask=ir_key_padding_mask,
                    spatial_shapes=ir_spatial_shapes,
                    level_start_index=ir_level_start_index,
                    valid_ratios=ir_valid_ratios,
                    reference_points=ir_reference_points)
            
        return_dict = dict()        
        return_dict['rgb'] = query
        if 'ir' in self.extra_return:
            return_dict['ir'] = ir_query
        return return_dict