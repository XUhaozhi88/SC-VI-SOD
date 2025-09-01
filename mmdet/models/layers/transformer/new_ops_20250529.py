# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Optional, no_type_check

import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule, constant_init, xavier_init
from mmengine.registry import MODELS
from mmengine.utils import deprecated_api_warning
from torch.autograd.function import Function, once_differentiable

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, ext_loader
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch, MultiScaleDeformableAttention

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])



@MODELS.register_module()
class MultiScaleDeformableAttention_MultiModal1(MultiScaleDeformableAttention):

    def __init__(self,
                embed_dims: int = 256,
                num_heads: int = 8,
                num_levels: int = 4,
                num_modals: int = 2,
                num_points: int = 4,
                im2col_step: int = 64,
                dropout: float = 0.1,
                batch_first: bool = False,
                norm_cfg: Optional[dict] = None,
                init_cfg: Optional[mmengine.ConfigDict] = None,
                value_proj_ratio: float = 1.0):
        super(MultiScaleDeformableAttention, self).__init__(init_cfg=init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                                f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_modals = num_modals
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_modals * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_modals * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.ir_value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()
    
    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1, 1, 2).repeat(1, self.num_modals, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
        
        xavier_init(self.ir_value_proj, distribution='uniform', bias=0.)

    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention_MultiModal1')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                ir_value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:

        if value is None:
            value = query
        if ir_value is None:
            ir_value = value

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            ir_value = ir_value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        ir_value = self.ir_value_proj(ir_value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
            ir_value = ir_value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        ir_value = ir_value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_modals, self.num_levels, self.num_points, 2)    # bs, Nq, Nh, Nm, Nl, Np, 2
        attention_weights = self.attention_weights(query).view( 
            bs, num_query, self.num_heads, self.num_modals * self.num_levels * self.num_points)   # bs, Nq, Nh, Nm*Nl*Np
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_modals,
                                                   self.num_levels, self.num_points) # bs, Nq, Nh, Nm, Nl, Np
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)   # bs, Nq, 1(Nh), 1(Nm), Nl, 1(Np), 4
            sampling_locations = reference_points[:, :, None, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, None, :, None, :]   # 1(bs), 1(Nq), 1(Nh), 1(Nm), Nl, 1(Np), 2
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, None, :, None, 2:] \
                * 0.5
            # 需要验证sampling_offsets的不同归一化方法，是否需要平均各模态
            # sampling_locations = reference_points[:, :, None, None, :, None, :2] \
            #     + sampling_offsets / (self.num_points * self.num_modals) \
            #     * reference_points[:, :, None, None, :, None, 2:] \
            #     * 0.5contiguous
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        sampling_locations, ir_sampling_locations = sampling_locations.split([1, 1], dim=3)
        attention_weights, ir_attention_weights = attention_weights.split([1, 1], dim=3)
        sampling_locations = sampling_locations.contiguous()
        ir_sampling_locations = ir_sampling_locations.contiguous()
        attention_weights = attention_weights.contiguous()
        ir_attention_weights = ir_attention_weights.contiguous()

        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, 
                sampling_locations,  # bs, Nq, Nh, Nl, Np, 4
                attention_weights,   # bs, Nq, Nh, Nl, Np
                self.im2col_step) + \
                    MultiScaleDeformableAttnFunction.apply(
                        ir_value, spatial_shapes, level_start_index, 
                        ir_sampling_locations,
                        ir_attention_weights, 
                        self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, 
                sampling_locations, 
                attention_weights) + \
                    multi_scale_deformable_attn_pytorch(
                        ir_value, spatial_shapes, 
                        ir_sampling_locations, 
                        ir_attention_weights)
            
        output = self.output_proj(output)
        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@MODELS.register_module()
class MultiScaleDeformableAttention_MultiModal2(MultiScaleDeformableAttention):

    def __init__(self,
                embed_dims: int = 256,
                num_heads: int = 8,
                num_levels: int = 4,
                num_modals: int = 2,
                num_points: int = 4,
                im2col_step: int = 64,
                dropout: float = 0.1,
                batch_first: bool = False,
                norm_cfg: Optional[dict] = None,
                init_cfg: Optional[mmengine.ConfigDict] = None,
                value_proj_ratio: float = 1.0):
        super(MultiScaleDeformableAttention, self).__init__(init_cfg=init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                                f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_modals = num_modals
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_modals * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_modals * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.ir_value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.ir_output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()
    
    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1, 1, 2).repeat(1, self.num_modals, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
        
        xavier_init(self.ir_value_proj, distribution='uniform', bias=0.)
        xavier_init(self.ir_output_proj, distribution='uniform', bias=0.)

    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention_MultiModal2')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                ir_value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            ir_value (torch.Tensor): The ir_value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if ir_value is None:
            ir_value = value

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            ir_value = ir_value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        ir_value = self.ir_value_proj(ir_value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
            ir_value = ir_value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        ir_value = ir_value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_modals, self.num_levels, self.num_points, 2)    # bs, Nq, Nh, Nm, Nl, Np, 2
        # diff
        attention_weights = self.attention_weights(query).view( 
            bs, num_query, self.num_heads, self.num_modals, self.num_levels * self.num_points)   # bs, Nq, Nh, Nm, Nl*Np
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_modals,
                                                   self.num_levels, self.num_points) # bs, Nq, Nh, Nm, Nl, Np
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)   # bs, Nq, 1(Nh), 1(Nm), Nl, 1(Np), 4
            sampling_locations = reference_points[:, :, None, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, None, :, None, :]   # 1(bs), 1(Nq), 1(Nh), 1(Nm), Nl, 1(Np), 2
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, None, :, None, 2:] \
                * 0.5
            # 需要验证sampling_offsets的不同归一化方法，是否需要平均各模态
            # sampling_locations = reference_points[:, :, None, None, :, None, :2] \
            #     + sampling_offsets / (self.num_points * self.num_modals) \
            #     * reference_points[:, :, None, None, :, None, 2:] \
            #     * 0.5contiguous
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        sampling_locations, ir_sampling_locations = sampling_locations.split([1, 1], dim=3)
        attention_weights, ir_attention_weights = attention_weights.split([1, 1], dim=3)
        sampling_locations = sampling_locations.contiguous()
        ir_sampling_locations = ir_sampling_locations.contiguous()
        attention_weights = attention_weights.contiguous()
        ir_attention_weights = ir_attention_weights.contiguous()

        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, 
                sampling_locations,  # bs, Nq, Nh, Nl, Np, 4
                attention_weights,   # bs, Nq, Nh, Nl, Np
                self.im2col_step)
            ir_output = MultiScaleDeformableAttnFunction.apply(
                ir_value, spatial_shapes, level_start_index, 
                ir_sampling_locations,
                ir_attention_weights, 
                self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, 
                sampling_locations, 
                attention_weights)
            ir_output = multi_scale_deformable_attn_pytorch(
                ir_value, spatial_shapes, 
                ir_sampling_locations, 
                ir_attention_weights)
            
        output = self.output_proj(output)
        ir_output = self.ir_output_proj(ir_output)
        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)
            ir_output = ir_output.permute(1, 0, 2)

        return self.dropout(output + ir_output) + identity


@MODELS.register_module()
class MultiScaleDeformableAttention_MultiModal3(MultiScaleDeformableAttention):

    def __init__(self,
                embed_dims: int = 256,
                num_heads: int = 8,
                num_levels: int = 4,
                num_modals: int = 2,
                num_points: int = 4,
                im2col_step: int = 64,
                dropout: float = 0.1,
                batch_first: bool = False,
                norm_cfg: Optional[dict] = None,
                init_cfg: Optional[mmengine.ConfigDict] = None,
                value_proj_ratio: float = 1.0):
        super(MultiScaleDeformableAttention, self).__init__(init_cfg=init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                                f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_modals = num_modals
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.ir_sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.ir_attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.ir_value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.ir_output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()
    
    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        constant_init(self.ir_sampling_offsets, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        self.ir_sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        constant_init(self.ir_attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.ir_value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        xavier_init(self.ir_output_proj, distribution='uniform', bias=0.)
        self._is_init = True        

    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention_MultiModal3')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                ir_value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            ir_value (torch.Tensor): The ir_value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if ir_value is None:
            ir_value = value

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            ir_value = ir_value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        ir_value = self.ir_value_proj(ir_value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
            ir_value = ir_value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        ir_value = ir_value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)    # bs, Nq, Nh, Nl, Np, 2
        ir_sampling_offsets = self.ir_sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)    # bs, Nq, Nh, Nl, Np, 2
        # diff
        attention_weights = self.attention_weights(query).view( 
            bs, num_query, self.num_heads, self.num_levels * self.num_points)   # bs, Nq, Nh, Nl*Np
        attention_weights = attention_weights.softmax(-1)
        ir_attention_weights = self.ir_attention_weights(query).view( 
            bs, num_query, self.num_heads, self.num_levels * self.num_points)   # bs, Nq, Nh, Nl*Np
        ir_attention_weights = ir_attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query, self.num_heads,
                                                   self.num_levels, self.num_points) # bs, Nq, Nh, Nl, Np
        ir_attention_weights = ir_attention_weights.view(bs, num_query, self.num_heads,
                                                   self.num_levels, self.num_points) # bs, Nq, Nh, Nl, Np
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)   # bs, Nq, 1(Nh), Nl, 1(Np), 4
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]   # 1(bs), 1(Nq), 1(Nh), Nl, 1(Np), 2
            ir_sampling_locations = reference_points[:, :, None, :, None, :] \
                + ir_sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]   # 1(bs), 1(Nq), 1(Nh), Nl, 1(Np), 2
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
            ir_sampling_locations = reference_points[:, :, None, :, None, :2] \
                + ir_sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, 
                sampling_locations,  # bs, Nq, Nh, Nl, Np, 4
                attention_weights,   # bs, Nq, Nh, Nl, Np
                self.im2col_step)
            ir_output = MultiScaleDeformableAttnFunction.apply(
                ir_value, spatial_shapes, level_start_index, 
                ir_sampling_locations,
                ir_attention_weights, 
                self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, 
                sampling_locations, 
                attention_weights)
            ir_output = multi_scale_deformable_attn_pytorch(
                ir_value, spatial_shapes, 
                ir_sampling_locations, 
                ir_attention_weights)
            
        output = self.output_proj(output)
        ir_output = self.ir_output_proj(ir_output)
        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)
            ir_output = ir_output.permute(1, 0, 2)
        # 这里需不需要根据value生成一个权重，然后加权求和output和ir_output
        return self.dropout(output + ir_output) + identity
    

@MODELS.register_module()
class MultiScaleDeformableAttention_MultiModal4(MultiScaleDeformableAttention):

    def __init__(self,
                embed_dims: int = 256,
                num_heads: int = 8,
                num_levels: int = 4,
                num_modals: int = 2,
                num_points: int = 4,
                im2col_step: int = 64,
                dropout: float = 0.1,
                batch_first: bool = False,
                norm_cfg: Optional[dict] = None,
                init_cfg: Optional[mmengine.ConfigDict] = None,
                value_proj_ratio: float = 1.0):
        super(MultiScaleDeformableAttention, self).__init__(init_cfg=init_cfg)
        # super(DETRHead, self).__init__(init_cfg=init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                                f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_modals = num_modals
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_modals * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_modals * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.ir_value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()
    
    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1, 1, 2).repeat(1, self.num_levels, self.num_modals, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
        
        xavier_init(self.ir_value_proj, distribution='uniform', bias=0.)

    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention_MultiModal')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                ir_value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention. """

        if value is None:
            value = query
        if ir_value is None:
            ir_value = value

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            ir_value = ir_value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        ir_value = self.ir_value_proj(ir_value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
            ir_value = ir_value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        ir_value = ir_value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_modals, self.num_points, 2)    # bs, Nq, Nh, Nl, Nm, Np, 2
        attention_weights = self.attention_weights(query).view( 
            bs, num_query, self.num_heads, self.num_levels * self.num_modals * self.num_points)   # bs, Nq, Nh, Nl*Nm*Np
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,   # bs, Nq, Nh, Nl, Nm, Np
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_modals, 
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)   # bs, Nq, 1(Nh), Nl, 1(Nm), 1(Np), 2
            sampling_locations = reference_points[:, :, None, :, None, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, None, :]   # 1(bs), 1(Nq), 1(Nh), Nl, 1(Nm), 1(Np), 2
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        # 把modal这个维度叠加到level上
        sampling_locations = sampling_locations.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_modals, self.num_points, 2)  # bs, Nq, Nh, Nl, Nm, Np, 2

        output = self.multi_modal_multi_scale_deformable_attn(
                value, ir_value, spatial_shapes, 
                sampling_locations, 
                attention_weights)
            
        output = self.output_proj(output)
        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    @staticmethod
    def multi_modal_multi_scale_deformable_attn(
            value: torch.Tensor, ir_value: torch.Tensor, value_spatial_shapes: torch.Tensor,
            sampling_locations: torch.Tensor,
            attention_weights: torch.Tensor) -> torch.Tensor:
        """CPU version of multi-scale deformable attention.

        Args:
            value (torch.Tensor): The value has shape
                (bs, num_keys, num_heads, embed_dims//num_heads)
            value_spatial_shapes (torch.Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (torch.Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (torch.Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),

        Returns:
            torch.Tensor: has shape (bs, num_queries, embed_dims)
        """

        bs, _, num_heads, embed_dims = value.shape
        _, num_queries, num_heads, num_levels, num_modals, num_points, _ =\
            sampling_locations.shape
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        ir_value_list = ir_value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level, (H_, W_) in enumerate(value_spatial_shapes):
            # bs, H_*W_, num_heads, embed_dims ->
            # bs, H_*W_, num_heads*embed_dims ->
            # bs, num_heads*embed_dims, H_*W_ ->
            # bs*num_heads, embed_dims, H_, W_
            value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
                bs * num_heads, embed_dims, H_, W_)
            ir_value_l_ = ir_value_list[level].flatten(2).transpose(1, 2).reshape(
                bs * num_heads, embed_dims, H_, W_)
            # bs, num_queries, num_heads, num_modals, num_points, 2 ->
            # bs, num_heads, num_queries, num_modals, num_points, 2 ->
            # bs*num_heads, num_queries, num_modals, num_points, 2
            sampling_grid_l_ = sampling_grids[:, :, :,
                                            level].transpose(1, 2).flatten(0, 1)
            # bs*num_heads, embed_dims, num_queries, num_points
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_[:, :, 0],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            ir_sampling_value_l_ = F.grid_sample(
                ir_value_l_,
                sampling_grid_l_[:, :, 1],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            # [bs*num_heads, embed_dims, num_queries, num_points] * num_modals ->
            # bs*num_heads, embed_dims, num_queries, num_modals, num_points
            sampling_value_l_ = torch.stack((sampling_value_l_, ir_sampling_value_l_), dim=-2)
            sampling_value_list.append(sampling_value_l_)
        # (bs, num_queries, num_heads, num_levels, num_points) ->
        # (bs, num_heads, num_queries, num_levels, num_points) ->
        # (bs*num_heads, 1, num_queries, num_levels*num_points)
        attention_weights = attention_weights.transpose(1, 2).reshape(
            bs * num_heads, 1, num_queries, num_levels * num_modals * num_points)
        output = (torch.stack(sampling_value_list, dim=-3).flatten(-3) *
                attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                                num_queries)
        return output.transpose(1, 2).contiguous()