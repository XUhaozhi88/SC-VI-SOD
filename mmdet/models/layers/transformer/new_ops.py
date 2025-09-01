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
class MultiScaleDeformableAttention_Local(MultiScaleDeformableAttention):

    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention_Local')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                valid_ratios = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                query_level_start_index: Optional[torch.Tensor] = None,
                value_level_start_index: Optional[torch.Tensor] = None,
                receptive_field_sizes: Optional[list] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
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
            query_level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            value_level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        ##############################################################################################################
        # 生成感受野掩码
        device = query.device
        level_ids = torch.zeros(num_query, device=device)
        for start_index in query_level_start_index[1:]:
            level_ids[start_index:] += 1
    
        receptive_field_sizes = torch.tensor(receptive_field_sizes, dtype=torch.float32, device=device) # strides 8, 16, 32, 64
        receptive_field_sizes = receptive_field_sizes[:, None] / spatial_shapes # num_level, 2

        location_range_min = torch.clamp(reference_points - receptive_field_sizes[None, None, ...], min=0.0)    # bs, num_query, num_level, 2
        # location_range_max = torch.clamp(reference_points + receptive_field_sizes[None, None, ...], max=1.0)    # bs, num_query, num_level, 2 这种方式比较粗糙
        location_range_max = []
        for i in range(bs):
            location_range_max_level = []
            for level in range(self.num_levels):
                location_range_max_x = torch.clamp(reference_points[i, :, level, 0] + receptive_field_sizes[level, 0], max=valid_ratios[i, level, 0])    # num_query,
                location_range_max_y = torch.clamp(reference_points[i, :, level, 1] + receptive_field_sizes[level, 1], max=valid_ratios[i, level, 1])    # num_query,
                location_range_max_level.append(torch.stack((location_range_max_x, location_range_max_y), dim=-1))  # num_level * [num_query, 2]
            location_range_max.append(torch.stack(location_range_max_level, dim=1))   # bs * [num_query, num_level, 2]
        location_range_max = torch.stack(location_range_max, dim=0)   # bs, num_query, num_level, 2
        location_range = torch.cat((location_range_min, location_range_max), dim=-1)    # bs, num_query, num_level, 4
        location_range = location_range[:, :, None, :, None, :] # bs, num_query, 1, num_level, 1, 4

        # 下面这一段逻辑好像有点问题，是根据num_query那个维度创建的感受野，但是实际上这个感受野是给value用的，对应于num_level那个维度
        # location_range = []
        # for i in range(num_query):
        #     level_id = int(level_ids[i])
        #     receptive_field_size = receptive_field_sizes[:, level_id]   # bs, 2

        #     query_location_x, query_location_y = reference_points[:, i, level_id, 0], reference_points[:, i, level_id, 1]
        #     query_location_range = []
        #     for level in range(self.num_levels):
        #         # xmin, ymin, xmax, ymax
        #         xmin = torch.clamp(query_location_x - receptive_field_size[:, 0], min=0.0)  # (bs,)
        #         ymin = torch.clamp(query_location_y - receptive_field_size[:, 1], min=0.0)
        #         xmax = torch.clamp(query_location_x + receptive_field_size[:, 0], max=spatial_shapes[level, 0])
        #         ymax = torch.clamp(query_location_y + receptive_field_size[:, 1], max=spatial_shapes[level, 1])
        #         query_location_range.append(torch.stack([xmin, ymin, xmax, ymax], dim=-1))  # num_level*[bs, 4]
        #     location_range.append(torch.stack(query_location_range, dim=1)) # num_query*[bs, num_level, 4]
        # location_range = torch.stack(location_range, dim=1) # bs, num_query, num_level, 4        
        # location_range = location_range[:, :, None, :, None, :] # bs, num_query, num_head, num_level, num_point, 4   
        ##############################################################################################################

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]    # bs, num_query, num_head, num_level, num_point, 2
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.')        

        ##############################################################################################################
        # 生成mask，并对注意力分数赋零
        xmin_mask = sampling_locations[..., 0] > location_range[..., 0]
        ymin_mask = sampling_locations[..., 1] > location_range[..., 1]
        xmax_mask = sampling_locations[..., 0] < location_range[..., 2]
        ymax_mask = sampling_locations[..., 1] < location_range[..., 3]

        mask = xmin_mask & ymin_mask & xmax_mask & ymax_mask    # bs, num_query, num_head, num_level, num_point
        attention_weights = attention_weights * mask.to(torch.float)
        ##############################################################################################################

        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, value_level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity



@MODELS.register_module()
class MultiScaleDeformableAttention_MultiModal(MultiScaleDeformableAttention):

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
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)   # bs, Nq, 1(Nh), Nl, 1(Nm), 1(Np), 4
            sampling_locations = reference_points[:, :, None, :, None, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, None, :]   # 1(bs), 1(Nq), 1(Nh), Nl, 1(Nm), 1(Np), 2
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, None, 2:] \
                * 0.5
            # 需要验证sampling_offsets的不同归一化方法，是否需要平均各模态
            # sampling_locations = reference_points[:, :, None, :, None, None, :2] \
            #     + sampling_offsets / (self.num_points * self.num_modals) \
            #     * reference_points[:, :, None, :, None, None, 2:] \
            #     * 0.5contiguous
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        sampling_locations, ir_sampling_locations = sampling_locations.split([1, 1], dim=-3)
        attention_weights, ir_attention_weights = attention_weights.split([1, 1], dim=-2)
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

