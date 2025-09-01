import math
import warnings
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import MMLogger
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner.checkpoint import CheckpointLoader

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS

import math
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.drop import Dropout
from mmengine.model import BaseModule, ModuleList
from mmengine.utils import to_2tuple
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig

from ..layers.transformer.utils import AdaptivePadding


class DownSamples(BaseModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.proj(x)
        hw_shape = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, hw_shape

class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmengine.ConfigDict`, optional): The Config for
            initialization. Default: None.
    """

    def __init__(self,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 conv_type: str = 'Conv2d',
                 kernel_size: int = 16,
                 stride: int = 16,
                 padding: Union[int, tuple, str] = 'corner',
                 dilation: int = 1,
                 bias: bool = True,
                 norm_cfg: OptConfigType = None,
                 input_size: Union[int, tuple] = None,
                 init_cfg: OptConfigType = None) -> None:
        super(PatchEmbed, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int]]:
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size
    

class StemPatchEmbed(BaseModule):
    def __init__(self, in_channels, stem_hidden_dim, out_channels, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            build_norm_layer(
                dict(
                    type='BN',
                    requires_grad=False),
                hidden_dim)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            build_norm_layer(
                dict(
                    type='BN',
                    requires_grad=False),
                hidden_dim)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            build_norm_layer(
                dict(
                    type='BN',
                    requires_grad=False),
                hidden_dim)[1],
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)    # B, C, H, W -> B, C, H/2, W/2
        x = self.proj(x)    # B
        hw_shape = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)    # B, C, H, W ->B, C, H*W -> B, H*W, C
        x = self.norm(x)
        return x, hw_shape
    
class DWConv(BaseModule):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PVT2FFNEmbed(BaseModule):

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x, H, W, num_extra_token):
        x = self.fc1(x)

        extra_token = x[:, :num_extra_token, :]
        x = x[:, num_extra_token:num_extra_token + H * W * 2, :]

        x = x.chunk(2, dim=1)
        x = torch.cat([x[0], x[1]], dim=0)
        x = self.dwconv(x, H, W)
        x = x.chunk(2, dim=0)
        x = torch.cat([x[0], x[1]], dim=1)

        x = torch.cat([extra_token, x], dim=1)
        # 处理B 2N C --> 2B N C--> B 2N C
        x = self.act(x)
        x = self.fc2(x)
        return x

def get_masked_attn_output_weights(attn_weights,
                                   bsz,
                                   tgt_len,
                                   src_len,
                                   attn_mask=None,
                                   key_padding_mask=None,
                                   num_heads=1):

    attn_weights_org_size = attn_weights.size()
    if list(attn_weights_org_size) != [bsz * num_heads, tgt_len, src_len]:
        attn_weights = attn_weights.view(bsz * num_heads, tgt_len, src_len)

    assert list(attn_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, tgt_len, src_len]:
                print('attn_mask size:', attn_mask.size(), flush=True)
                print('[1, tgt_len, src_len]:', [1, tgt_len, src_len],
                      flush=True)
                raise RuntimeError(
                    'The size of the 2D attn_mask is not correct.')
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.dtype == torch.bool:
            attn_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_weights += attn_mask

    if key_padding_mask is not None:

        attn_weights = attn_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_weights = attn_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_weights = attn_weights.view(bsz * num_heads, tgt_len, src_len)
    if list(attn_weights_org_size) != [bsz * num_heads, tgt_len, src_len]:
        attn_weights = attn_weights.view(attn_weights_org_size)

    return attn_weights

class DualMaskedAttention(BaseModule):
    def __init__(self, dim, num_heads, drop_path=0.0):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_proxy = nn.Linear(dim, dim)
        self.kv_proxy = nn.Linear(dim, dim * 2)
        self.q_proxy_ln = nn.LayerNorm(dim)

        self.p_ln = nn.LayerNorm(dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp_proxy = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * dim, dim))
        self.proxy_ln = nn.LayerNorm(dim)

        self.qkv_proxy = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 3))

        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma3 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None

    def selfatt(self, semantics):
        B, N, C = semantics.shape
        qkv = self.qkv_proxy(semantics).reshape(
            B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = (qkv[0] @ qkv[1].transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        semantics = (attn @ qkv[2]).transpose(1, 2).reshape(B, N, C)
        return semantics

    def forward(self, x, H, W, semantics,
                key_padding_mask=None,
                attn_mask=None):
        semantics = semantics + self.drop_path(
            self.gamma1 * self.selfatt(semantics))

        B, N, C = x.shape
        B_p, N_p, C_p = semantics.shape
        q = self.q(x).reshape(
            B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q_semantics = self.q_proxy(
            self.q_proxy_ln(semantics)).reshape(
            B_p, N_p, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_semantics = self.kv_proxy(x).reshape(
            B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        bsz, num_heads, tgt_len, _ = q_semantics.shape
        _, _, src_len, _ = kv_semantics[0].shape

        attn = (q_semantics @ kv_semantics[0].transpose(-2, -1)) * self.scale
        # diff
        attn = get_masked_attn_output_weights(
            attn_weights=attn,
            bsz=bsz,
            num_heads=num_heads,
            tgt_len=tgt_len,
            src_len=src_len,
            key_padding_mask=key_padding_mask)

        attn = attn.softmax(dim=-1)

        semantics = semantics + self.drop_path(
            (attn @ kv_semantics[1]).transpose(1, 2).reshape(B, N_p, C) * self.gamma2)
        semantics = semantics + self.drop_path(
            self.gamma3 * self.mlp_proxy(self.p_ln(semantics)))

        kv = self.kv(self.proxy_ln(semantics)).reshape(
            B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        bsz, num_heads, tgt_len, _ = q.shape
        _, _, src_len, _ = kv[0].shape

        attn = (q @ kv[0].transpose(-2, -1)) * self.scale
        # diff，感觉下面这一块没用，确实没用
        attn = get_masked_attn_output_weights(
            attn_weights=attn,
            bsz=bsz,
            num_heads=num_heads,
            tgt_len=tgt_len,
            src_len=src_len,
            attn_mask=None,
            key_padding_mask=None)
        attn = attn.softmax(dim=-1)
        x = (attn @ kv[1]).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, semantics
    

class DualBlockMaskedEmbed(BaseModule):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path=0., 
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 norm_layer=nn.LayerNorm, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = DualMaskedAttention(dim, num_heads, drop_path=drop_path)
        self.mlp = PVT2FFNEmbed(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, H, W, semantics, 
                num_extra_token,
                key_padding_mask=None,
                attn_mask=None):
        _x, semantics = self.attn(self.norm1(x), H, W, semantics,
                                    key_padding_mask=key_padding_mask,
                                    attn_mask=attn_mask)
        x = x + self.drop_path(self.gamma1 * _x)
        # 这里有点意思的是PVT2FFN这里把各模态分开处理了，之后再cat回来，可能是DWconv导致的
        x = x + self.drop_path(
            self.gamma2 * self.mlp(self.norm2(x), H, W, num_extra_token))
        return x, semantics
    

class MaskedAttention(BaseModule):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        bsz, num_heads, tgt_len, _ = q.shape
        src_len = k.shape[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = get_masked_attn_output_weights(
            attn,
            bsz,
            tgt_len,
            src_len,
            num_heads=num_heads,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MergeFFNEmbed(BaseModule):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

        self.fc_proxy = nn.Sequential(
            nn.Linear(in_features, 2 * in_features),
            nn.GELU(),
            nn.Linear(2 * in_features, in_features))

    def forward(self, x, H, W, num_extra_token):
        semantics = x[:, num_extra_token + H * W * 2:, :]
        semantics = self.fc_proxy(semantics)

        x = self.fc1(x)
        extra_token = x[:, :num_extra_token, :]
        x = x[:, num_extra_token:num_extra_token + H * W * 2, :]

        x = x.chunk(2, dim=1)
        x = torch.cat([x[0], x[1]], dim=0)
        x = self.dwconv(x, H, W)
        x = x.chunk(2, dim=0)
        x = torch.cat([x[0], x[1]], dim=1)

        x = torch.cat([extra_token, x], dim=1)
        x = self.act(x)
        x = self.fc2(x)
        x = torch.cat([x, semantics], dim=1)
        return x

class MergeBlockMaskedEmbed(BaseModule):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 is_last=False):
        super(MergeBlockMaskedEmbed, self).__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MaskedAttention(dim, num_heads)
        if is_last:
            self.mlp = PVT2FFNEmbed(in_features=dim,
                                    hidden_features=int(dim * mlp_ratio))
        else:
            self.mlp = MergeFFNEmbed(in_features=dim,
                                     hidden_features=int(dim * mlp_ratio))
        self.is_last = is_last
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, H, W,
                num_extra_token,
                key_padding_mask=None,
                attn_mask=None):
        x = x + self.drop_path(
            self.gamma1 * self.attn(self.norm1(x),
                                    key_padding_mask=key_padding_mask,
                                    attn_mask=attn_mask))
        if self.is_last:
            x = x[:, :2 * H * W + num_extra_token]
            x = x + self.drop_path(self.gamma2 * self.mlp(
                self.norm2(x), H, W, num_extra_token))
        else:
            x = x + self.drop_path(self.gamma2 * self.mlp(
                self.norm2(x), H, W, num_extra_token))
        return x