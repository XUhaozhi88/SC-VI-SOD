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

class DWConv(BaseModule):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PVT2FFN(BaseModule):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MergeFFN(BaseModule):
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

    def forward(self, x, H, W):

        x, semantics = torch.split(x, [H * W, x.shape[1] - H * W], dim=1)
        semantics = self.fc_proxy(semantics)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        x = torch.cat([x, semantics], dim=1)
        return x


class Attention(BaseModule):
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

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C //
                                self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DualAttention(BaseModule):
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

    def forward(self, x, H, W, semantics):
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
        # kp, vp = kv_semantics[0], kv_semantics[1]

        attn = (q_semantics @ kv_semantics[0].transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # _semantics = (attn @ kv_semantics[1]).transpose(1, 2).reshape(B, N_p, C) * self.gamma2
        semantics = semantics + self.drop_path(
            (attn @ kv_semantics[1]).transpose(1, 2).reshape(B, N_p, C) * self.gamma2)
        semantics = semantics + self.drop_path(
            self.gamma3 * self.mlp_proxy(self.p_ln(semantics)))

        kv = self.kv(self.proxy_ln(semantics)).reshape(
            B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v = kv[0], kv[1]

        attn = (q @ kv[0].transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ kv[1]).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, semantics


class MergeBlock(BaseModule):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path=0.,
                 norm_layer=nn.LayerNorm, is_last=False, with_cp=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)

        if is_last:
            self.mlp = PVT2FFN(
                in_features=dim,
                hidden_features=int(
                    dim * mlp_ratio))
        else:
            self.mlp = MergeFFN(
                in_features=dim,
                hidden_features=int(
                    dim * mlp_ratio))
        self.is_last = is_last
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.with_cp = with_cp

    def forward(self, x, H, W):

        def innerforword(x):
            x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), H, W))

            if self.is_last:
                x, _ = torch.split(x, [H * W, x.shape[1] - H * W], dim=1)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), H, W))
            else:
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), H, W))
            return x
        if self.with_cp:
            x = checkpoint.checkpoint(innerforword, x)
        else:
            x = innerforword(x)

        return x


class DualBlock(BaseModule):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path=0., 
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 norm_layer=nn.LayerNorm, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = DualAttention(dim, num_heads, drop_path=drop_path)
        self.mlp = PVT2FFN(
            in_features=dim,
            hidden_features=int(
                dim * mlp_ratio))
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, H, W, semantics):
        def inner_forward(x, semantics, H, W):
            _x, semantics = self.attn(self.norm1(x), H, W, semantics)
            x = x + self.drop_path(self.gamma1 * _x)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), H, W))
            return x, semantics
        if self.with_cp:
            x, semantics = checkpoint.checkpoint(
                inner_forward, x, semantics, H, W)
        else:
            x, semantics = inner_forward(x, semantics, H, W)
        return x, semantics


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
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Stem(BaseModule):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
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
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)    # B, C, H, W ->B, C, H*W -> B, H*W, C
        x = self.norm(x)
        return x, H, W


class SemanticEmbed(BaseModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj_proxy = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels))

    def forward(self, semantics):
        semantics = self.proj_proxy(semantics)
        return semantics


@MODELS.register_module()
class DualVit(BaseModule):

    """Dual Vision Transformer https://arxiv.org/pdf/2207.04976.pdf Yao, T.,
    Li, Y., Pan, Y., Wang, Y., Zhang, X.P.

    and Mei, T., 2023. Dual vision transformer. IEEE transactions on pattern
    analysis and machine intelligence.
    """

    def __init__(self,
                 stem_width=32,
                 in_chans=3,
                 embed_dims=[64, 128, 320, 448],
                 num_heads=[2, 4, 10, 14],
                 mlp_ratios=[8, 8, 4, 3],
                 drop_path_rate=0.15,
                 norm_layer='LN',
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 out_indices=(1, 2, 3),
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(DualVit, self).__init__(init_cfg=init_cfg)
        self.out_indices = out_indices

        if norm_layer == 'LN':
            norm_layer = nn.LayerNorm
        self.with_cp = with_cp
        self.depths = depths
        self.num_stages = num_stages

        self.sep_stage = 2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_width, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            if i == 0:
                self.q = nn.Parameter(
                    torch.empty((64, embed_dims[0])),
                    requires_grad=True)
                self.q_embed = nn.Sequential(
                    nn.LayerNorm(embed_dims[0]),
                    nn.Linear(embed_dims[0], embed_dims[0])
                )
                self.pool = nn.AvgPool2d((7, 7), stride=7)
                self.kv = nn.Linear(embed_dims[0], 2 * embed_dims[0])
                self.scale = embed_dims[0] ** -0.5
                self.proxy_ln = nn.LayerNorm(embed_dims[0])
                self.se = nn.Sequential(
                    nn.Linear(embed_dims[0], embed_dims[0]),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dims[0], 2 * embed_dims[0]))
                trunc_normal_(self.q, std=.02)
            else:
                semantic_embed = SemanticEmbed(
                    embed_dims[i - 1], embed_dims[i]
                )
                setattr(self, f'proxy_embed{i + 1}', semantic_embed)

            if i >= self.sep_stage:
                block = nn.ModuleList([
                    MergeBlock(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i] - 1 if (j % 2 != 0 and i == 2) else mlp_ratios[i],
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        is_last=((i == 3) and (j == depths[i] - 1)),
                        with_cp=with_cp)
                    for j in range(depths[i])])
            else:
                block = nn.ModuleList([
                    DualBlock(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        with_cp=with_cp)
                    for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            norm_proxy = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

            if i != num_stages - 1:
                setattr(self, f'norm_proxy{i + 1}', norm_proxy)

        # self.apply(self.init_weights)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DualVit, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            # if self.use_abs_pos_embed:
            for m in self.modules():            
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        m.bias.data.zero_()
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = dualvit_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict
            self.load_state_dict(state_dict, False)            

    def forward_sep(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.sep_stage):
            patch_embed = getattr(self, f'patch_embed{i + 1}')  
            block = getattr(self, f'block{i + 1}')

            x, H, W = patch_embed(x)    # B, C, H, W -> B, H*W, C
            C = x.shape[-1]

            if i == 0:
                x_down = self.pool(x.reshape(B, H, W, C).permute(0, 3, 1, 2))   #  B, H*W, C -> B, H, W, C -> B, C, H, W -> pooling(1/7)
                x_down_H, x_down_W = x_down.shape[2:]
                x_down = x_down.view(B, C, -1).permute(0, 2, 1)     # B, C, H, W -> B, C, H*W -> B, H*W, C
                kv = self.kv(x_down).view(B, -1, 2, C).permute(2, 0, 1, 3)  # # B, H*W, C -> B, H*W, 2C -> B, H*W, 2, C -> 2, B, H*W, C
                # k, v = kv[0], kv[1]  # B, N, C

                attn_self_q = self.q.reshape(8, 8, -1).permute(2, 0, 1) # 8, 8, C -> C, 8, 8
                attn_self_q = F.interpolate(    # C, 8, 8 -> 1, C, 8, 8 -> 1, C, H, W -> C, H, W -> H, W, C
                    attn_self_q.unsqueeze(0),
                    size=(x_down_H, x_down_W),
                    mode='bicubic').squeeze(0).permute(
                    1, 2, 0)
                attn_self_q = attn_self_q.reshape(-1, attn_self_q.shape[-1])    # H*W, C

                # q: 1, M, C,   k: B, N, C -> B, M, N
                attn_self_q = (self.q_embed(attn_self_q) @  # H*W, C @ B, C, H*W -> B, H*W, H*W
                               kv[0].transpose(-1, -2)) * self.scale
                attn_self_q = attn_self_q.softmax(-1)  # B, M, N
                semantics = attn_self_q @ kv[1]  # B, M, C  # B, H*W, H*W @ B, H*W, C -> B, H*W, C
                semantics = semantics.view(B, -1, C)    # B, H*W, C

                semantics = torch.cat(
                    [semantics.unsqueeze(2), x_down.unsqueeze(2)], dim=2)   # B, H*W, 1, C cat B, H*W, 1, C -> B, H*W, 2, C
                se = self.se(semantics.sum(2).mean(1)).view(B, 2, C).softmax(1) # B, H*W, 2, C -> B, H*W, C -> B, C -> B, 2C -> B, 2, C
                # se = se.view(B, 2, C).softmax(1)
                semantics = (semantics * se.unsqueeze(1)).sum(2) # B, H*W, 2, C * B, 1, 2, C -> B, H*W, 2, C -> B, H*W, C
                semantics = self.proxy_ln(semantics)
            else:
                def inner_get_semantics(semantics):
                    semantics_embed = getattr(self, f'proxy_embed{i + 1}')
                    semantics = semantics_embed(semantics)
                    return semantics
                if self.with_cp:
                    semantics = checkpoint.checkpoint(inner_get_semantics, semantics)
                else:
                    semantics = inner_get_semantics(semantics)  # C[i-1] -> C[i]

            for blk in block:
                x, semantics = blk(x, H, W, semantics)

            norm = getattr(self, f'norm{i + 1}')
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            outs.append(x)

            norm_semantics = getattr(self, f'norm_proxy{i + 1}')
            semantics = norm_semantics(semantics)
        return x, semantics, tuple(outs)

    def forward_merge(self, x, semantics):
        B = x.shape[0]
        outs = []
        for i in range(self.sep_stage, self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            x, H, W = patch_embed(x)

            semantics_embed = getattr(self, f'proxy_embed{i + 1}')
            semantics = semantics_embed(semantics)

            x = torch.cat([x, semantics], dim=1)
            for blk in block:
                x = blk(x, H, W)

            if i != self.num_stages - 1:
                semantics = x[:, H * W:]
                x = x[:, 0:H * W]
                norm_semantics = getattr(self, f'norm_proxy{i + 1}')
                semantics = norm_semantics(semantics)

            norm = getattr(self, f'norm{i + 1}')
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        outs = tuple(outs)
        return outs

    def forward(self, x):

        x, semantics, out1 = self.forward_sep(x)

        def inner_forward(x, semantics):
            out2 = self.forward_merge(x, semantics)
            return out2

        if self.with_cp:
            out2 = checkpoint.checkpoint(inner_forward, x, semantics)
        else:
            out2 = inner_forward(x, semantics)
        outs = out1 + out2
        outs = [outs[i] for i in self.out_indices]
        return outs


class dualvit_s(DualVit):
    def __init__(self, **kwargs):
        super(dualvit_s, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 320, 448],
            num_heads=[2, 4, 10, 14],
            mlp_ratios=[8, 8, 4, 3],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            drop_path_rate=0.15,
            pretrained=kwargs['pretrained']
        )


class dualvit_b(DualVit):
    def __init__(self, **kwargs):
        super(dualvit_b, self).__init__(
            stem_width=64,
            embed_dims=[64, 128, 320, 512],
            num_heads=[2, 4, 10, 16],
            mlp_ratios=[8, 8, 4, 3],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 15, 3],
            drop_path_rate=0.15,
            pretrained=kwargs['pretrained']
        )

def dualvit_converter(ckpt):

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        if k.startswith('aux_head'):
            continue
        elif k.startswith('post_network'):
            continue
        elif k.startswith('block4.2.mlp.fc_proxy'):
            continue
        else:
            new_v = v
            new_k = k

        new_ckpt['backbone.' + new_k] = new_v

    return new_ckpt
