# Author: Xu Haozhi.
# Time:   2025.05.08

import math
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.utils import to_2tuple

from mmdet.registry import MODELS
from ..layers import PatchEmbed, PatchMerging


from .unixvit_utils import StemPatchEmbed, DownSamples, DualBlockMaskedEmbed, MergeBlockMaskedEmbed

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


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


class Semantics(BaseModule):

    def __init__(self, 
                 embed_dims=64, 
                 init_cfg=None):
        super().__init__(init_cfg)
        self.pool = nn.AvgPool2d((7, 7), stride=7)
        self.pool_ir = nn.AvgPool2d((7, 7), stride=7)
        self.kv_proj = nn.Linear(embed_dims, 2 * embed_dims)

        self.q = nn.Parameter(torch.empty((64, embed_dims)), requires_grad=True)
        trunc_normal_(self.q, std=.02)
        self.q_embed = nn.Sequential(
            nn.LayerNorm(embed_dims), nn.Linear(embed_dims, embed_dims))
        
        self.se = nn.Sequential(
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dims, 2 * embed_dims))
        self.proxy_ln = nn.LayerNorm(embed_dims)

        self.scale = embed_dims**-0.5

    def forward(self, x, x_ir, x_weight, x_ir_weight, bhwc_shape):
        B, H, W, C = bhwc_shape
        # pooling 7*7 downsample
        x_down = self.pool(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
        x_down_H, x_down_W = x_down.shape[2:]
        x_down = x_down.view(B, C, -1).permute(0, 2, 1) # B, HW, C
        x_ir_down = self.pool_ir(x_ir.reshape(B, H, W, C).permute(0, 3, 1, 2))   
        x_ir_down = x_ir_down.view(B, C, -1).permute(0, 2, 1)   # B, HW, C

        # concat -> generate key and value (via linear)
        x_down = torch.cat([x_down, x_ir_down], dim=1)  # B, 2HW, C
        kv = self.kv_proj(x_down).view(B, -1, 2, C).permute(2, 0, 1, 3)  # 2, B, 2HW, C
        # random initial query
        attn_self_q = self.q.reshape(8, 8, -1).permute(2, 0, 1) # C, 8, 8
        attn_self_q = F.interpolate(
            attn_self_q.unsqueeze(0),
            size=(x_down_H, x_down_W),
            mode='bicubic').squeeze(0).permute(1, 2, 0) # H, W, C
        # query projection (via linear)
        attn_self_q = attn_self_q.reshape(-1, attn_self_q.shape[-1])    # HW, C
        # attn_self_q = attn_self_q.reshape(-1, C)    # HW, C
        attn_self_q = self.q_embed(attn_self_q) # HW, C
        bsz, src_len, _ = kv[0].shape
        # generate attention mask
        N = x_down.shape[1]
        M = attn_self_q.shape[0]
        key_padding_mask = torch.zeros((B, N), dtype=torch.bool).to(x.device)
        attn_mask = torch.zeros((M, N), dtype=torch.bool).to(x.device)

        # 对没有输入的模态进行mask处理，有的为0，没有的为1
        total_patches = N // 2
        if x_weight == 1 and x_ir_weight == 0:
            key_padding_mask[:, total_patches:] = True
            attn_mask[:, total_patches:] = True

        elif x_weight == 0 and x_ir_weight == 1:
            key_padding_mask[:, :total_patches] = True
            attn_mask[:, :total_patches] = True
        # attention score mat (query with key)
        attn_self_q = (attn_self_q @ kv[0].transpose(-1, -2)) * self.scale  # B, HW, HW

        # mask attention weight防止有的模态没有，赋值为-inf，后续经过softmax后就为0了
        attn_self_q = get_masked_attn_output_weights(
            attn_self_q, bsz=bsz, num_heads=1, tgt_len=M, src_len=src_len,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # softmax normalize for attention score, and calculate results (score with value)
        attn_self_q = attn_self_q.softmax(-1)  # B, M, N    # B, HW, HW
        semantics = attn_self_q @ kv[1]  # B, M, C  # B, HW, C
        semantics = semantics.view(B, -1, C)    # B, HW, C
        # weighted sum (learnable weight)
        x_down = (x_down[:, : total_patches, :] * x_weight +
                  x_down[:, total_patches:, :] * x_ir_weight) / (x_weight + x_ir_weight)    # B, HW, C
        # semantics是由随机初始化的query(after atten with x)和下采样的特征一起构成的
        semantics = torch.cat([semantics.unsqueeze(2), x_down.unsqueeze(2)], dim=2)   # B, HW, 2, C
        # channel(query and x) attention
        se = self.se(semantics.sum(2).mean(1)).view(B, 2, C).softmax(1) # B, 2, C
        semantics = (semantics * se.unsqueeze(1)).sum(2)    # B, HW, C
        semantics = self.proxy_ln(semantics)
        return semantics


class SepBlockSequence(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 depth,
                 mlp_ratios,
                 drop_path_rate,
                 norm_layer,
                 score_embeds,
                 mod_embeds,
                 mod_nums,
                 score_embed_nums,
                 semantics_embed,
                 downsample=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = DualBlockMaskedEmbed(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            self.blocks.append(block)   

        self.score_embeds = score_embeds
        self.mod_embeds = mod_embeds

        self.semantics_embed = semantics_embed
        
        self.norm = nn.LayerNorm(embed_dims)
        self.out_norm = nn.LayerNorm(embed_dims)
        self.norm_semantics = nn.LayerNorm(embed_dims)

        self.downsample = downsample
        self.downsample_ir = deepcopy(downsample)

        self.scorenet = nn.Sequential(nn.Linear(embed_dims, embed_dims // 2),
                                       nn.LayerNorm(embed_dims // 2),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(embed_dims // 2, 2),
                                       nn.Sigmoid())
        self.mod_nums = mod_nums
        self.score_embed_nums = score_embed_nums
        self.extra_token_num = mod_nums + score_embed_nums

    def forward(self, x, x_ir, semantics, x_weight, x_ir_weight, 
                bhw_shape, score_end_idx, mod_emb_end_idx):
        score_embed = self.score_embeds # B, 1, C
        mod_embed = self.mod_embeds # B, 1, C

        B, H, W = bhw_shape
                
        semantics = self.semantics_embed(semantics)
        # 这里包含了MAF和MAA两个向量
        x = torch.cat([score_embed, mod_embed, x, x_ir], dim=1) # B, 1+1+2HW, C

        N = H * W * 2 + self.extra_token_num    # extra_token_num = mod_nums + score_embed_nums
        key_padding_mask = torch.zeros((B, N), dtype=torch.bool).to(x.device)
        attn_mask = None
        total_patches = H * W + self.extra_token_num
        # 对没有输入的模态进行mask处理，有的为0，没有的为1
        if x_weight == 1 and x_ir_weight == 0:
            key_padding_mask[:, total_patches:] = True  # B, HW

        elif x_weight == 0 and x_ir_weight == 1:
            key_padding_mask[:, self.extra_token_num:total_patches] = True

        #############################################################################
        for blk in self.blocks:
            x, semantics = blk(x, H, W, semantics, self.extra_token_num,
                                key_padding_mask, attn_mask)
        #############################################################################

        x = self.norm(x)
        extra_token = x[:, :self.extra_token_num, :]
        x = x[:, self.extra_token_num:, :]
        # 有可能需要变更score_embed和mod_embed
        score_embed = extra_token[:, :score_end_idx, :]
        mod_embed = extra_token[:, score_end_idx:mod_emb_end_idx, :]

        if self.score_embed_nums > 1:
            score_embed = (
                score_embed[:, 0, :] * x_weight + score_embed[:, 1, :] *
                x_ir_weight / (x_weight + x_ir_weight)).unsqueeze(0)
        if self.mod_nums > 1:
            mod_embed = (mod_embed[:, 0, :] * x_weight +
                            mod_embed[:, 1, :] * x_ir_weight /
                            (x_weight + x_ir_weight)).unsqueeze(0)
            
        # calculate weight for rgb/ir features
        score_weight = self.scorenet(score_embed).permute(2, 0, 1)    # B, 1, C -> 2, B, 1
        x, x_ir = x.chunk(2, dim=1)
        x = x * score_weight[0]
        x_ir = x_ir * score_weight[1]
        # mod embed有点类似于bias, 和score_embed一起构成了f(x) = ax + b的结构
        out_x = ((x * x_weight + x_ir * x_ir_weight) / (x_weight + x_ir_weight)) + mod_embed

        out_x = self.out_norm(out_x)
        out_x = out_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        semantics = self.norm_semantics(semantics)
        # 1D -> 2D -> downsample -> 1D
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_ir = x_ir.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, hw_shape = self.downsample(x)
        x_ir, _ = self.downsample_ir(x_ir)

        return x, x_ir, semantics, out_x, hw_shape

        
class MergeBlockSequence(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 depth,
                 mlp_ratios,
                 drop_path_rate,
                 norm_layer,
                 score_embeds,
                 mod_embeds,
                 mod_nums,
                 score_embed_nums,
                 is_norm_semantics,
                 semantics_embed,
                 downsample=None,
                 is_last=False,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = MergeBlockMaskedEmbed(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer,
                is_last=(is_last and (i == depth - 1)))
            self.blocks.append(block)  
        
        self.score_embeds = score_embeds
        self.mod_embeds = mod_embeds

        self.semantics_embed = semantics_embed
        
        self.norm = nn.LayerNorm(embed_dims)
        self.out_norm = nn.LayerNorm(embed_dims)
        self.norm_semantics = nn.LayerNorm(embed_dims) if is_norm_semantics else nn.Identity()
        
        self.downsample = downsample
        self.downsample_ir = deepcopy(downsample)

        self.scorenet = nn.Sequential(nn.Linear(embed_dims, embed_dims // 2),
                                       nn.LayerNorm(embed_dims // 2),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(embed_dims // 2, 2),
                                       nn.Sigmoid())
        
        self.mod_nums = mod_nums
        self.score_embed_nums = score_embed_nums
        self.extra_token_num = mod_nums + score_embed_nums

    def forward(self, x, x_ir, semantics, x_weight, x_ir_weight, 
                bhw_shape, score_end_idx, mod_emb_end_idx):
        B, H, W = bhw_shape

        semantics = self.semantics_embed(semantics)
        # diff, semantics is fused in x
        x = torch.cat([self.score_embeds, self.mod_embeds, x, x_ir, semantics], dim=1)

        M = semantics.shape[1]
        # 下面mask的生成方式不同是因为semantics这个时候也用来一起计算self atten了
        N = H * W * 2 + M + self.extra_token_num
        key_padding_mask = torch.zeros((B, N), dtype=torch.bool).to(x.device)
        attn_mask = torch.zeros((N, N), dtype=torch.bool).to(x.device)
        # x idx
        patches_start_1 = self.extra_token_num
        patches_end_1 = H * W + self.extra_token_num
        # x_ir idx
        patches_start_2 = H * W + self.extra_token_num
        patches_end_2 = H * W * 2 + self.extra_token_num
        # 对没有输入的模态进行mask处理，有的为0，没有的为1
        if x_weight == 1 and x_ir_weight == 0:
            key_padding_mask[:, patches_start_2:patches_end_2] = True
            attn_mask[patches_start_2:patches_end_2, patches_start_2:patches_end_2] = True

        elif x_weight == 0 and x_ir_weight == 1:
            key_padding_mask[:, patches_start_1:patches_end_1] = True
            attn_mask[patches_start_1:patches_end_1, patches_start_1:patches_end_1] = True

        #############################################################################
        for blk in self.blocks:
            x = blk(x, H, W, self.extra_token_num, key_padding_mask, attn_mask)
        #############################################################################

        semantics = x[:, patches_end_2:, :]
        semantics = self.norm_semantics(semantics)

        x = self.norm(x)
        extra_token = x[:, :self.extra_token_num, :]
        x = x[:, patches_start_1:patches_end_2, :]
        score_embed = extra_token[:, :score_end_idx, :]
        mod_embed = extra_token[:, score_end_idx: mod_emb_end_idx, :]
        if self.score_embed_nums > 1:
            score_embed = (
                score_embed[:, 0, :] * x_weight + score_embed[:, 1, :] *
                x_ir_weight / (x_weight + x_ir_weight)).unsqueeze(0)

        if self.mod_nums > 1:
            mod_embed = (mod_embed[:, 0, :] * x_weight +
                            mod_embed[:, 1, :] * x_ir_weight /
                            (x_weight + x_ir_weight)).unsqueeze(0)

        score_weight = self.scorenet(score_embed).permute(2, 0, 1)
        x_rgb, x_ir = x.chunk(2, dim=1)
        x_rgb = x_rgb * score_weight[0]
        x_ir = x_ir * score_weight[1]

        out_x = ((x_rgb * x_weight + x_ir * x_ir_weight) / (x_weight + x_ir_weight)) + mod_embed

        out_x = self.out_norm(out_x)
        out_x = out_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_ir = x_ir.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()        

        hw_shape = None
        if self.downsample and self.downsample_ir:
            x_rgb, hw_shape = self.downsample(x_rgb)
            x_ir, _ = self.downsample_ir(x_ir)

        return x_rgb, x_ir, semantics, out_x, hw_shape


@MODELS.register_module()
class UNIXVit_New(BaseModule):

    def __init__(self,
                 mode='unix',
                 stem_width=32,
                 in_chans=3,
                 embed_dims=[64, 128, 320, 448],
                 num_heads=[2, 4, 10, 14],
                 mlp_ratios=[8, 8, 4, 3],
                 drop_path_rate=0.15,
                 norm_layer='LN',
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 score_embed_nums=1,
                 num_scores=2,
                 mod_nums=1,
                 with_cp=False,
                 out_indices=(1, 2, 3),
                 patch_size=4,
                 patch_norm=True,
                 norm_cfg=dict(type='LN'),
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None):
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

        super(UNIXVit_New, self).__init__(init_cfg=init_cfg)

        ## unixvit
        self.out_indices = out_indices
        self.score_embed_nums = score_embed_nums
        self.num_scores = num_scores
        self.mod_nums = mod_nums
        self.extra_token_num = self.mod_nums + self.score_embed_nums

        # patch embedding
        if mode == 'unix':
            self.patch_embed = StemPatchEmbed(in_chans, stem_width, embed_dims[0])            
            self.patch_embed_ir = StemPatchEmbed(in_chans, stem_width, embed_dims[0])
        elif mode == 'vit':
            self.patch_embed = PatchEmbed(
                in_channels=in_chans,
                embed_dims=embed_dims[0],
                conv_type='Conv2d',
                kernel_size=patch_size,
                norm_cfg=norm_cfg if patch_norm else None,
                init_cfg=None)
            self.patch_embed_ir = PatchEmbed(
                in_channels=in_chans,
                embed_dims=embed_dims[0],
                conv_type='Conv2d',
                kernel_size=patch_size,
                norm_cfg=norm_cfg if patch_norm else None,
                init_cfg=None)
        
        self.get_semantics = Semantics(embed_dims=embed_dims[0])

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]
        norm_layer = nn.LayerNorm if norm_layer == 'LN' else nn.Identity
        self.stages = ModuleList()
        for i in range(num_stages):
            semantics_embed = nn.Sequential(
                nn.Linear(embed_dims[i - 1], embed_dims[i]),
                nn.LayerNorm(embed_dims[i])) if i != 0 else nn.Identity()
            downsample = DownSamples(embed_dims[i], embed_dims[i + 1]) \
                if i != num_stages - 1 else None
            if i < 2:
                stage = SepBlockSequence(
                    embed_dims=embed_dims[i],
                    num_heads=num_heads[i],
                    depth=depths[i],
                    mlp_ratios=mlp_ratios[i],
                    drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    norm_layer=norm_layer,
                    score_embeds=nn.Parameter(torch.rand([1, score_embed_nums, embed_dims[i]])),
                    mod_embeds=nn.Parameter(torch.rand([1, mod_nums, embed_dims[i]])),
                    mod_nums=mod_nums,
                    score_embed_nums=score_embed_nums,
                    semantics_embed=semantics_embed,
                    downsample=downsample,
                    init_cfg=None)
            else:
                stage = MergeBlockSequence(
                    embed_dims=embed_dims[i],
                    num_heads=num_heads[i],
                    depth=depths[i],
                    mlp_ratios=mlp_ratios[i],
                    drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    norm_layer=norm_layer,
                    score_embeds=nn.Parameter(torch.rand([1, score_embed_nums, embed_dims[i]])),
                    mod_embeds=nn.Parameter(torch.rand([1, mod_nums, embed_dims[i]])),
                    mod_nums=mod_nums,
                    score_embed_nums=score_embed_nums,
                    is_norm_semantics=True if i != num_stages - 1 else False,
                    semantics_embed=semantics_embed,
                    downsample=downsample,
                    is_last=False if i != num_stages - 1 else True,
                    init_cfg=None)                
            if with_cp:
                self.stages.append(checkpoint_wrapper(stage))
            else:
                self.stages.append(stage)   

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(UNIXVit_New, self).train(mode)
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

    def forward(self, x, x_ir):
        # 不同模态的权重
        x_weight = torch.tensor([1], requires_grad=False, device=x.device)
        x_ir_weight = torch.tensor([1], requires_grad=False, device=x.device)
        with torch.no_grad():
            # 判断哪些模态是存在的
            flage_x = torch.sum(x.detach())
            flage_x_ir = torch.sum(x_ir.detach())
            if flage_x == 0: x_weight = 0 * x_weight
            if flage_x_ir == 0: x_ir_weight = 0 * x_ir_weight
   
        bs = x.shape[0]
        outs = []
        score_start_idx = 0
        score_end_idx = score_start_idx + self.score_embed_nums
        mod_emb_end_idx = score_end_idx + self.mod_nums
        # 生成可学习的semantics query
        x, hw_shape = self.patch_embed(x)
        x_ir, _ = self.patch_embed_ir(x_ir)
        semantics = self.get_semantics(x, x_ir, x_weight, x_ir_weight, (bs, *hw_shape, x.shape[-1]))

        outs = []
        for stage in self.stages:
            x, x_ir, semantics, out_x, hw_shape = stage(x, x_ir, semantics, x_weight, x_ir_weight, 
                (bs, *hw_shape), score_end_idx, mod_emb_end_idx)
            outs.append(out_x)
        # x_fus = torch.cat([x, x_ir], dim=0) # 我用这个作为输出，不就可以用来处理后续的内容了，在encoder那里做做文章        
        return tuple([outs[i] for i in self.out_indices])
        