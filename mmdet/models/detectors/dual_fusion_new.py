import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmengine.model.weight_init import trunc_normal_

from mmengine.model import BaseModule

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


def get_masked_attn_output_weights(
        attn_weights,
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


class DWConv(BaseModule):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
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

    def selfatt(self, semantics, key_padding_mask=None, attn_mask=None):

        B, N, C = semantics.shape
        qkv = self.qkv_proxy(semantics).reshape(B, -1, 3, self.num_heads,
                                                C // self.num_heads).permute(
                                                    2, 0, 3, 1, 4)
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
    

class ScoreNet(BaseModule):

    def __init__(self, dim, num_scores):
        super(ScoreNet, self).__init__()
        self.dim = dim
        self.score_net = nn.Sequential(nn.Linear(self.dim, self.dim // 2),
                                       nn.LayerNorm(self.dim // 2),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.dim // 2, num_scores),
                                       nn.Sigmoid())

    def forward(self, x):
        x = self.score_net(x).permute(2, 0, 1)
        return x
        

class DualBlockMaskedEmbed(BaseModule):

    def __init__(self, dim, num_heads, mlp_ratio, drop_path=0., 
                 norm_layer=nn.LayerNorm, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
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


class SemanticEmbed(BaseModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj_proxy = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels))

    def forward(self, semantics):
        semantics = self.proj_proxy(semantics)
        return semantics
        

class Dual_Fusion_New(BaseModule):

    def __init__(self,
                 embed_dims=[256] * 4,
                 num_heads=[8] * 4,
                 mlp_ratios=[8] * 4,
                 drop_path_rate=0.15,
                 depths=[1] * 4,
                 num_stages=4,
                 reduction=4,
                 score_embed_nums=1,
                 num_scores=2,
                 mod_nums=1,
                 batch_size=1,
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        # 核心配置初始化
        self.num_stages = num_stages
        self.depths = depths
        self.reduction = reduction
        self.init_cfg = init_cfg
        self.extra_token_num = mod_nums + score_embed_nums

        # 输入和输出的投影层
        self.proj_in = nn.ModuleList(   
            [nn.Conv2d(embed_dim, embed_dim // reduction, kernel_size=1)
            for embed_dim in embed_dims])
        self.ir_proj_in = nn.ModuleList(   
            [nn.Conv2d(embed_dim, embed_dim // reduction, kernel_size=1)
            for embed_dim in embed_dims])
        self.proj_out = nn.ModuleList(
            [nn.Conv2d(embed_dim // reduction, embed_dim, kernel_size=1)
            for embed_dim in embed_dims])
        
        # 用于降低semantics的数量N
        self.pool = nn.AvgPool2d((7, 7), stride=7)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # 初始化多个模块
        self.proxy_embeds, self.blocks, self.norms, self.out_norms, self.norm_semantics, self.downsamples, self.ir_downsamples, self.scorenet = \
            [nn.ModuleList() for _ in range(8)]
        self.mod_embeds, self.score_embeds = nn.ParameterList(), nn.ParameterList()
        for i in range(num_stages):
            embed_dim = embed_dims[i] // reduction
            if i == 0:
                self._init_first_stage(embed_dim)
            else:
                self.proxy_embeds.append(SemanticEmbed(embed_dim_prev, embed_dim))
            embed_dim_prev = copy.deepcopy(embed_dim)

            block = nn.ModuleList([
                DualBlockMaskedEmbed(
                    dim=embed_dim,
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_path=dpr[cur + j],
                    norm_layer=nn.LayerNorm)
                for j in range(depths[i])])
            cur += depths[i]
            self.blocks.append(block if not with_cp else checkpoint_wrapper(block))

            # 初始化层归一化
            self.norms.append(nn.LayerNorm(embed_dim))
            self.out_norms.append(nn.LayerNorm(embed_dim))
            if i != num_stages - 1: 
                self.norm_semantics.append(nn.LayerNorm(embed_dim))
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(embed_dim)))
                self.ir_downsamples.append(
                    nn.Sequential(
                        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(embed_dim)))

            # 初始化模态和评分嵌入
            self.mod_embeds.append(nn.Parameter(torch.rand(1, mod_nums, embed_dim)))
            self.score_embeds.append(nn.Parameter(torch.rand(1, score_embed_nums, embed_dim)))
            self.scorenet.append(ScoreNet(dim=embed_dim, num_scores=num_scores))            
        
        self.mod_nums = mod_nums
        self.score_embed_nums = score_embed_nums        
        self.init_weights()

    def _init_first_stage(self, embed_dim):
        """初始化第一个阶段的特定层和参数"""
        self.q = nn.Parameter(torch.empty((64, embed_dim)), requires_grad=True)
        self.q_embed = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim))
        self.kv = nn.Linear(embed_dim, 2 * embed_dim)
        self.scale = embed_dim**-0.5
        self.proxy_ln = nn.LayerNorm(embed_dim)
        self.se = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2 * embed_dim))
        trunc_normal_(self.q, std=.02)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f"Training {self.__class__.__name__} from scratch.")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_fist_semantics(self, x, x_ir, x_weight, x_ir_weight):
        B, C, _, _ = x.shape
        x_down = self.pool(x)
        x_ir_down = self.pool(x_ir)
        x_down_H, x_down_W = x_down.shape[2:]
        x_down = x_down.view(B, C, -1).permute(0, 2, 1)  # B, HW, C
        x_ir_down = x_ir_down.view(B, C, -1).permute(0, 2, 1)  # B, HW, C

        x_down = torch.cat([x_down, x_ir_down], dim=1)  # B, 2HW, C
        kv = self.kv(x_down).view(B, -1, 2, C).permute(2, 0, 1, 3)  # 2, B, 2HW, C

        attn_self_q = self.q.reshape(8, 8, -1).permute(2, 0, 1) # C, 8, 8
        attn_self_q = F.interpolate(
            attn_self_q.unsqueeze(0),
            size=(x_down_H, x_down_W),
            mode='bicubic').squeeze(0).permute(1, 2, 0) # H, W, C

        attn_self_q = attn_self_q.reshape(-1, attn_self_q.shape[-1])    # HW, C
        attn_self_q = self.q_embed(attn_self_q) # HW, C
        bsz, src_len, _ = kv[0].shape

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

        attn_self_q = (attn_self_q @ kv[0].transpose(-1, -2)) * self.scale  # B, HW, 2HW

        # mask attention weight防止有的模态没有，赋值为-inf，后续经过softmax后就为0了
        attn_self_q = get_masked_attn_output_weights(
            attn_self_q, bsz=bsz, num_heads=1, tgt_len=M, src_len=src_len,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        attn_self_q = attn_self_q.softmax(-1)  # B, M, N    # B, HW, HW
        semantics = attn_self_q @ kv[1]  # B, M, C  # B, HW, C
        semantics = semantics.view(B, -1, C)    # B, HW, C

        x_down = (x_down[:, : total_patches, :] * x_weight +
                  x_down[:, total_patches:, :] * x_ir_weight) / (x_weight + x_ir_weight)    # B, HW, C
        semantics = torch.cat(
            [semantics.unsqueeze(2), x_down.unsqueeze(2)], dim=2)   # B, HW, 2, C

        se = self.se(semantics.sum(2).mean(1)).view(B, 2, C).softmax(1) # B, 2, C
        semantics = (semantics * se.unsqueeze(1)).sum(2)    # B, HW, C
        semantics = self.proxy_ln(semantics)
        return semantics

    def forward_sep(self, x, x_ir, x_weight, x_ir_weight, i, semantics=None):
        B, _, H, W = x.shape

        score_embed = self.score_embeds[i]
        mod_embed = self.mod_embeds[i]
        if score_embed.shape[0] != B:
            score_embed = score_embed.repeat(B, 1, 1)
        if mod_embed.shape[0] != B:
            mod_embed = mod_embed.repeat(B, 1, 1)

        if (i == 0) & (semantics is None):
            # 基本类似dualvit，多了多模态操作和mask操作
            semantics = self.get_fist_semantics(x, x_ir, x_weight, x_ir_weight)
        else:
            semantics = self.proxy_embeds[i - 1](semantics)

        # 这里包含了MAF和MAA两个向量    # 拼接模态特征
        x = torch.cat(
            [score_embed, mod_embed, x.flatten(2).permute(0, 2, 1), 
             x_ir.flatten(2).permute(0, 2, 1)], dim=1)

        # 对没有输入的模态进行mask处理，有的为0，没有的为1
        key_padding_mask = self._get_key_padding_mask(
            B, 
            N=x.shape[1],
            total_patches=(H * W + self.extra_token_num), 
            x_weight=x_weight, x_ir_weight=x_ir_weight, device=x.device)

        # 特征交互，提取
        for blk in self.blocks[i]:
            x, semantics = blk(x, H, W, semantics, self.extra_token_num, key_padding_mask, attn_mask=None)

        x = self.norms[i](x)
        extra_token = x[:, :self.extra_token_num, :]
        x = x[:, self.extra_token_num:, :]

        # 有可能需要变更score_embed和mod_embed        
        score_start_idx = 0
        score_end_idx = score_start_idx + self.score_embed_nums
        mod_emb_end_idx = score_end_idx + self.mod_nums
        score_embed = extra_token[:, :score_end_idx, :]
        mod_embed = extra_token[:, score_end_idx:mod_emb_end_idx, :]

        if self.score_embed_nums > 1:
            score_embed = (
                score_embed[:, 0, :] * x_weight + score_embed[:, 1, :] *
                x_ir_weight / (x_weight + x_ir_weight)).unsqueeze(0)

        if self.mod_nums > 1:
            mod_embed = (
                mod_embed[:, 0, :] * x_weight + mod_embed[:, 1, :] * 
                x_ir_weight / (x_weight + x_ir_weight)).unsqueeze(0)

        score_weight = self.scorenet[i](score_embed)    # B, 1, C -> 2, B, 1
        x, x_ir = x.chunk(2, dim=1)
        x = x * score_weight[0].unsqueeze(-1)
        x_ir = x_ir * score_weight[1].unsqueeze(-1)

        out_x = ((x * x_weight + x_ir * x_ir_weight) /
                    (x_weight + x_ir_weight)) + mod_embed

        out_x = self.out_norms[i](out_x)

        out_x = out_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if i < self.num_stages - 1:
            semantics = self.norm_semantics[i](semantics)

        return x.permute(0, 2, 1).view(B, -1, H, W), x_ir.permute(0, 2, 1).view(B, -1, H, W), semantics, out_x

    def forward(self, xs, xs_ir, x_weight=1, x_ir_weight=1):

        x_weight = torch.tensor([x_weight], requires_grad=False, device=xs[0].device)
        x_ir_weight = torch.tensor([x_ir_weight], requires_grad=False, device=xs[0].device)

        outs = []   # fused feats
        semantics = None
        for i, (x, x_ir) in enumerate(zip(xs, xs_ir)):
            if i == 0:
                x = self.proj_in[i](x)
                x_ir = self.ir_proj_in[i](x_ir)
            else:
                assert x.shape[-2:] == x_down.shape[-2:]
                x = self.proj_in[i](x) + x_down
                x_ir = self.ir_proj_in[i](x_ir) + x_ir_down

            x, x_ir, semantics, out = self.forward_sep(x, x_ir, x_weight, x_ir_weight, i, semantics)
            if i < self.num_stages - 1:
                x_down = self.downsamples[i](x)
                x_ir_down = self.ir_downsamples[i](x_ir)
            out = self.proj_out[i](out)
            outs.append(out)
        return tuple(outs)
    
    def _get_key_padding_mask(self, B, N, total_patches, x_weight, x_ir_weight, device):
        """生成Key Padding Mask"""
        mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        if x_weight == 1 and x_ir_weight == 0:
            mask[:, total_patches:] = True
        elif x_weight == 0 and x_ir_weight == 1:
            mask[:, self.extra_token_num:total_patches] = True
        return mask
    