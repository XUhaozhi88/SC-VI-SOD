import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmengine.logging import MMLogger
from mmengine.model.weight_init import trunc_normal_

from mmengine.model import BaseModule


from ..backbones.unixvit import (DualVit, DualBlock, PVT2FFNEmbed, DualMaskedAttention,
                                 SemanticEmbed, ScoreNet, get_masked_attn_output_weights)

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

class DualBlockMaskedEmbed(DualBlock):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super(DualBlockMaskedEmbed, self).__init__(dim=dim,
                                                   num_heads=num_heads,
                                                   mlp_ratio=mlp_ratio,
                                                   drop_path=drop_path,
                                                   norm_layer=norm_layer)
        self.mlp = PVT2FFNEmbed(in_features=dim,
                                hidden_features=int(dim * mlp_ratio))
        self.attn = DualMaskedAttention(dim, num_heads, drop_path=drop_path)

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

class Dual_Fusion(BaseModule):

    def __init__(self,
                #  embed_dims=[64, 128, 320, 448],
                #  num_heads=[2, 4, 10, 14],
                #  mlp_ratios=[8, 8, 4, 3],
                 embed_dims=[256] * 4,
                 num_heads=[8] * 4,
                 mlp_ratios=[8] * 4,
                 drop_path_rate=0.15,
                #  depths=[3, 4, 6, 3],
                 depths=[3] * 4,
                 num_stages=4,
                 reduction=4,
                 score_embed_nums=1,
                 num_scores=2,
                 mod_nums=1,
                 batch_size=1,
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.depths = depths
        self.num_stages = num_stages

        self.proj_in = nn.ModuleList(   # share param rgb/ir
            [nn.Conv2d(embed_dim, embed_dim // reduction, kernel_size=1, stride=1)
            for embed_dim in embed_dims])
        self.proj_out = nn.ModuleList(
            [nn.Conv2d(embed_dim // reduction, embed_dim, kernel_size=1, stride=1)
            for embed_dim in embed_dims])
        
        # 用于降低semantics的数量N
        self.pool = nn.AvgPool2d((7, 7), stride=7)
        self.pool_ir = nn.AvgPool2d((7, 7), stride=7)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.blocks, self.norms, self.out_norms, self.norm_semantics = \
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(num_stages):
            embed_dim = embed_dims[i] // reduction
            if i == 0:
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
            else:
                semantic_embed = SemanticEmbed(embed_dim_later, embed_dim)
                setattr(self, f'proxy_embed{i + 1}', semantic_embed)
            embed_dim_later = copy.deepcopy(embed_dim)

            block = nn.ModuleList([
                DualBlockMaskedEmbed(
                    dim=embed_dim,
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_path=dpr[cur + j],
                    norm_layer=nn.LayerNorm)
                for j in range(depths[i])])
            if with_cp:
                block = checkpoint_wrapper(block)

            norm = nn.LayerNorm(embed_dim)
            norm_proxy = nn.LayerNorm(embed_dim)
            out_norm = nn.LayerNorm(embed_dim)
            cur += depths[i]

            self.blocks.append(block)
            self.norms.append(norm)
            self.out_norms.append(out_norm)
            if i != num_stages - 1: self.norm_semantics.append(norm_proxy)

        self.mod_nums = mod_nums
        self.mod_embed = None
        self.score_embed = None

        self.mod_embeds = nn.ParameterList([
            nn.Parameter(torch.rand([batch_size, self.mod_nums, embed_dim]))
            for _ in range(num_stages)])

        self.score_embed_nums = score_embed_nums
        self.num_scores = num_scores

        self.score_embeds = nn.ParameterList([
            nn.Parameter(
                torch.rand([batch_size, self.score_embed_nums, embed_dim]))
            for _ in range(num_stages)])

        self.scorenet = nn.ModuleList([
            ScoreNet(dim=(embed_dims[i] // reduction), num_scores=self.num_scores)
            for i in range(num_stages)])
        self.extra_token_num = self.mod_nums + self.score_embed_nums
        
        self.init_weights()

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
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        m.bias.data.zero_()

    def get_fist_semantics(self, x, x_ir, x_weight, x_ir_weight):
        B, C, _, _ = x.shape
        x_down = self.pool(x)
        x_ir_down = self.pool(x_ir)
        x_down_H, x_down_W = x_down.shape[2:]
        x_down = x_down.view(B, C, -1).permute(0, 2, 1).contiguous()  # B, HW, C
        x_ir_down = x_ir_down.view(B, C, -1).permute(0, 2, 1).contiguous()  # B, HW, C

        x_down = torch.cat([x_down, x_ir_down], dim=1)  # B, 2HW, C
        kv = self.kv(x_down).view(B, -1, 2, C).permute(2, 0, 1, 3).contiguous()  # 2, B, 2HW, C

        attn_self_q = self.q.reshape(8, 8, -1).permute(2, 0, 1).contiguous() # C, 8, 8
        attn_self_q = F.interpolate(
            attn_self_q.unsqueeze(0),
            size=(x_down_H, x_down_W),
            mode='bicubic').squeeze(0).permute(1, 2, 0).contiguous() # H, W, C

        attn_self_q = attn_self_q.reshape(-1, attn_self_q.shape[-1]).contiguous()    # HW, C
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

        attn_self_q = (attn_self_q @ kv[0].transpose(-1, -2).contiguous()) * self.scale  # B, HW, 2HW

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
        B, C, H, W = x.shape
        
        score_start_idx = 0
        score_end_idx = score_start_idx + self.score_embed_nums

        mod_emb_end_idx = score_end_idx + self.mod_nums

        score_embed = self.score_embeds[i]
        mod_embed = self.mod_embeds[i]

        if (i == 0) & (semantics is None):
            # 基本类似dualvit，多了多模态操作和mask操作
            semantics = self.get_fist_semantics(x, x_ir, x_weight, x_ir_weight)
        else:
            semantics_embed = getattr(self, f'proxy_embed{i + 1}')
            semantics = semantics_embed(semantics)

        # 这里包含了MAF和MAA两个向量
        x = torch.cat([score_embed, mod_embed, 
            x.view(B, C, H*W).permute(0, 2, 1).contiguous(), 
            x_ir.view(B, C, H*W).permute(0, 2, 1).contiguous()], dim=1)

        N = H * W * 2 + self.extra_token_num    # extra_token_num = mod_nums + score_embed_nums
        key_padding_mask = torch.zeros((B, N), dtype=torch.bool).to(x.device)
        attn_mask = None
        total_patches = H * W + self.extra_token_num
        # 对没有输入的模态进行mask处理，有的为0，没有的为1
        if x_weight == 1 and x_ir_weight == 0:
            key_padding_mask[:, total_patches:] = True  # B, HW

        elif x_weight == 0 and x_ir_weight == 1:
            key_padding_mask[:, self.extra_token_num:total_patches] = True

        for blk in self.blocks[i]:
            x, semantics = blk(x, H, W, semantics, self.extra_token_num, key_padding_mask, attn_mask)

        x = self.norms[i](x)
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

        return semantics, out_x

        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x_ir = x_ir.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # x = torch.cat([x, x_ir], dim=0) # 我用这个作为输出，不就可以用来处理后续的内容了，在encoder那里做做文章

        # return x, semantics, out_x

    def forward(self, xs, xs_ir, x_weight=1, x_ir_weight=1):

        x_weight = torch.tensor([x_weight], requires_grad=False, device=xs[0].device)
        x_ir_weight = torch.tensor([x_ir_weight], requires_grad=False, device=xs[0].device)

        # x_out = dict(rgb=[], ir=[])  # per-modal feats
        outs = []   # fused feats
        semantics = None
        for i, (x, x_ir) in enumerate(zip(xs, xs_ir)):
            x = self.proj_in[i](x)
            x_ir = self.proj_in[i](x_ir)
            semantics, out = self.forward_sep(x, x_ir, x_weight, x_ir_weight, i, semantics)
            # x, semantics, out = self.forward_sep(x, x_ir, x_weight, x_ir_weight, i, semantics)
            # x_out['rgb'].append(x[[0]])
            # x_out['ir'].append(x[[1]])
            out = self.proj_out[i](out)
            outs.append(out)
        return tuple(outs)