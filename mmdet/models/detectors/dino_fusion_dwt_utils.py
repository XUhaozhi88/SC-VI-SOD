import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return x_LL, torch.cat((x_HL, x_LH, x_HH), 0)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x, bs):
        _, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = bs, in_channel, 2 * in_height, 2 * in_width
        x1 = x[0:out_batch, :, :] / 2
        x2 = x[out_batch:out_batch * 2, :, :, :] / 2
        x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
        x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

        # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
        h = torch.zeros([out_batch, out_channel, out_height, out_width], device=x.device, dtype=torch.float32)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h


class BAM(nn.Module):
    def __init__(self, in_channels):
        super(BAM, self).__init__()
        
        # 通道注意力部分
        self.channel_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.channel_fc1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, bias=False)  # 瓶颈
        self.channel_fc2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, bias=False)  # 恢复通道数
        
        # 空间注意力部分
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)  # 空间卷积

    def forward(self, x):
        # 通道注意力机制
        avg_pool = self.channel_pool(x)  # 进行全局平均池化
        channel_att = F.relu(self.channel_fc1(avg_pool))  # 瓶颈层
        channel_att = self.channel_fc2(channel_att)  # 恢复通道数
        channel_att = torch.sigmoid(channel_att)  # Sigmoid激活获得通道权重

        # 空间注意力机制
        spatial_att = self.spatial_conv(x)  # 计算空间注意力
        spatial_att = torch.sigmoid(spatial_att)  # Sigmoid激活获得空间权重

        # 融合通道注意力和空间注意力
        x = x * channel_att * spatial_att  # 最终的输出是通道注意力和空间注意力加权后的特征图

        return x
    
    
class SingleScaleFusion(nn.Module):
    def __init__(self, in_channels, mid_channels, bias=False,
                #  num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8], 
                #  num_blocks=[2, 2, 2, 2], 
                 num_blocks=[1, 1, 1, 1], 
                 heads=[8, 8, 8], 
                 ffn_factor = 4.0):
        super(SingleScaleFusion, self).__init__()
        self.rgb_dwt = DWT()
        self.ir_dwt = DWT()
        self.idwt = IWT()

        self.LL_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.HL_LH_HH_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.BAM = BAM(2 * in_channels)

        self.LL_patch_embed = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.H_patch_embed = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # 1st block
        self.block_LL_level1 = FRDB(nChannels=mid_channels)         
        self.block_H_level1 = nn.Sequential(*[
            DTB(dim=mid_channels, num_heads=heads[0], ffn_factor=ffn_factor, bias=bias) 
            for _ in range(num_blocks[0])]) 
        
        # 2nd block
        self.block_LL_level2 = FRDB(nChannels=mid_channels)
        self.block_H_level2 = nn.Sequential(*[
            DTB(dim=mid_channels, num_heads=heads[1], ffn_factor=ffn_factor, bias=bias) 
            for _ in range(num_blocks[1])])

        # 
        self.cross_attention = cross_attention(dim=mid_channels, num_heads=8)

        # 3rd block
        self.block_LL_level3 = FRDB(nChannels=mid_channels)
        self.block_H_level3 = nn.Sequential(*[
            DTB(dim=mid_channels, num_heads=heads[2], ffn_factor=ffn_factor, bias=bias) 
            for _ in range(num_blocks[2])])

        # 4th block
        self.reduce_channel_LL_level4 = nn.Conv2d(mid_channels * 2 ** 1, mid_channels, kernel_size=1, bias=bias)
        self.reduce_channel_H_level4 = nn.Conv2d(mid_channels * 2 ** 1, mid_channels, kernel_size=1, bias=bias)
        self.block_LL_level4 = FRDB(nChannels=mid_channels)
        self.block_H_level4 = nn.Sequential(*[
            DTB(dim=mid_channels, num_heads=heads[1], ffn_factor=ffn_factor, bias=bias) 
            for _ in range(num_blocks[1])])

        # 5th block
        self.reduce_channel_LL_level5 = nn.Conv2d(mid_channels * 2 ** 1, mid_channels, kernel_size=1, bias=bias)
        self.reduce_channel_H_level5 = nn.Conv2d(mid_channels * 2 ** 1, mid_channels, kernel_size=1, bias=bias)
        self.block_LL_level5 = FRDB(nChannels=mid_channels)
        self.block_H_level5 = nn.Sequential(*[
            DTB(dim=mid_channels, num_heads=heads[0], ffn_factor=ffn_factor, bias=bias) 
            for _ in range(num_blocks[0])])
        
        # output
        self.output_LL = nn.Conv2d(mid_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_H = nn.Conv2d(mid_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    
    def fusion_LL(self, x_rgb_LL, x_ir_LL):
        return self.LL_weight * x_rgb_LL + (1.0 - self.LL_weight) * x_ir_LL

    def fusion_HL_LH_HH(self, x_rgb_HL_LH_HH, x_ir_HL_LH_HH):
        return self.HL_LH_HH_weight * x_rgb_HL_LH_HH + (1.0 - self.HL_LH_HH_weight) * x_ir_HL_LH_HH

    def forward_dft(self, x_LL, x_H):
        """
            Input:
            x_LL: Tensor(bs, c, h, w)
            x_H: Tensor(3*bs, c, h, w)
        """
        # projection
        x_LL_level1 = self.LL_patch_embed(x_LL) # bs, 256, h, w -> bs, 64, h, w
        x_H_level1 = self.H_patch_embed(x_H)    # 3*bs, 256, h, w -> 3*bs, 64, h, w

        # block 1        
        x_LL_level1 = self.block_LL_level1(x_LL_level1) # bs, 64, h, w
        x_H_level1 = self.block_H_level1(x_H_level1)    # 3*bs, 64, h, w

        # block 2
        x_LL_level2 = self.block_LL_level2(x_LL_level1) # bs, 64, h, w        
        x_H_level2 = self.block_H_level2(x_H_level1)    # 3*bs, 64, h, w

        # cross attention module
        x_H, x_LL = self.cross_attention(x_LL_level2, x_H_level2) # (bs, 64, h, w) \ (3*bs, 64, h, w) 

        # block 3
        x_LL_level3 = self.block_LL_level3(x_LL_level2) # bs, 64, h, w
        x_H_level3 = self.block_H_level3(x_H_level2)    # 3*bs, 64, h, w

        # block 4
        x_LL_level4 = x_LL_level3 + x_LL
        x_LL_level4 = torch.cat([x_LL_level4, x_LL_level2], 1)  # (bs, 64, h, w)*2 -> bs, 128, h, w
        x_LL_level4 = self.reduce_channel_LL_level4(x_LL_level4)# bs, 128, h, w -> bs, 64, h, w
        x_LL_level4 = self.block_LL_level4(x_LL_level4)         # bs, 64, h, w

        # x_H = torch.cat((x_H, x_H, x_H),dim=0)                  # (bs, 64, h, w)*3 -> 3*bs, 64, h, w        
        x_H_level4 = x_H_level3 + x_H                           # 3*bs, 64, h, w
        x_H_level4 = torch.cat([x_H_level4, x_H_level2], 1)     # (3*bs, 64, h, w)*2 -> 3*bs, 128, h, w
        x_H_level4 = self.reduce_channel_H_level4(x_H_level4)   # 3*bs, 128, h, w -> 3*bs, 64, h, w
        x_H_level4 = self.block_H_level4(x_H_level4)            # 3*bs, 64, h, w

        # block 5
        x_LL_level5 = x_LL_level4 + x_LL
        x_LL_level5 = torch.cat([x_LL_level5, x_LL_level1], 1)  # (bs, 64, h, w)*2 -> bs, 128, h, w
        x_LL_level5 = self.reduce_channel_LL_level5(x_LL_level5)# bs, 128, h, w -> bs, 64, h, w
        x_LL_level5 = self.block_LL_level5(x_LL_level5)         # bs, 64, h, w
        
        x_H_level5 = x_H_level4 + x_H                           # 3*bs, 64, h, w
        x_H_level5 = torch.cat([x_H_level5, x_H_level1], 1)     # (3*bs, 64, h, w)*2 -> 3*bs, 128, h, w
        x_H_level5 = self.reduce_channel_H_level5(x_H_level5)   # 3*bs, 128, h, w -> 3*bs, 64, h, w
        x_H_level5 = self.block_H_level5(x_H_level5)            # 3*bs, 64, h, w
        
        # output
        output_LL = self.output_LL(x_LL_level5) # bs, 64, h, w -> # bs, 256, h, w
        out_H = self.output_H(x_H_level5)       # 3*bs, 64, h, w -> # 3*bs, 256, h, w

        return output_LL, out_H

    
    def forward(self, x_rgb, x_ir):
        bs, channels, height, width = x_rgb.shape

        # 使用BAM进行通道和空间上的初步对齐
        x = torch.cat((x_rgb, x_ir), dim=1)
        x = self.BAM(x)
        x_rgb, x_ir = torch.split(x, [channels, channels], dim=1)
        # 离散小波分解
        x_rgb_LL, x_rgb_HL_LH_HH = self.rgb_dwt(x_rgb)
        x_ir_LL, x_ir_HL_LH_HH = self.ir_dwt(x_ir)
        # 融合
        x_LL = self.fusion_LL(x_rgb_LL, x_ir_LL)
        x_HL_LH_HH = self.fusion_HL_LH_HH(x_rgb_HL_LH_HH, x_ir_HL_LH_HH)
        # 离散小波重构
        x_LL, x_HL_LH_HH = self.forward_dft(x_LL, x_HL_LH_HH)
        x = self.idwt(torch.cat((x_LL, x_HL_LH_HH), dim=0), bs)
        return x
    
class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1).contiguous()
        x = self.body(x)
        return x.permute(0, 2, 1).view(b, c, h, w).contiguous()


##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class DTB(nn.Module):

    def __init__(self, dim, num_heads, ffn_factor, bias):
        super(DTB, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_factor, bias)

    def forward(self, x):       
        x = self.norm1(x)
        x = x + self.attn(x.contiguous())
        x = self.norm2(x)
        x = x + self.ffn(x.contiguous())
        return x

#####################################Diffusion Transformer DFT################################

class cross_attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.valueh = Depth_conv(in_ch=dim, out_ch=dim)
        self.valuel = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, hidden_states, ctx):
        n, c, h, w = hidden_states.shape
        ctx1 = ctx[:n, ...]
        ctx2 =  ctx[n:n+n, ...]
        ctx3 =  ctx[n+n:, ...]
        ctx=ctx1+ctx2+ctx3
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layerh = self.valueh(ctx)
        mixed_value_layerl = self.valuel(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layerh = self.transpose_for_scores(mixed_value_layerh)
        value_layerl = self.transpose_for_scores(mixed_value_layerl)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2).contiguous())
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layerh = torch.matmul(attention_probs, value_layerh)
        ctx_layerh = ctx_layerh.permute(0, 2, 1, 3).contiguous()

        ctx_layerl = torch.matmul(attention_probs, value_layerl)
        ctx_layerl = ctx_layerl.permute(0, 2, 1, 3).contiguous()

        return ctx_layerh,ctx_layerl

class make_fdense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=1):
        super(make_fdense, self).__init__()
        #self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              #bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False),nn.BatchNorm2d(growthRate)
        )
        self.bat = nn.BatchNorm2d(growthRate),
        self.leaky=nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        out = self.leaky(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
    
class FRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer=1, growthRate=32):
        super(FRDB, self).__init__()
        nChannels_1 = nChannels
        nChannels_2 = nChannels
        modules1 = []
        for _ in range(nDenselayer):
            modules1.append(make_fdense(nChannels_1, growthRate))
            nChannels_1 += growthRate
        self.dense_layers1 = nn.Sequential(*modules1)
        modules2 = []
        for _ in range(nDenselayer):
            modules2.append(make_fdense(nChannels_2, growthRate))
            nChannels_2 += growthRate
        self.dense_layers2 = nn.Sequential(*modules2)
        self.conv_1 = nn.Conv2d(nChannels_1, nChannels, kernel_size=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(nChannels_2, nChannels, kernel_size=1, padding=0, bias=False)
        self.SRDB=SRDB(nChannels)

    def forward(self, x):
        x=self.SRDB(x)
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.dense_layers1(mag)
        mag = self.conv_1(mag)
        pha = self.dense_layers2(pha)
        pha = self.conv_2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        out = out + x
        return out


class SRDB(nn.Module):
    def __init__(self, nChannels, growthRate=64):
        super(SRDB, self).__init__()
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=(1 - 1) // 2, bias=False)
        self.conv2 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.conv3 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2, bias=False)
        self.conv4 = nn.Conv2d(growthRate*3, nChannels, kernel_size=1, padding=(1 - 1) // 2, bias=False)
        self.leaky1=nn.LeakyReLU(0.1, inplace=True)
        self.leaky2=nn.LeakyReLU(0.1, inplace=True)
        self.leaky3=nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x_1 = self.leaky1(self.conv1(x))
        x_2 = self.leaky2(self.conv2(x))
        x_3 = self.leaky3(self.conv3(x))
        x_0 = torch.cat((x_1, x_2, x_3), dim=1)
        x_0 = self.conv4(x_0)
        x = x_0 + x
        return x
