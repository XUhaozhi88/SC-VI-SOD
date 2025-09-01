# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union
import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.init import normal_
from mmengine.runner.amp import autocast
from mmengine.model import BaseModel

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType
from .dino import DINO

from .dino_fusion_dwt_utils import FRDB, MySequential, cross_attention
from einops import rearrange

from ..layers import (DeformableDetrTransformerDecoder, DeformableDetrTransformerEncoder, SinePositionalEncoding)

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

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
        x1 = x[0:out_batch, :, :] / 2
        x2 = x[out_batch:out_batch * 2, :, :, :] / 2
        x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
        x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

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
    def __init__(self, in_channels, mid_channels, 
                 num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8], ffn_factor = 4.0):
        self.rgb_dwt = DWT()
        self.rgb_idwt = IWT()
        self.ir_dwt = DWT()
        self.ir_idwt = IWT()

        self.LL_weight = nn.Parameter(torch.Tensor(0.5), requires_grad=True)
        self.HL_LH_HH_weight = nn.Parameter(torch.Tensor(0.5), requires_grad=True)

        self.BAM = BAM(2 * in_channels)

        self.LL_patch_embed = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.HL_LH_HH_patch_embed = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # 1st block
        self.encoder_level1 = FRDB(nChannels=mid_channels) 
        self.encoder_level11 = DeformableDetrTransformerEncoder(
            num_layers=num_blocks[0], 
            layer_cfg=dict(self_attn_cfg=dict(embed_dims=256, batch_first=True), 
                           ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
            num_cp=-1)        
        self.down1_2 = nn.Identity()
        self.down1_21 = nn.Identity()

        # 2nd block
        self.encoder_level2 = FRDB(nChannels=mid_channels * 2 ** 1)
        self.encoder_level21 = DeformableDetrTransformerEncoder(
            num_layers=num_blocks[1], 
            layer_cfg=dict(self_attn_cfg=dict(embed_dims=256, batch_first=True), 
                           ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
            num_cp=-1)        
        self.down2_3 = nn.Identity()
        self.down2_31 = nn.Identity()

        self.cross_attention0 = cross_attention(dim=int(mid_channels * 2 ** 2), num_heads=8)

    def forward_dft(self, x_LL, x_HL_LH_HH):

        inp_enc_level1 = self.LL_patch_embed(x_LL)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        inp_enc_level11 = self.HL_LH_HH_patch_embed(x_HL_LH_HH)
        inp_enc_level11 = inp_enc_level11.view(*inp_enc_level11.shape[:2], -1).permute(0, 2, 1)
        out_enc_level11 = self.encoder_level11(inp_enc_level11)
        out_enc_level11 = out_enc_level11.permute(0, 2, 1).view(*inp_enc_level11.shape)
        inp_enc_level21 = self.down1_21(out_enc_level11)
        inp_enc_level21 = inp_enc_level21.view(*inp_enc_level21.shape[:2], -1).permute(0, 2, 1)
        out_enc_level21 = self.encoder_level21(inp_enc_level21)
        out_enc_level21 = out_enc_level21.permute(0, 2, 1).view(*inp_enc_level21.shape)
        inp_enc_level31 = self.down2_31(out_enc_level21)

        x_HH, x_LL = self.cross_attention0(inp_enc_level3, inp_enc_level31)
        x_LL1 = self.upL1(x_LL)
        x_HH1 = self.upH1(x_HH)



        latent = self.latent(inp_enc_level3)
        #latent,_ = self.latent(inp_enc_level4, t)
        latent1 = self.latent1(inp_enc_level31)
        #print(latent.shape)
        #print(latent1.shape)
        #print( x_HH_LH.shape)
        # inp_dec_level3 = self.up4_3(latent+x_LL)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level2], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # #out_dec_level3,_ = self.decoder_level3(inp_dec_level3, t)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(latent+x_LL)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2= self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2+x_LL1)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1)

        x_HH=torch.cat((x_HH,x_HH,x_HH),dim=0)
        # inp_dec_level31 = self.up4_31(latent1+x_HH)
        # inp_dec_level31 = torch.cat([inp_dec_level31, out_enc_level31], 1)
        # inp_dec_level31 = self.reduce_chan_level31(inp_dec_level31)
        # out_dec_level31= self.decoder_level31(inp_dec_level31)
        
        inp_dec_level21 = self.up3_21(latent1+x_HH)
        inp_dec_level21 = torch.cat([inp_dec_level21, out_enc_level21], 1)
        inp_dec_level21 = self.reduce_chan_level21(inp_dec_level21)
        out_dec_level21 = self.decoder_level21(inp_dec_level21)
        x_HH1=torch.cat((x_HH1,x_HH1,x_HH1),dim=0)
        inp_dec_level11 = self.up2_11(out_dec_level21+x_HH1)
        inp_dec_level11 = torch.cat([inp_dec_level11, out_enc_level11], 1)
        out_dec_level11 = self.decoder_level11(inp_dec_level11)

        #### For Dual-Pixel Defocus Deblurring Task ####
        
        ###########################
       
        out_dec_level11 = self.output(out_dec_level11)

        out_dec_level=idwt(torch.cat((out_dec_level1,out_dec_level11),dim=0))

        return out_dec_level, out_dec_level1, out_dec_level11

    
    def forward(self, x_rgb, x_ir):
        bs, channels, _, _ = x_rgb.shape

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
        x = self.idwt(x_LL, x_HL_LH_HH)

        return x

    def fusion_LL(self, x_rgb_LL, x_ir_LL):
        return self.LL_weight * x_rgb_LL + (1.0 - self.LL_weight) * x_ir_LL

    def fusion_HL_LH_HH(self, x_rgb_HL_LH_HH, x_ir_HL_LH_HH):
        return self.HL_LH_HH_weight * x_rgb_HL_LH_HH + (1.0 - self.HL_LH_HH_weight) * x_ir_HL_LH_HH


@MODELS.register_module()
class DINO_Fusion(DINO):

    # def __init__(self, *args, 
    #              mod_list=["img", "ir_img"], **kwargs) -> None:
    #     super().__init__(*args, **kwargs)
    #     self.mod_list = mod_list  
        
    #     # 初始化结构
    #     backbone = nn.ModuleList()
    #     if self.with_neck: neck = nn.ModuleList()
    #     for _ in range(len(self.mod_list)):
    #         backbone.append(copy.deepcopy(self.backbone))
    #         if self.with_neck: neck.append(copy.deepcopy(self.neck))
    #     self.backbone = backbone
    #     if self.with_neck: self.neck = neck

        # 融合
        # self.fusion = Fusion(channels=256, reduction=4, num_layer=4)
        # self.fusions = nn.ModuleList(SingleScaleFusion(channels=256, reduction=4) 
        #                             for _ in range(4))
        # self.fusions = Dual_Fusion(with_cp=True, batch_size=batch_size)
        # self.fusions = Dual_Fusion_New()#with_cp=True)        
        
    def extract_feat1(self, batch_inputs: Tensor,
                    batch_extra_inputs: Tensor) -> Tuple[Tensor]:
        xs = []
        for i, backbone in enumerate(self.backbone):
            # 选择输入：第一个模态使用 batch_inputs，其余模态使用 batch_extra_inputs
            inputs = batch_inputs if i == 0 else batch_extra_inputs[i - 1]

            # 提取 backbone 特征
            x = backbone(inputs)            
            if self.with_neck: 
                neck = self.neck[i]
                x = neck(x)  
            xs.append(x)

            if i == 0: 
                out = list(x)
            else:
                for level in range(len(out)):
                    out[level] += x[level]
        
        ## 简单的sum融合
        # out = [torch.sum(torch.stack([x[i] for x in xs], dim=0), dim=0)
        #         for i in range(len(xs[0]))]
        ## DWT融合
        # out = [fusion(rgb_x, ir_x) for rgb_x, ir_x, fusion in zip(xs[0], xs[1], self.fusions)]
        return tuple(out)   # fusion
        # return self.fusions(*xs)
    
    def extract_feat(self, batch_inputs: Tensor,
                    batch_extra_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs, batch_extra_inputs[0])            
        if self.with_neck: 
            x = self.neck(x)
        return x
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,             
             batch_extra_inputs: Tensor,
             batch_extra_data_samples: SampleList) -> Union[dict, list]:
        
        img_feats = self.extract_feat(batch_inputs, batch_extra_inputs)       

        for img_feat in img_feats:
            if torch.isnan(img_feat).any(): # search nan
                raise AssertionError
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs,
                batch_data_samples, batch_extra_inputs, rescale: bool = True):

        # image feature extraction
        img_feats = self.extract_feat(batch_inputs, batch_extra_inputs)

        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def forward(self,
                inputs: torch.Tensor,
                data_samples=None,
                mode: str = 'tensor',
                **kwargs):
        extra_inputs = kwargs.get('extra_inputs', None)
        extra_data_samples = kwargs.get('extra_data_samples', None)

        if extra_inputs is None:
            super().forward(inputs, data_samples, mode)
        else:
            if mode == 'loss':
                return self.loss(inputs, data_samples, extra_inputs, extra_data_samples)
            elif mode == 'predict':
                return self.predict(inputs, data_samples, extra_inputs)
            elif mode == 'tensor':
                return self._forward(inputs, data_samples)
            else:
                raise RuntimeError(f'Invalid mode "{mode}". '
                                'Only supports loss, predict and tensor mode')