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

from .dual_fusion_new import Dual_Fusion_New

# from .dual_fusion import Dual_Fusion
# from .dual_fusion_new import Dual_Fusion_New

class Fusion(BaseModel):
    def __init__(self, 
                 channels: int,
                 reduction: int,
                 num_layer: int=4) -> None:
        super().__init__()
        self.channels = channels
        mid_channels = channels // reduction

        # rgb
        self.rgb_LL_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
            for _ in num_layer)
        self.rgb_LH_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
            for _ in num_layer)
        self.rgb_HL_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
            for _ in num_layer)
        self.rgb_HH_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
            for _ in num_layer)
        self.init_weights(self.rgb_LL_convs)
        self.init_weights(self.rgb_LH_convs)
        self.init_weights(self.rgb_HL_convs)
        self.init_weights(self.rgb_HH_convs)

        # ir
        self.ir_LL_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
            for _ in num_layer)
        self.ir_LH_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
            for _ in num_layer)
        self.ir_HL_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
            for _ in num_layer)
        self.ir_HH_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
            for _ in num_layer)
        self.init_weights(self.ir_LL_convs)
        self.init_weights(self.ir_LH_convs)
        self.init_weights(self.ir_HL_convs)
        self.init_weights(self.ir_HH_convs)
        
        # fused
        self.LL = nn.Parameter(0.5, requires_grad=True)
        self.LH = nn.Parameter(0.5, requires_grad=True)
        self.HL = nn.Parameter(0.5, requires_grad=True)
        self.HH = nn.Parameter(0.5, requires_grad=True)
        self.L_fused_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
            for _ in num_layer)
        self.H_fused_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
            for _ in num_layer)
        self.fused_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
            for _ in num_layer)
        self.init_weights(self.H_fused_convs)
        self.init_weights(self.L_fused_convs)
        self.init_weights(self.fused_convs)

        # inital dwt/idwt weight
        # Haar 小波滤波器
        self.dwt_low_pass = torch.tensor([0.5, 0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1)  # 低通滤波器
        self.dwt_high_pass = torch.tensor([0.5, -0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1)  # 高通滤波器
        self.idwt_low_pass = torch.tensor([0.5, 0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1)  # 低通滤波器
        self.idwt_high_pass = torch.tensor([0.5, -0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1)  # 高通滤波器

        # 转为可学习参数
        self.register_buffer('dwt_low_pass', self.dwt_low_pass)
        self.register_buffer('dwt_high_pass', self.dwt_high_pass)
        self.register_buffer('idwt_low_pass', self.idwt_low_pass)
        self.register_buffer('idwt_high_pass', self.idwt_high_pass)

    def init_weights(self, module) -> None:
        """Initialize weights for components."""
        for submodule in module.modules():
            if isinstance(submodule, nn.Conv2d):
                # 对卷积层使用 Kaiming 初始化
                nn.init.kaiming_normal_(submodule.weight, mode='fan_out', nonlinearity='relu')
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.Linear):
                # 对线性层使用 Xavier 初始化
                nn.init.xavier_normal_(submodule.weight)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.BatchNorm2d):
                # 对 BatchNorm 层的 weight 设置为1，bias 设置为0
                nn.init.constant_(submodule.weight, 1)
                nn.init.constant_(submodule.bias, 0)

    # 离散小波变换 (DWT)
    def dwt2d(self, input: torch.Tensor, channel: int):
        """
        执行2D离散小波变换 (DWT)，提取低频和高频分量
        Args:
            input (Tensor): 输入张量，形状为 (B, C, H, W)
        Returns:
            四个子带 (LowLow, LowHigh, HighLow, HighHigh)
        """
        
        input = F.pad(input, (1, 1, 0, 0), mode='reflect')  # 填充列，复杂填充无法直接用conv中的padding

        # 进行卷积操作，得到低频和高频分量
        low = F.conv2d(input, self.dwt_low_pass, stride=(1, 2), groups=channel)
        high = F.conv2d(input, self.dwt_high_pass, stride=(1, 2), groups=channel)        
        
        low = F.pad(low, (0, 0, 1, 1), mode='reflect')  # 填充行
        high = F.pad(high, (0, 0, 1, 1), mode='reflect')

        LowLow = F.conv2d(low, self.dwt_low_pass.transpose(2, 3), groups=channel, stride=(2, 1))
        LowHigh = F.conv2d(low, self.dwt_high_pass.transpose(2, 3), groups=channel, stride=(2, 1))
        HighLow = F.conv2d(high, self.dwt_low_pass.transpose(2, 3), groups=channel, stride=(2, 1))
        HighHigh = F.conv2d(high, self.dwt_high_pass.transpose(2, 3), groups=channel, stride=(2, 1))

        return LowLow, LowHigh, HighLow, HighHigh

    # 逆离散小波变换 (IDWT)
    def idwt2d(self, low: torch.Tensor, high: torch.Tensor, channel: int):
        """
        执行2D逆离散小波变换 (IDWT)，将低频和高频分量恢复为原始图像
        Args:
            param low: 低频分量
            param high: 高频分量
        Returns:
            output (Tensor): 恢复后的图像
        """

        # 使用逆卷积 (转置卷积) 恢复图像
        low = F.conv_transpose2d(low, self.idwt_low_pass.transpose(2, 3), stride=(2, 1), groups=channel)
        high = F.conv_transpose2d(high, self.idwt_high_pass.transpose(2, 3), stride=(2, 1), groups=channel)

        # 恢复后的图像是低频和高频分量的和
        output = low + high
        return output

    def forward(self, rgb_inputs, ir_inputs):
        outputs = []
        for rgb_input, ir_input, \
            rgb_LL_conv, rgb_LH_conv, rgb_HL_conv, rgb_HH_conv, \
            ir_LL_conv, ir_LH_conv, ir_HL_conv, ir_HH_conv, \
            L_fused_conv, H_fused_conv, fused_conv \
            in zip(rgb_inputs, ir_inputs, \
                    self.rgb_LL_convs, self.rgb_LH_convs, self.rgb_HL_convs, self.rgb_HH_convs, \
                    self.ir_LL_convs, self.ir_LH_convs, self.ir_HL_convs, self.ir_HH_convs, \
                    self.L_fused_convs, self.H_fused_convs, self.fused_convs):

            # 输入特征图的高度和宽度必须为偶数
            _, _, Height, Width = rgb_input.shape
            if Height % 2 != 0:
                phd = (0, 0, 0, 1)
                is_h_pad = True
                rgb_input = F.pad(rgb_input, phd, mod="reflect")
                ir_input = F.pad(ir_input, phd, mod="reflect")                
            else:
                is_h_pad = False
            if Width % 2 != 0:
                pwd = (0, 1)
                is_w_pad = True
                rgb_input = F.pad(rgb_input, pwd, mod="reflect")
                ir_input = F.pad(ir_input, pwd, mod="reflect")     
            else:
                is_w_pad = False

            # Discrete Wavelet Transformation
            rgb_LL, rgb_LH, rgb_HL, rgb_HH = self.dwt2d(rgb_input, channel=self.channels)
            ir_LL, ir_LH, ir_HL, ir_HH = self.dwt2d(ir_input, channel=self.channels)

            # residual refine freq
            rgb_LL = rgb_LL + rgb_LL_conv(rgb_LL)
            rgb_LH = rgb_LH + rgb_LH_conv(rgb_LH)
            rgb_HL = rgb_HL + rgb_HL_conv(rgb_HL)
            rgb_HH = rgb_HH + rgb_HH_conv(rgb_HH)
            ir_LL = ir_LL + ir_LL_conv(ir_LL)
            ir_LH = ir_LH + ir_LH_conv(ir_LH)
            ir_HL = ir_HL + ir_HL_conv(ir_HL)
            ir_HH = ir_HH + ir_HH_conv(ir_HH)

            # fuse rgb and ir in low/high
            LowLow = self.LL * rgb_LL + (1 - self.LL) * ir_LL
            LowHigh = self.LH * rgb_LH + (1 - self.LH) * ir_LH
            HighLow = self.HL * rgb_HL + (1 - self.HL) * ir_HL
            HighHigh = self.HH * rgb_HH + (1 - self.HH) * ir_HH

            # Inverse Discrete Wavelet Transformation
            Low = self.idwt2d(LowLow, LowHigh, channel=self.channels)
            High = self.idwt2d(HighLow, HighHigh, channel=self.channels)
            Low = L_fused_conv(Low)
            High = H_fused_conv(High)
            output = self.idwt2d(Low, High, channel=self.channels)
            output = fused_conv(output)
            # output = rgb_input + ir_input + output   # residual ?

            # return raw shape
            if is_h_pad:
                output = output[:, :, :-1, :]
            if is_w_pad:
                output = output[:, :, :, :-1]
            outputs.append(output)

        return tuple(outputs)       


class SingleScaleFusion_old(BaseModel):
    def __init__(self, 
                 channels: int,
                 reduction: int) -> None:
        super().__init__()
        self.channels = channels
        mid_channels = channels // reduction

        # rgb
        self.rgb_LL_convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
        self.rgb_LH_convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
        self.rgb_HL_convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
        self.rgb_HH_convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
        self.init_weights(self.rgb_LL_convs)
        self.init_weights(self.rgb_LH_convs)
        self.init_weights(self.rgb_HL_convs)
        self.init_weights(self.rgb_HH_convs)

        # ir
        self.ir_LL_convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
        self.ir_LH_convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
        self.ir_HL_convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
        self.ir_HH_convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
        self.init_weights(self.ir_LL_convs)
        self.init_weights(self.ir_LH_convs)
        self.init_weights(self.ir_HL_convs)
        self.init_weights(self.ir_HH_convs)
        
        # fused
        self.LL = nn.Parameter(0.5, requires_grad=True)
        self.LH = nn.Parameter(0.5, requires_grad=True)
        self.HL = nn.Parameter(0.5, requires_grad=True)
        self.HH = nn.Parameter(0.5, requires_grad=True)
        self.L_fused_convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
        self.H_fused_convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
        self.fused_convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1))
        self.init_weights(self.H_fused_convs)
        self.init_weights(self.L_fused_convs)
        self.init_weights(self.fused_convs)

        # inital dwt/idwt weight
        # Haar 小波滤波器
        self.dwt_low_pass = torch.tensor([0.5, 0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1)  # 低通滤波器
        self.dwt_high_pass = torch.tensor([0.5, -0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1)  # 高通滤波器
        self.idwt_low_pass = torch.tensor([0.5, 0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1)  # 低通滤波器
        self.idwt_high_pass = torch.tensor([0.5, -0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1)  # 高通滤波器

        # 转为可学习参数
        self.register_buffer('dwt_low_pass', self.dwt_low_pass)
        self.register_buffer('dwt_high_pass', self.dwt_high_pass)
        self.register_buffer('idwt_low_pass', self.idwt_low_pass)
        self.register_buffer('idwt_high_pass', self.idwt_high_pass)

    def init_weights(self, module) -> None:
        """Initialize weights for components."""
        for submodule in module.modules():
            if isinstance(submodule, nn.Conv2d):
                # 对卷积层使用 Kaiming 初始化
                nn.init.kaiming_normal_(submodule.weight, mode='fan_out', nonlinearity='relu')
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.Linear):
                # 对线性层使用 Xavier 初始化
                nn.init.xavier_normal_(submodule.weight)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.BatchNorm2d):
                # 对 BatchNorm 层的 weight 设置为1，bias 设置为0
                nn.init.constant_(submodule.weight, 1)
                nn.init.constant_(submodule.bias, 0)

    # 离散小波变换 (DWT)
    def dwt2d(self, input: torch.Tensor, channel: int):
        """
        执行2D离散小波变换 (DWT)，提取低频和高频分量
        Args:
            input (Tensor): 输入张量，形状为 (B, C, H, W)
        Returns:
            四个子带 (LowLow, LowHigh, HighLow, HighHigh)
        """
        
        input = F.pad(input, (1, 1, 0, 0), mode='reflect')  # 填充列，复杂填充无法直接用conv中的padding

        # 进行卷积操作，得到低频和高频分量
        low = F.conv2d(input, self.dwt_low_pass, stride=(1, 2), groups=channel)
        high = F.conv2d(input, self.dwt_high_pass, stride=(1, 2), groups=channel)        
        
        low = F.pad(low, (0, 0, 1, 1), mode='reflect')  # 填充行
        high = F.pad(high, (0, 0, 1, 1), mode='reflect')

        LowLow = F.conv2d(low, self.dwt_low_pass.transpose(2, 3), groups=channel, stride=(2, 1))
        LowHigh = F.conv2d(low, self.dwt_high_pass.transpose(2, 3), groups=channel, stride=(2, 1))
        HighLow = F.conv2d(high, self.dwt_low_pass.transpose(2, 3), groups=channel, stride=(2, 1))
        HighHigh = F.conv2d(high, self.dwt_high_pass.transpose(2, 3), groups=channel, stride=(2, 1))

        return LowLow, LowHigh, HighLow, HighHigh

    # 逆离散小波变换 (IDWT)
    def idwt2d(self, low: torch.Tensor, high: torch.Tensor, channel: int):
        """
        执行2D逆离散小波变换 (IDWT)，将低频和高频分量恢复为原始图像
        Args:
            param low: 低频分量
            param high: 高频分量
        Returns:
            output (Tensor): 恢复后的图像
        """

        # 使用逆卷积 (转置卷积) 恢复图像
        low = F.conv_transpose2d(low, self.idwt_low_pass.transpose(2, 3), stride=(2, 1), groups=channel)
        high = F.conv_transpose2d(high, self.idwt_high_pass.transpose(2, 3), stride=(2, 1), groups=channel)

        # 恢复后的图像是低频和高频分量的和
        output = low + high
        return output

    def forward(self, rgb_input, ir_input):
        # 输入特征图的高度和宽度必须为偶数
        _, _, Height, Width = rgb_input.shape
        if Height % 2 != 0:
            phd = (0, 0, 0, 1)
            is_h_pad = True
            rgb_input = F.pad(rgb_input, phd, mod="reflect")
            ir_input = F.pad(ir_input, phd, mod="reflect")                
        else:
            is_h_pad = False
        if Width % 2 != 0:
            pwd = (0, 1)
            is_w_pad = True
            rgb_input = F.pad(rgb_input, pwd, mod="reflect")
            ir_input = F.pad(ir_input, pwd, mod="reflect")     
        else:
            is_w_pad = False

        # Discrete Wavelet Transformation
        rgb_LL, rgb_LH, rgb_HL, rgb_HH = self.dwt2d(rgb_input, channel=self.channels)
        ir_LL, ir_LH, ir_HL, ir_HH = self.dwt2d(ir_input, channel=self.channels)

        # residual refine freq
        rgb_LL = rgb_LL + self.rgb_LL_convs(rgb_LL)
        rgb_LH = rgb_LH + self.rgb_LH_convs(rgb_LH)
        rgb_HL = rgb_HL + self.rgb_HL_convs(rgb_HL)
        rgb_HH = rgb_HH + self.rgb_HH_convs(rgb_HH)
        ir_LL = ir_LL + self.ir_LL_convs(ir_LL)
        ir_LH = ir_LH + self.ir_LH_convs(ir_LH)
        ir_HL = ir_HL + self.ir_HL_convs(ir_HL)
        ir_HH = ir_HH + self.ir_HH_convs(ir_HH)

        # fuse rgb and ir in low/high
        LowLow = self.LL * rgb_LL + (1 - self.LL) * ir_LL
        LowHigh = self.LH * rgb_LH + (1 - self.LH) * ir_LH
        HighLow = self.HL * rgb_HL + (1 - self.HL) * ir_HL
        HighHigh = self.HH * rgb_HH + (1 - self.HH) * ir_HH

        # Inverse Discrete Wavelet Transformation
        Low = self.idwt2d(LowLow, LowHigh, channel=self.channels)
        High = self.idwt2d(HighLow, HighHigh, channel=self.channels)
        Low = self.L_fused_convs(Low)
        High = self.H_fused_convs(High)
        output = self.idwt2d(Low, High, channel=self.channels)
        output = self.fused_convs(output)
        # output = rgb_input + ir_input + output   # residual ?

        # return raw shape
        if is_h_pad:
            output = output[:, :, :-1, :]
        if is_w_pad:
            output = output[:, :, :, :-1]
        return output


class SingleScaleFusion(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super(SingleScaleFusion, self).__init__()
        self.channels = channels
        mid_channels = channels // reduction

        # Helper to create convolutional blocks
        def create_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1))
        
        # RGB and IR convolutional blocks
        self.conv_blocks = nn.ModuleDict({
            f"{modality}_{freq}": create_conv_block(channels, channels)
            for modality in ["rgb", "ir"]
            for freq in ["LL", "LH", "HL", "HH"]})
        
        # Fusion parameters
        self.fusion_weights = nn.ParameterDict({
            freq: nn.Parameter(torch.tensor(0.5), requires_grad=True)
            for freq in ["LL", "LH", "HL", "HH"]})

        # Fused convolutional blocks
        self.L_fused_convs = create_conv_block(channels, channels)
        self.H_fused_convs = create_conv_block(channels, channels)
        self.fused_convs = create_conv_block(channels, channels)

        # Initialize weights
        for module in self.conv_blocks.values():
            self.init_weights(module)
        self.init_weights(self.L_fused_convs)
        self.init_weights(self.H_fused_convs)
        self.init_weights(self.fused_convs)

        # Haar Wavelet Filters
        self.register_buffer('dwt_low_pass', torch.tensor([0.5, 0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1))
        self.register_buffer('dwt_high_pass', torch.tensor([0.5, -0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1))
        self.register_buffer('idwt_low_pass', torch.tensor([0.5, 0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1))
        self.register_buffer('idwt_high_pass', torch.tensor([0.5, -0.5]).view(1, 1, 1, 2).repeat(channels, 1, 1, 1))

    def init_weights(self, module) -> None:
        """Initialize weights for components."""
        for submodule in module.modules():
            if isinstance(submodule, nn.Conv2d):
                nn.init.kaiming_normal_(submodule.weight, mode='fan_out', nonlinearity='relu')
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)

    def dwt2d(self, input: torch.Tensor, channel: int):
        """Discrete Wavelet Transform (DWT)."""
        # input = F.pad(input, (1, 1, 0, 0), mode='reflect')
        low = F.conv2d(input, self.dwt_low_pass, stride=(1, 2), groups=channel)
        high = F.conv2d(input, self.dwt_high_pass, stride=(1, 2), groups=channel)
        # low = F.pad(low, (0, 0, 1, 1), mode='reflect')
        # high = F.pad(high, (0, 0, 1, 1), mode='reflect')
        LowLow = F.conv2d(low, self.dwt_low_pass.transpose(2, 3), groups=channel, stride=(2, 1))
        LowHigh = F.conv2d(low, self.dwt_high_pass.transpose(2, 3), groups=channel, stride=(2, 1))
        HighLow = F.conv2d(high, self.dwt_low_pass.transpose(2, 3), groups=channel, stride=(2, 1))
        HighHigh = F.conv2d(high, self.dwt_high_pass.transpose(2, 3), groups=channel, stride=(2, 1))
        return LowLow, LowHigh, HighLow, HighHigh

    def idwt2d(self, low: torch.Tensor, high: torch.Tensor, channel: int, is_transpose=True):
        """Inverse Discrete Wavelet Transform (IDWT)."""
        if is_transpose:
            low = F.conv_transpose2d(low, self.idwt_low_pass.transpose(2, 3), stride=(2, 1), groups=channel)
            high = F.conv_transpose2d(high, self.idwt_high_pass.transpose(2, 3), stride=(2, 1), groups=channel)
        else:
            low = F.conv_transpose2d(low, self.idwt_low_pass, stride=(1, 2), groups=channel)
            high = F.conv_transpose2d(high, self.idwt_high_pass, stride=(1, 2), groups=channel)
        return low + high

    def forward(self, rgb_input, ir_input):
        _, _, Height, Width = rgb_input.shape
        is_h_pad, is_w_pad = False, False

        # Ensure even dimensions
        if Height % 2 != 0:
            rgb_input = F.pad(rgb_input, (0, 0, 0, 1), mode="reflect")
            ir_input = F.pad(ir_input, (0, 0, 0, 1), mode="reflect")
            is_h_pad = True
        if Width % 2 != 0:
            rgb_input = F.pad(rgb_input, (0, 1, 0, 0), mode="reflect")
            ir_input = F.pad(ir_input, (0, 1, 0, 0), mode="reflect")
            is_w_pad = True

        # DWT
        rgb_freq = self.dwt2d(rgb_input, channel=self.channels)
        ir_freq = self.dwt2d(ir_input, channel=self.channels)

        # Residual refinement and fusion
        fused_freq = []
        for i, freq in enumerate(["LL", "LH", "HL", "HH"]):
            rgb = rgb_freq[i] + self.conv_blocks[f"rgb_{freq}"](rgb_freq[i])
            ir = ir_freq[i] + self.conv_blocks[f"ir_{freq}"](ir_freq[i])
            fused = self.fusion_weights[freq] * rgb + (1 - self.fusion_weights[freq]) * ir
            fused_freq.append(fused)

        # IDWT
        Low = self.idwt2d(fused_freq[0], fused_freq[1], channel=self.channels, is_transpose=True)
        High = self.idwt2d(fused_freq[2], fused_freq[3], channel=self.channels, is_transpose=True)
        Low = self.L_fused_convs(Low)
        High = self.H_fused_convs(High)
        output = self.idwt2d(Low, High, channel=self.channels, is_transpose=False)
        output = self.fused_convs(output)

        # Restore original dimensions
        if is_h_pad:
            output = output[:, :, :-1, :]
        if is_w_pad:
            output = output[:, :, :, :-1]
        return output.contiguous()


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

    #     # 融合
    #     # self.fusion = Fusion(channels=256, reduction=4, num_layer=4)
    #     self.fusions = nn.ModuleList(SingleScaleFusion(channels=256, reduction=4) 
    #                                 for _ in range(4))
    #     # self.fusions = Dual_Fusion(with_cp=True, batch_size=batch_size)
    #     # self.fusions = Dual_Fusion_New()#with_cp=True)        
        
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

            # if i == 0: 
            #     out = list(x)
            # else:
            #     for level in range(len(out)):
            #         out[level] += x[level]
        
        ## 简单的sum融合
        # out = [torch.sum(torch.stack([x[i] for x in xs], dim=0), dim=0)
        #         for i in range(len(xs[0]))]
        # DWT融合
        out = [fusion(rgb_x, ir_x) for rgb_x, ir_x, fusion in zip(xs[0], xs[1], self.fusions)]
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