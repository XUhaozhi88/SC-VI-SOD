import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmengine.model import BaseModule
from mmdet.registry import MODELS


#-- Gmask --------------------------------------------------------------------

class TinyGMask(nn.Module):
    def __init__(self, img_shape, patch_num=20):
        super(TinyGMask, self).__init__()
        W, H = img_shape
        self.patch_num = patch_num
        self.conv1 = nn.Conv2d(3, 16, 7, 4, 3)     #2,3,512,640
        self.conv2 = nn.Conv2d(16, 32, 7, 4, 3)    #2,3,100,124
        self.conv3 = nn.Conv2d(32, 64, 7, 4, 3)    #2,3,20,24
        self.flatten = nn.Flatten()

        self.trans2list = nn.Sequential(            #b,400
            nn.Linear(int(64 * H / 64 * W / 64), 1000),
            # nn.Linear(5000, 1000),
            nn.Linear(in_features=1000, out_features=np.power(self.patch_num, 2), bias=True),
            nn.Sigmoid())

    def forward(self, x):
        # Unet++
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x_final = self.trans2list(self.flatten(x))
        return x_final

class GMaskBinaryList(nn.Module):
    def __init__(self, img_shape):
        super(GMaskBinaryList, self).__init__()
        self.g_mask_binary_list = TinyGMask(img_shape)

    def forward(self, x):
        mask_list = self.g_mask_binary_list(x)
        return mask_list

class MaskFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mask_list_):
        mask_list_topk, _ = mask_list_.topk(320)
        mask_list_min = torch.min(mask_list_topk, dim=1).values
        mask_list_min_ = mask_list_min.unsqueeze(-1)
        ge = torch.ge(mask_list_, mask_list_min_)
        zero = torch.zeros_like(mask_list_)
        one = torch.ones_like(mask_list_)
        mask_list = torch.where(ge, one, zero)
        return mask_list

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

@MODELS.register_module()
class UniqueMaskGenerator(BaseModule):
    """
    Args:
        patch_num (int): raw or column patch number
        keep_low (bool):
    """
    def __init__(self,
                 img_shape,
                 keep_low,
                 patch_num=20):
        super(UniqueMaskGenerator, self).__init__()
        self.patch_num = patch_num
        self.keep_low  = keep_low
        self.img_shape = img_shape

        self.Gmaskbinarylist_vis = GMaskBinaryList(img_shape)
        self.Gmaskbinarylist_lwir = GMaskBinaryList(img_shape)
        self.MaskFun_vis = MaskFunction()
        self.MaskFun_lwir = MaskFunction()

    def forward(self, img_vis, img_lwir):
        """Forward function."""

        vis_fre = torch.fft.fft2(img_vis)
        fre_m_vis = torch.abs(vis_fre)  # 幅度谱，求模得到
        fre_m_vis = torch.fft.fftshift(fre_m_vis)
        # fre_p_vis = torch.angle(vis_fre)  # 相位谱，求相角得到

        lwir_fre = torch.fft.fft2(img_lwir)
        fre_m_lwir = torch.abs(lwir_fre)  # 幅度谱，求模得到
        fre_m_lwir = torch.fft.fftshift(fre_m_lwir)
        # fre_p_lwir = torch.angle(lwir_fre)  # 相位谱，求相角得到
        mask_vis_list_ = self.Gmaskbinarylist_vis(fre_m_vis)
        mask_lwir_list_ = self.Gmaskbinarylist_lwir(fre_m_lwir)
        mask_vis_list = self.MaskFun_vis.apply(mask_vis_list_).reshape((-1, 1, self.patch_num , self.patch_num))
        mask_vis_list[:, :, int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1,int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1] = 1
        mask_lwir_list = self.MaskFun_lwir.apply(mask_lwir_list_).reshape((-1, 1,self.patch_num , self.patch_num))
        mask_lwir_list[:, :, int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1,int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1] = 1

        mask_vis = F.interpolate(
            mask_vis_list, 
            scale_factor=[self.img_shape[0] / self.patch_num, self.img_shape[1] / self.patch_num], 
            mode='nearest')
        mask_lwir = F.interpolate(
            mask_lwir_list, 
            scale_factor=[self.img_shape[0] / self.patch_num, self.img_shape[1] / self.patch_num], 
            mode='nearest')

        return mask_vis, mask_lwir



#-- Gcommon --------------------------------------------------------------------

class Extract_Edge(BaseModule):
    def __init__(self):
        super(Extract_Edge, self).__init__()
        # 生成自定义kernal，高通滤波器
        kernel = torch.Tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]) / 9
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False
        
        self.bn = nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        img = self.bn(img)
        edge_detect = self.conv(img)
        edge_detect = self.relu(edge_detect)
        return edge_detect

@MODELS.register_module()
class CommonFeatureGenerator(BaseModule):
    """Common Feature Mask Generator
        This is the implementation of paper '  '

    Args:
        img_vis (Tensor): The RGB input
        img_lwir (Tensor): The infrared input
    """

    def __init__(self, strides, backbone, neck) -> None:
        super(CommonFeatureGenerator,self).__init__()
        self.EE = Extract_Edge()
        self.strides = strides

        self.backbone = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)
        self.neck  = MODELS.build(neck)
        self.neck_lwir = MODELS.build(neck)

    def edge_fusion(self,img_vis_edge, img_lwir_edge):
        #simple fusion
        fused_edge = img_vis_edge + img_lwir_edge
        fused_edge = fused_edge / fused_edge.std()
        return fused_edge
    
    def extract_feat(self, batch_inputs):
        x = self.backbone(batch_inputs)
        x = self.neck(x)
        return x
    
    def extract_feat_lwir(self, batch_inputs):
        x = self.backbone_lwir(batch_inputs)
        x = self.neck_lwir(x)
        return x
    
    def forward(self, img_vis, img_lwir):
        """Forward function."""

        common_features = []
        #获取边缘特征
        img_vis_edge = self.EE(img_vis) # 拉普拉斯高通滤波器，锐化边缘信息（不清楚为啥不用canny算子）
        img_lwir_edge = self.EE(img_lwir)
        img_fused_edge = self.edge_fusion(img_vis_edge, img_lwir_edge)
        img_fused_edge = 0.05 * img_fused_edge

        # 提取特征（是不是可以用主干中已经提取的特征）
        x_vis = self.extract_feat(img_vis)
        x_lwir = self.extract_feat_lwir(img_lwir)
        
        #TODO 加入边缘信息
        assert len(x_vis) == len(self.strides)
        common_features = []
        for i, stride in enumerate(self.strides):
            x_common = 0.5 * (x_vis[i] + x_lwir[i])
            # img_fused_edge_down = F.interpolate(img_fused_edge, scale_factor=1/stride, mode='bicubic')
            img_fused_edge_down = F.interpolate(img_fused_edge, size=x_common.shape[-2:], mode='bicubic')
            x_common = torch.cat([x_common, img_fused_edge_down], dim=1)
            common_features.append(x_common)

        return tuple(common_features)
    

@MODELS.register_module()
class CommonFeatureGenerator_New(BaseModule):
    """Common Feature Mask Generator
        This is the implementation of paper '  '

    Args:
        img_vis (Tensor): The RGB input
        img_lwir (Tensor): The infrared input
    """

    def __init__(self, strides) -> None:
        super(CommonFeatureGenerator_New,self).__init__()
        self.EE = Extract_Edge()
        self.strides = strides

    def edge_fusion(self,img_vis_edge, img_lwir_edge):
        #simple fusion
        fused_edge = img_vis_edge + img_lwir_edge
        fused_edge = fused_edge / fused_edge.std()
        return fused_edge
    
    def forward(self, x_vis, x_lwir, img_vis, img_lwir):
        """Forward function."""

        common_features = []
        #获取边缘特征
        img_vis_edge = self.EE(img_vis) # 拉普拉斯高通滤波器，锐化边缘信息（不清楚为啥不用canny算子）
        img_lwir_edge = self.EE(img_lwir)
        img_fused_edge = self.edge_fusion(img_vis_edge, img_lwir_edge)
        img_fused_edge = 0.05 * img_fused_edge
        
        #TODO 加入边缘信息
        assert len(x_vis) == len(self.strides)
        common_features = []
        for i, stride in enumerate(self.strides):
            x_common = 0.5 * (x_vis[i] + x_lwir[i])
            # img_fused_edge_down = F.interpolate(img_fused_edge, scale_factor=1/stride, mode='bicubic')
            img_fused_edge_down = F.interpolate(img_fused_edge, size=x_common.shape[-2:], mode='bicubic')
            x_common = torch.cat([x_common, img_fused_edge_down], dim=1)
            common_features.append(x_common)

        return tuple(common_features)
    


#-- FeaFusion --------------------------------------------------------------------
import math
class _Gate(nn.Module):
    def __init__(self, num_gate, img_shape):
        super(_Gate, self).__init__()
        H = math.ceil(img_shape[0] / 32)
        W = math.ceil(img_shape[1] / 32)
        # for vis
        self.flatten1 = nn.Flatten()
        self.pool1 = nn.AvgPool2d(kernel_size=32)
        self.IA_fc11 = nn.Linear(in_features=H * W * 6, out_features=1000)  # rgbt_tiny
        self.IA_fc12 = nn.Linear(in_features=1000, out_features=100)
        self.IA_fc13 = nn.Linear(in_features=100, out_features=num_gate) # 5
        # for lwir
        self.flatten2 = nn.Flatten()
        self.pool2 = nn.AvgPool2d(kernel_size=32)
        self.IA_fc21 = nn.Linear(in_features=H * W * 6, out_features=1000)  # rgbt_tiny
        self.IA_fc22 = nn.Linear(in_features=1000, out_features=100)
        self.IA_fc23 = nn.Linear(in_features=100, out_features=num_gate)

    def forward(self, img_vis, img_lwir):
        weights = []        
        # vis的weight
        x11= self.pool1(torch.cat([img_vis, img_lwir], dim=1))
        x12 = self.flatten1(x11)
        x13 = self.IA_fc11(x12)
        x14 = self.IA_fc12(x13)
        weight = self.IA_fc13(x14)
        weights.append(weight)
        # lwir的weight
        x21= self.pool2(torch.cat([img_vis, img_lwir], dim=1))
        x22 = self.flatten2(x21)
        x23 = self.IA_fc21(x22)
        x24 = self.IA_fc22(x23)
        weight = self.IA_fc23(x24)
        weights.append(weight)
        return weights

@MODELS.register_module()
class Conv_Fusion(BaseModule):
    """Common Feature Mask Generator 
        This is the implementation of paper '  '

    注：其实看作一个门控（或者是feature-wise的dropout)

    Args:
        img_vis (Tensor): The RGB input
        img_lwir (Tensor): The infrared input
    """

    def __init__(self,
                 num_gate,
                 img_shape,
                 neck,
                 loss_MI) -> None:
        super(Conv_Fusion, self).__init__()
        self.Gate = _Gate(num_gate, img_shape)
        self.expert_vis = MODELS.build(neck)
        self.expert_lwir = MODELS.build(neck)
        self.MILoss1 = MODELS.build(loss_MI)
        self.MILoss2 = MODELS.build(loss_MI)

    def forward(self, x_vis, x_lwir, x_common, img_vis, img_lwir):
        """Forward function."""
        gate = self.Gate(img_vis, img_lwir) # 2*[B, 5, H, W]
        gate_sms=[]
        gate_sm1=F.softmax(gate[0], dim=0)
        gate_sm1= torch.where(torch.abs(gate_sm1) > 0.01, gate_sm1, torch.tensor(0.).cuda())
        gate_sms.append(gate_sm1)
        gate_sm2=F.softmax(gate[1], dim=0)
        gate_sm2= torch.where(torch.abs(gate_sm2) > 0.01, gate_sm2, torch.tensor(0.).cuda())
        gate_sms.append(gate_sm2)
        x_vis_exclusive = self.expert_vis(x_vis)
        x_lwir_exclusive = self.expert_lwir(x_lwir)

        # 损失函数 KL散度和交叉熵
        miloss1_vis = self.MILoss1(x_vis_exclusive[1], x_lwir_exclusive[1])/(x_lwir_exclusive[1].shape[2]*x_lwir_exclusive[1].shape[3])
        miloss2_vis = self.MILoss2(x_vis_exclusive[2], x_lwir_exclusive[2])/(x_lwir_exclusive[2].shape[2]*x_lwir_exclusive[2].shape[3])
        MIloss_vis = miloss1_vis + miloss2_vis
        miloss1_lwir = self.MILoss1(x_vis_exclusive[1], x_lwir_exclusive[1])/(x_lwir_exclusive[1].shape[2]*x_lwir_exclusive[1].shape[3])
        miloss2_lwir = self.MILoss2(x_vis_exclusive[2], x_lwir_exclusive[2])/(x_lwir_exclusive[2].shape[2]*x_lwir_exclusive[2].shape[3])
        MIloss_lwir = miloss1_lwir + miloss2_lwir
        
        # 融合
        assert len(x_common) == len(x_vis_exclusive)
        outs = []
        for i in range(len(x_common)):
            gate_sms_vis = gate_sms[0][:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            gate_sms_lwir = gate_sms[1][:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            unique_feature_fusion = \
                gate_sms_vis * x_vis_exclusive[i] + gate_sms_lwir * x_lwir_exclusive[i]
            outs.append(0.7 * x_common[i] + 0.3 * unique_feature_fusion) # 这个权重也可以设置可学习

        # unique_feature_fusion = []
        # for i in range(len(x_vis_exclusive)):
        #     unique_feature_fusion.append(
        #         gate_sms[0][:,i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_vis_exclusive[i] +
        #         gate_sms[1][:,i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_lwir_exclusive[i])
        # unique_feature_fusion = tuple(unique_feature_fusion)
        # outs = []
        # for i in range(len(x_common)):
        #     outs.append(0.7 * x_common[i] + 0.3 * unique_feature_fusion[i]) # 这个权重也可以设置可学习

        return tuple(outs), MIloss_vis, MIloss_lwir
    

@MODELS.register_module()
class Conv_Fusion_New(BaseModule):
    """Common Feature Mask Generator 
        This is the implementation of paper '  '

    注：其实看作一个门控（或者是feature-wise的dropout)

    Args:
        img_vis (Tensor): The RGB input
        img_lwir (Tensor): The infrared input
    """

    def __init__(self,
                 num_gate,
                 img_shape,
                 loss_MI,
                 neck) -> None:
        super(Conv_Fusion_New, self).__init__()
        self.Gate = _Gate(num_gate, img_shape)
        self.MILoss1 = MODELS.build(loss_MI)
        self.MILoss2 = MODELS.build(loss_MI)
        
        self.channel_exchange_vis = MODELS.build(neck)
        self.channel_exchange_lwir = MODELS.build(neck)
        # self.channel_exchange = MODELS.build(neck)
        # self.channel_exchange_vis = MODELS.build(neck)
        # self.channel_exchange_lwir = MODELS.build(neck)
        
        self.weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, x_vis, x_lwir, x_common, img_vis, img_lwir):
        """Forward function."""
        gate = self.Gate(img_vis, img_lwir) # 2*[B, 5, H, W]
        gate_sms=[]
        gate_sm1=F.softmax(gate[0], dim=0)
        gate_sm1= torch.where(torch.abs(gate_sm1) > 0.01, gate_sm1, torch.tensor(0.).cuda())
        gate_sms.append(gate_sm1)
        gate_sm2=F.softmax(gate[1], dim=0)
        gate_sm2= torch.where(torch.abs(gate_sm2) > 0.01, gate_sm2, torch.tensor(0.).cuda())
        gate_sms.append(gate_sm2)

        x_vis = self.channel_exchange_vis(x_vis)
        x_lwir = self.channel_exchange_lwir(x_lwir)

        # 损失函数 KL散度和交叉熵
        miloss1_vis = self.MILoss1(x_vis[1], x_lwir[1]) / (x_lwir[1].shape[2] * x_lwir[1].shape[3])
        miloss2_vis = self.MILoss2(x_vis[2], x_lwir[2]) / (x_lwir[2].shape[2] * x_lwir[2].shape[3])
        MIloss_vis = miloss1_vis + miloss2_vis
        miloss1_lwir = self.MILoss1(x_lwir[1], x_vis[1]) / (x_vis[1].shape[2] * x_vis[1].shape[3])
        miloss2_lwir = self.MILoss2(x_lwir[2], x_vis[2]) / (x_vis[2].shape[2] * x_vis[2].shape[3])
        MIloss_lwir = miloss1_lwir + miloss2_lwir
        
        # 融合
        assert len(x_common) == len(x_vis)
        outs = []
        for i in range(len(x_common)):
            gate_sms_vis = gate_sms[0][:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            gate_sms_lwir = gate_sms[1][:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            unique_feature_fusion = \
                gate_sms_vis * x_vis[i] + gate_sms_lwir * x_lwir[i]
            outs.append(self.weight * x_common[i] + 
                        (1 - self.weight) * unique_feature_fusion) # 这个权重也可以设置可学习

        return tuple(outs), MIloss_vis, MIloss_lwir