import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class Extract_Edge(nn.Module):
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
        # img = self.bn(img)
        edge_detect = self.conv(img)
        edge_detect = self.relu(edge_detect)
        return edge_detect
    
class Extract_Edge_LoG(nn.Module):
    """
    高斯拉普拉斯算子（LoG）结合了高斯模糊和拉普拉斯算子，用于边缘检测。它可以通过去除噪声并增强边缘，得到更清晰的边缘图像。其操作步骤为：

    对图像进行高斯平滑处理。
    对平滑后的图像进行拉普拉斯操作，检测图像的二阶导数。
    """
    def __init__(self):
        super(Extract_Edge_LoG, self).__init__()
        # 生成自定义kernal，高通滤波器
        kernel = torch.tensor([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256.0
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.conv = nn.Conv2d(3, 3, kernel_size=5, padding=1, bias=False, groups=3)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False
        
        self.bn = nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        img = self.bn(img)
        edge_detect = self.conv(img)
        edge_detect = self.relu(edge_detect)
        return edge_detect


# # 测试
# if __name__ == '__main__':
#     # 假设输入图像是一个单通道的灰度图像
#     input_image = torch.randn(1, 1, 256, 256)  # 随机生成一个 256x256 的灰度图
    
#     model = CannyEdgeDetector(low_threshold=0.1, high_threshold=0.3)
#     edges = model(input_image)
    
#     print(f"Detected edges: {edges.shape}")

    
if __name__ == '__main__':
    img_path = '/workspace/mmdetection/demo/00000.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img, dtype=torch.float32)
    img = img.permute(2, 0, 1).unsqueeze(0)

    # mean = torch.mean(img.view(1, 3, -1), dim=-1)
    # var = torch.std(img.view(1, 3, -1), dim=-1)
    mean = torch.mean(img, dim=(-2, -1), keepdim=True)
    var = torch.std(img, dim=(-2, -1), keepdim=True)

    img = (img - mean) / (var + 1e-6)
    # filter = Extract_Edge()
    filter = Extract_Edge_LoG()

    img = filter(img)
    img.detach()

    min, _ = torch.min(img.view(1, 3, -1), dim=-1, keepdim=True)
    max, _ = torch.max(img.view(1, 3, -1), dim=-1, keepdim=True)

    img = (img - min.unsqueeze(-1)) / (max.unsqueeze(-1) - min.unsqueeze(-1)) * 255.
    # max = torch.max(img)
    img = img.squeeze(0).permute(1, 2, 0)
    img = img.type(torch.uint8).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('/workspace/mmdetection/demo/000001.jpg', img)
    # cv2.waitKey()
    # pass