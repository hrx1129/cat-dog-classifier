import torch
import torch.nn as nn
from torchvision import models

# 解决SSL证书问题（预训练模型下载时用）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class CatDogResNet(nn.Module):
    def __init__(self, freeze_ratio=0.9):
        super(CatDogResNet, self).__init__()
        # 加载预训练ResNet18
        self.resnet = models.resnet18(pretrained=True)
        # 按比例冻结层
        total_params = list(self.resnet.parameters())
        freeze_num = int(len(total_params) * freeze_ratio)
        for param in total_params[:freeze_num]:
            param.requires_grad = False
        # 替换最后一层全连接层为二分类
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 2)
    
    def forward(self, x):
        return self.resnet(x)

def get_resnet18_model(freeze_ratio=0.9):
    """返回配置好的ResNet18模型，支持自定义层冻结比例"""
    return CatDogResNet(freeze_ratio=freeze_ratio)