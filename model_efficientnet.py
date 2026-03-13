import torch
import torch.nn as nn
from torchvision import models

# 解决SSL证书问题（预训练模型下载时用）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class CatDogEfficientNet(nn.Module):
    def __init__(self, freeze_ratio=0.7):  # 核心修改：默认冻结比例从0.9→0.7
        super(CatDogEfficientNet, self).__init__()
        # 加载预训练EfficientNet-B3（兼容新旧版本torchvision）
        try:
            # 新版torchvision（0.13+）使用weights参数
            self.model = models.efficientnet_b3(
                weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
            )
        except AttributeError:
            # 旧版torchvision使用pretrained参数
            self.model = models.efficientnet_b3(pretrained=True)
        
        # 冻结指定比例的层（增加边界检查，确保比例合法）
        total_params = list(self.model.parameters())
        freeze_ratio = max(0.0, min(1.0, freeze_ratio))  # 限制在0-1之间
        freeze_num = int(len(total_params) * freeze_ratio)
        
        # 冻结前freeze_num层，解冻剩余层（更多层参与训练）
        for i, param in enumerate(total_params):
            param.requires_grad = (i >= freeze_num)  # i >= 冻结数 → 允许梯度更新
        
        # 替换输出层为二分类（保持与任务匹配）
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)
        
        # 打印调试信息（确认冻结比例和参与训练的层数量）
        print(f"EfficientNet-B3 配置：")
        print(f"- 冻结比例：{freeze_ratio}（冻结{freeze_num}个参数组）")
        print(f"- 参与训练的参数组：{len(total_params) - freeze_num}个")
        print(f"- 输出层已替换为二分类（输入特征维度：{in_features}）")
    
    def forward(self, x):
        return self.model(x)

def get_efficientnet_model(freeze_ratio=0.7):  # 同步修改默认冻结比例为0.7
    """返回配置好的EfficientNet-B3模型（更多层参与微调）"""
    return CatDogEfficientNet(freeze_ratio=freeze_ratio)