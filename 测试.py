import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model_pytorch import get_resnet18_model
from model_efficientnet import get_efficientnet_model

# 数据预处理（与验证集一致）
def get_test_loader(base_dir, batch_size=32, img_size=150):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(os.path.join(base_dir, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# 模型测试函数
def test_model(model_path, model_constructor, base_dir, device='cuda'):
    model = model_constructor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    test_loader = get_test_loader(base_dir)
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    test_acc = correct / total
    test_loss = running_loss / len(test_loader)
    return test_acc, test_loss

# 主函数：测试双模型
if __name__ == "__main__":
    base_dir = "/home/stu021/cats_and_dogs_small"  # 数据集根路径
    resnet_model_path = "best_resnet18_model.pth"
    effnet_model_path = "best_efficientnet_b3_model.pth"
    
    resnet_acc, resnet_loss = test_model(
        resnet_model_path, get_resnet18_model, base_dir
    )
    effnet_acc, effnet_loss = test_model(
        effnet_model_path, get_efficientnet_model, base_dir
    )
    
    print(f"ResNet18 测试准确率：{resnet_acc:.4f}，测试损失：{resnet_loss:.4f}")
    print(f"EfficientNet-B3 测试准确率：{effnet_acc:.4f}，测试损失：{effnet_loss:.4f}")