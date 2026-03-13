import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import os
from data_loader import get_data_loaders  # 数据加载模块
from model_pytorch import get_resnet18_model  # ResNet18模型定义
from model_efficientnet import get_efficientnet_model  # EfficientNet-B3模型定义
from visualization import save_training_history  # 训练历史保存
import ssl
import argparse

# 1. 解决SSL证书问题（预训练模型下载）
ssl._create_default_https_context = ssl._create_unverified_context

# 2. 命令行参数配置
parser = argparse.ArgumentParser(description="猫狗识别双模型统一训练脚本")
parser.add_argument("--base_dir", type=str, default="D:\cats_and_dogs_small", 
                    help="数据集根路径")
parser.add_argument("--batch_size", type=int, default=32, 
                    help="训练批次大小")
parser.add_argument("--epochs", type=str, default="50,60", 
                    help="双模型训练轮次（ResNet18,EfficientNet-B3，逗号分隔）")
parser.add_argument("--save_dir", type=str, default="./", 
                    help="模型权重与训练历史保存路径")
args = parser.parse_args()

# 3. 解析参数
base_dir = args.base_dir
resnet_epochs, efficientnet_epochs = map(int, args.epochs.split(','))
batch_size = args.batch_size
save_dir = args.save_dir

# 4. 创建保存目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"已创建保存目录：{save_dir}")

# ---------------------- 5. ResNet18 训练函数 ----------------------
def train_resnet(base_dir, epochs, batch_size, save_dir):
    # 加载数据
    train_loader, val_loader, _ = get_data_loaders(base_dir, batch_size)
    print(f"\n=== ResNet18 训练启动 ===")
    print(f"训练集：{len(train_loader.dataset)}张 | 验证集：{len(val_loader.dataset)}张")
    print(f"批次大小：{batch_size} | 最大轮次：{epochs} | 保存路径：{save_dir}")
    
    # 加载模型
    model = get_resnet18_model(freeze_ratio=0.9)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"使用设备：{device}")
    
    # 初始化训练历史与早停参数
    history = {'train_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    early_stop_counter = 0
    model_save_path = os.path.join(save_dir, "best_resnet18_model.pth")
    
    # 训练循环
    for epoch in range(epochs):
        # 5.1 训练阶段
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度计算与参数更新
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # 【删除批次级打印】
        
        # 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 5.2 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        
        # 计算验证准确率
        val_acc = val_correct / val_total
        history['val_acc'].append(val_acc)
        # 【精简Epoch级输出】
        print(f"ResNet18 Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 5.3 早停机制与最优模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"ResNet18 保存最优模型（准确率：{best_val_acc:.4f}）至 {model_save_path}")
        else:
            early_stop_counter += 1
            print(f"ResNet18 验证准确率无提升（连续{early_stop_counter}轮）")
            if early_stop_counter >= 5:
                print(f"ResNet18 连续5轮无提升，提前停止训练")
                break
        
        # 5.4 学习率衰减
        scheduler.step()
    
    # 5.5 训练结束
    save_training_history(history, model_name="ResNet18", save_dir=save_dir)
    print(f"\n=== ResNet18 训练结束 ===")
    print(f"最佳验证准确率：{best_val_acc:.4f} | 训练历史已保存至 {save_dir}")
    return best_val_acc

# ---------------------- 6. EfficientNet-B3 训练函数 ----------------------
def train_efficientnet(base_dir, epochs, batch_size, save_dir):
    # 加载数据
    train_loader, val_loader, _ = get_data_loaders(base_dir, batch_size)
    print(f"\n=== EfficientNet-B3 训练启动 ===")
    print(f"训练集：{len(train_loader.dataset)}张 | 验证集：{len(val_loader.dataset)}张")
    print(f"批次大小：{batch_size} | 最大轮次：{epochs} | 保存路径：{save_dir}")
    
    # 加载模型
    model = get_efficientnet_model(freeze_ratio=0.9)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"使用设备：{device}")
    
    # 初始化训练历史与早停参数
    history = {'train_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    early_stop_counter = 0
    model_save_path = os.path.join(save_dir, "best_efficientnet_b3_model.pth")
    
    # 训练循环
    for epoch in range(epochs):
        # 6.1 训练阶段
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # 【删除批次级打印】
        
        # 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 6.2 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        
        val_acc = val_correct / val_total
        history['val_acc'].append(val_acc)
        # 【精简Epoch级输出】
        print(f"EfficientNet-B3 Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 6.3 早停机制与最优模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"EfficientNet-B3 保存最优模型（准确率：{best_val_acc:.4f}）至 {model_save_path}")
        else:
            early_stop_counter += 1
            print(f"EfficientNet-B3 验证准确率无提升（连续{early_stop_counter}轮）")
            if early_stop_counter >= 6:
                print(f"EfficientNet-B3 连续6轮无提升，提前停止训练")
                break
        
        # 6.4 学习率更新
        scheduler.step()
    
    # 6.5 训练结束
    save_training_history(history, model_name="EfficientNet-B3", save_dir=save_dir)
    print(f"\n=== EfficientNet-B3 训练结束 ===")
    print(f"最佳验证准确率：{best_val_acc:.4f} | 训练历史已保存至 {save_dir}")
    return best_val_acc

# ---------------------- 7. 主函数：启动双模型训练 ----------------------
if __name__ == "__main__":
    resnet_best_acc = train_resnet(base_dir, resnet_epochs, batch_size, save_dir)
    effnet_best_acc = train_efficientnet(base_dir, efficientnet_epochs, batch_size, save_dir)
    
    # 输出双模型对比结果
    print(f"\n=====================================")
    print(f"双模型训练结果对比")
    print(f"=====================================")
    print(f"ResNet18（基础模型）最佳验证准确率：{resnet_best_acc:.4f}")
    print(f"EfficientNet-B3（优化模型）最佳验证准确率：{effnet_best_acc:.4f}")
    print(f"精度提升幅度：{((effnet_best_acc - resnet_best_acc) / resnet_best_acc * 100):.2f}%")
    print(f"=====================================")