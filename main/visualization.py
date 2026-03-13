import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams

# 全局设置：让曲线更清晰（适配报告排版）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 避免中文乱码（若报告用英文，可保留）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 100  # 图像分辨率（100-200为宜，兼顾清晰度与文件大小）
plt.rcParams['lines.linewidth'] = 2  # 曲线线条宽度
plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 14  # 子图标题字体大小
plt.rcParams['legend.fontsize'] = 10  # 图例字体大小

def save_training_history(history, model_name, save_dir='./'):
    """训练时自动调用：保存训练历史（损失+准确率）"""
    save_path = os.path.join(save_dir, f'{model_name}_training_history.npy')
    np.save(save_path, history)
    print(f"[{model_name}] 训练历史已保存至：{save_path}")

def plot_single_model_curve(model_name, history_path=None, save_dir='./'):
    """绘制单个模型的训练曲线（1行2列：损失+准确率）"""
    # 1. 加载训练历史
    if history_path is None:
        history_path = os.path.join(save_dir, f'{model_name}_training_history.npy')
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"未找到[{model_name}]的训练历史文件：{history_path}")
    history = np.load(history_path, allow_pickle='TRUE').item()  # 加载字典格式的历史数据
    
    # 2. 提取关键数据
    train_loss = history['train_loss']  # 每轮训练损失
    val_acc = history['val_acc']        # 每轮验证准确率
    epochs = range(1, len(train_loss) + 1)  # 训练轮次（x轴）
    
    # 3. 创建画布与子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # 1行2列，总宽度14、高度5
    
    # 3.1 子图1：训练损失曲线
    ax1.plot(epochs, train_loss, 'bo-', color='#1f77b4', label=f'{model_name} Training Loss')
    ax1.set_title(f'{model_name} - Training Loss Curve')
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Loss Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)  # 添加网格（便于读取数值）
    
    # 3.2 子图2：验证准确率曲线
    ax2.plot(epochs, val_acc, 'ro-', color='#ff7f0e', label=f'{model_name} Validation Accuracy')
    ax2.set_title(f'{model_name} - Validation Accuracy Curve')
    ax2.set_xlabel('Training Epochs')
    ax2.set_ylabel('Accuracy (0-1)')
    ax2.set_ylim(0.8, 1.0)  # 限定y轴范围（0.8-1.0），突出准确率变化细节
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 4. 调整布局+保存图像
    plt.tight_layout()  # 自动调整子图间距，避免标签重叠
    save_path = os.path.join(save_dir, f'{model_name}_train_curve.png')
    plt.savefig(save_path, bbox_inches='tight')  # 保存图像（bbox_inches避免标签被截断）
    plt.show()
    print(f"[{model_name}] 训练曲线已保存至：{save_path}")

def plot_double_model_comparison(save_dir='./'):
    """绘制双模型对比曲线（1行2列：损失对比+准确率对比），突出精度差异"""
    # 1. 加载两个模型的训练历史
    resnet_history_path = os.path.join(save_dir, 'ResNet18_training_history.npy')
    effnet_history_path = os.path.join(save_dir, 'EfficientNet-B3_training_history.npy')
    
    if not os.path.exists(resnet_history_path) or not os.path.exists(effnet_history_path):
        raise FileNotFoundError("缺少训练历史文件，请先运行train.py完成双模型训练")
    
    resnet_hist = np.load(resnet_history_path, allow_pickle='TRUE').item()
    effnet_hist = np.load(effnet_history_path, allow_pickle='TRUE').item()
    
    # 2. 提取数据（统一x轴轮次，取较短的轮次长度，避免数组长度不匹配）
    min_epochs = min(len(resnet_hist['train_loss']), len(effnet_hist['train_loss']))
    epochs = range(1, min_epochs + 1)
    
    # ResNet18数据
    resnet_loss = resnet_hist['train_loss'][:min_epochs]
    resnet_acc = resnet_hist['val_acc'][:min_epochs]
    
    # EfficientNet-B3数据
    effnet_loss = effnet_hist['train_loss'][:min_epochs]
    effnet_acc = effnet_hist['val_acc'][:min_epochs]
    
    # 3. 创建对比画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 3.1 子图1：训练损失对比
    ax1.plot(epochs, resnet_loss, 'bo-', color='#1f77b4', label='ResNet18 (Base Model)')
    ax1.plot(epochs, effnet_loss, 'go-', color='#2ca02c', label='EfficientNet-B3 (Optimized Model)')
    ax1.set_title('Training Loss Comparison (ResNet18 vs EfficientNet-B3)')
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Loss Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 3.2 子图2：验证准确率对比（重点突出）
    ax2.plot(epochs, resnet_acc, 'bo-', color='#1f77b4', label='ResNet18 (Base Model)')
    ax2.plot(epochs, effnet_acc, 'go-', color='#2ca02c', label='EfficientNet-B3 (Optimized Model)')
    
    # 标注最佳准确率（增强可读性）
    resnet_best_acc = max(resnet_acc)
    effnet_best_acc = max(effnet_acc)
    resnet_best_epoch = resnet_acc.index(resnet_best_acc) + 1
    effnet_best_epoch = effnet_acc.index(effnet_best_acc) + 1
    
    ax2.annotate(f'Best: {resnet_best_acc:.4f}\nEpoch {resnet_best_epoch}',
                 xy=(resnet_best_epoch, resnet_best_acc),
                 xytext=(resnet_best_epoch-5, resnet_best_acc-0.03),
                 arrowprops=dict(arrowstyle='->', color='#1f77b4'),
                 fontsize=9)
    
    ax2.annotate(f'Best: {effnet_best_acc:.4f}\nEpoch {effnet_best_epoch}',
                 xy=(effnet_best_epoch, effnet_best_acc),
                 xytext=(effnet_best_epoch-5, effnet_best_acc+0.01),
                 arrowprops=dict(arrowstyle='->', color='#2ca02c'),
                 fontsize=9)
    
    ax2.set_title('Validation Accuracy Comparison (ResNet18 vs EfficientNet-B3)')
    ax2.set_xlabel('Training Epochs')
    ax2.set_ylabel('Accuracy (0-1)')
    ax2.set_ylim(0.85, 1.0)  # 限定y轴范围，放大精度差异
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 4. 保存对比图
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'double_model_comparison_curve.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"双模型对比曲线已保存至：{save_path}")
    print(f"\n对比结论：")
    print(f"- ResNet18最佳验证准确率：{resnet_best_acc:.4f}（Epoch {resnet_best_epoch}）")
    print(f"- EfficientNet-B3最佳验证准确率：{effnet_best_acc:.4f}（Epoch {effnet_best_epoch}）")
    print(f"- 精度提升幅度：{((effnet_best_acc - resnet_best_acc) / resnet_best_acc * 100):.2f}%")

# 主函数：支持命令行调用（可单独运行该文件生成曲线）
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="猫狗识别模型训练曲线绘制脚本")
    parser.add_argument("--save_dir", type=str, default="./", 
                        help="训练历史文件与曲线保存路径（需与train.py的--save_dir一致）")
    parser.add_argument("--model", type=str, default="both", 
                        help="绘制单个模型（resnet/efficientnet）或双模型对比（both）")
    args = parser.parse_args()
    
    if args.model.lower() == "resnet":
        plot_single_model_curve(model_name="ResNet18", save_dir=args.save_dir)
    elif args.model.lower() == "efficientnet":
        plot_single_model_curve(model_name="EfficientNet-B3", save_dir=args.save_dir)
    elif args.model.lower() == "both":
        plot_double_model_comparison(save_dir=args.save_dir)
    else:
        print("参数错误：--model 仅支持 resnet/efficientnet/both")