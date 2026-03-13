```
# 猫狗识别图像分类项目
基于 PyTorch 实现的猫狗图像二分类系统，包含模型训练、测试、可视化与部署。

## 项目介绍
本项目使用经典卷积神经网络 ResNet18 与轻量高效网络 EfficientNet-B3 完成猫狗图像分类任务，实现了数据集划分、模型训练、指标可视化、结果分析与 Gradio 演示功能。

## 项目结构
cat-dog-classifier/├── main/ # 主程序代码├── models/ # 训练好的最优模型权重├── results/ # 训练曲线、图表、历史数据├── README.md # 项目说明└── 配置文件
plaintext

## 功能清单
- 数据集自动划分（训练集 / 验证集 / 测试集）
- 双模型训练与对比（ResNet18 / EfficientNet-B3）
- 训练损失、准确率曲线可视化
- 混淆矩阵、数据分布分析
- 单图预测与批量测试
- Gradio 网页可视化 Demo

## 技术栈
- PyTorch
- Pandas / NumPy
- Matplotlib / Seaborn
- Gradio

## 运行方式
1. 安装依赖
```bash
pip install torch torchvision pandas numpy matplotlib seaborn gradio
训练模型
bash
运行
python main/train.py
启动可视化 Demo
bash
运行
python main/gradio_demo.py
项目成果
ResNet18 测试准确率：92%+
EfficientNet-B3 测试准确率：95%+
完整训练日志、可视化图表、模型权重
作者
GitHub: https://github.com/hrx1129
```

