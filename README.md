# 猫狗识别系统 (Cat-Dog Classifier)

基于 PyTorch 实现的猫狗图像二分类系统，对比了 ResNet18 与 EfficientNet-B3 两种模型的性能。

## ✨ 功能特性
- 支持数据集自动划分（训练集/验证集/测试集）
- 实现 ResNet18 与 EfficientNet-B3 双模型训练与对比
- 提供可视化训练曲线、数据分布直方图与混淆矩阵
- 集成 Gradio 可视化 Demo，可快速部署在线演示
- 支持单张图片预测与批量测试

## 🛠️ 技术栈
- **深度学习框架**：PyTorch
- **模型**：ResNet18 / EfficientNet-B3
- **数据处理**：Pandas, NumPy
- **可视化**：Matplotlib, Seaborn
- **部署**：Gradio

## 📂 项目结构
├── train.py # 模型训练入口├── test_loader.py # 数据加载器├── split_data.py # 数据集划分├── model_pytorch.py # ResNet18 模型定义├── model_efficientnet.py # EfficientNet-B3 模型定义├── predict_pytorch.py # ResNet18 预测脚本├── predict_efficientnet.py # EfficientNet-B3 预测脚本├── gradio_demo.py # Gradio 可视化 Demo├── visualization.py # 结果可视化（损失 / 准确率曲线）├── 数据集特征分析.py # 数据分布与可视化分析├── best_resnet18_model.pth # 训练好的 ResNet18 模型权重├── best_efficientnet_b3_model.pth # 训练好的 EfficientNet-B3 模型权重└── README.md
plaintext

## 🚀 快速开始
### 1. 安装依赖
```bash
pip install torch torchvision pandas numpy matplotlib seaborn gradio
2. 训练模型
bash
运行
python train.py
3. 运行预测 Demo
bash
运行
python gradio_demo.py
访问终端输出的本地地址即可上传图片进行测试。
📊 实验结果
ResNet18：测试集准确率约 94.4%
EfficientNet-B3：测试集准确率约 94.4%
训练过程可视化：
训练 / 验证损失曲线
训练 / 验证准确率曲线
数据类别分布直方图
样本散点图
📝 备注
数据集需放置在 data/ 目录下，按 cat/ 和 dog/ 子目录分类
模型权重文件较大，未上传至仓库，可通过 train.py 重新训练生成
如需复现实验，请确保环境与 requirements.txt 一致
📧 作者
GitHub: hrx1129
项目地址: https://github.com/hrx1129/cat-dog-classifier