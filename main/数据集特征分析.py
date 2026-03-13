import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# 设置中文字体（确保中文显示正常）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 数据集路径（替换为你的实际路径）
base_dir = "D:\\cats_and_dogs_small"  
train_cats_dir = os.path.join(base_dir, "train", "cats")
train_dogs_dir = os.path.join(base_dir, "train", "dogs")

# 提取样本特征（随机采样，避免计算量过大）
def extract_features(img_dir, sample_size=500):
    widths = []  # 图像宽度
    heights = []  # 图像高度
    brightness = []  # 平均亮度（0~255）
    all_imgs = os.listdir(img_dir)
    sampled_imgs = random.sample(all_imgs, min(sample_size, len(all_imgs)))  # 随机选样本
    
    for img_name in sampled_imgs:
        img_path = os.path.join(img_dir, img_name)
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")  # 统一转为RGB
                w, h = img_rgb.size  # 尺寸
                widths.append(w)
                heights.append(h)
                # 计算平均亮度
                img_np = np.array(img_rgb)
                avg_bright = np.mean(img_np)  # 0~255
                brightness.append(avg_bright)
        except Exception as e:
            print(f"跳过损坏图像 {img_name}：{e}")
    return widths, heights, brightness

# 提取猫和狗的特征（核心数据）
cat_widths, cat_heights, cat_bright = extract_features(train_cats_dir)
dog_widths, dog_heights, dog_bright = extract_features(train_dogs_dir)


# ---------------------- 1. 绘制直方图（2个子图，在同一张图中） ----------------------
plt.figure(figsize=(12, 5))  # 新建画布

# 子图1：宽度分布
plt.subplot(1, 2, 1)  # 1行2列，第1个位置
plt.hist(cat_widths, bins=20, alpha=0.5, label="猫", color="blue")
plt.hist(dog_widths, bins=20, alpha=0.5, label="狗", color="orange")
plt.xlabel("图像宽度（像素）")
plt.ylabel("样本数量")
plt.title("猫和狗图像宽度分布")
plt.legend()

# 子图2：亮度分布
plt.subplot(1, 2, 2)  # 1行2列，第2个位置
plt.hist(cat_bright, bins=20, alpha=0.5, label="猫", color="blue")
plt.hist(dog_bright, bins=20, alpha=0.5, label="狗", color="orange")
plt.xlabel("平均亮度（0~255）")
plt.ylabel("样本数量")
plt.title("猫和狗图像亮度分布")
plt.legend()

plt.tight_layout()  # 自动调整布局，避免标签重叠
plt.savefig("histograms.png", dpi=300)  # 保存直方图
plt.show()  # 显示直方图（这一步会阻塞，关闭后才会显示散点图）


# ---------------------- 2. 绘制散点图（单独一张图） ----------------------
plt.figure(figsize=(8, 6))  # 新建另一个画布
plt.scatter(cat_widths, cat_heights, alpha=0.5, label="猫", color="blue", s=10)
plt.scatter(dog_widths, dog_heights, alpha=0.5, label="狗", color="orange", s=10)
plt.xlabel("图像宽度（像素）")
plt.ylabel("图像高度（像素）")
plt.title("猫和狗图像尺寸分布（宽vs高）")
plt.legend()
plt.grid(linestyle="--", alpha=0.7)
plt.savefig("scatter_plot.png", dpi=300)  # 保存散点图
plt.show()  # 显示散点图