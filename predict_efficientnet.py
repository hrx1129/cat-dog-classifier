import torch
from torchvision import transforms
from PIL import Image
import os
import sys
from model_efficientnet import get_efficientnet_model

# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 增加批次维度

# 预测函数
def predict(image_path, model_path='best_efficientnet_b3_model.pth'):
    # 加载模型
    model = get_efficientnet_model(freeze_ratio=0.9)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 预处理图像
    image = preprocess_image(image_path)
    
    # 预测
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, 1).numpy()[0]
    
    # 输出结果
    class_names = ['cat', 'dog']
    result = class_names[preds.item()]
    print(f"预测类别: {result}")
    print(f"类别概率: 猫 {probs[0]:.4f}, 狗 {probs[1]:.4f}")
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python predict_efficientnet.py 图片路径")
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"错误: 图片文件 '{image_path}' 不存在")
        sys.exit(1)
    predict(image_path)