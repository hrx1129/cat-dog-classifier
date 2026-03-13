import torch
import torch.nn as nn
from model_pytorch import get_resnet18_model  # 从model_pytorch.py导入模型函数
from torchvision import transforms
from PIL import Image
import os
import sys

# 加载模型（添加路径验证）
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先训练模型生成.pth文件")
    
    model = get_resnet18_model()  # 调用model_pytorch.py中的模型生成函数
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"模型加载成功，使用设备: {device}")
    return model

# 预处理单张图片（与训练时的验证集预处理一致）
def preprocess_image(image_path, img_size=150):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    try:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor
    except Exception as e:
        raise RuntimeError(f"图片预处理失败: {str(e)}")

# 预测函数
def predict_image(image_path, model):
    img_tensor = preprocess_image(image_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        # 新增.cpu()将CUDA张量转移到CPU后再转NumPy
        probs = torch.softmax(outputs, 1).cpu().numpy()[0]  
    
    class_names = ['cat', 'dog']
    result = class_names[preds.item()]
    print(f'预测类别：{result}')
    print(f'类别概率：猫 {probs[0]:.4f}，狗 {probs[1]:.4f}')
    return result

# 主函数
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python predict_pytorch.py 图片路径")
        print("示例: python predict_pytorch.py cat.1503.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = 'best_resnet18_model.pth'  # 匹配train.py生成的ResNet18权重文件
    
    try:
        model = load_model(model_path)
        predict_image(image_path, model)
    except Exception as e:
        print(f"预测失败: {str(e)}")
        sys.exit(1)