import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import sys
import os

# ====================== 核心路径适配（打包后可用）======================
def resource_path(relative_path):
    """获取打包后的资源路径，兼容开发/打包两种模式"""
    if hasattr(sys, '_MEIPASS'):
        # 打包后运行时的路径
        return os.path.join(sys._MEIPASS, relative_path)
    # 开发模式下的路径
    return os.path.join(os.path.abspath("."), relative_path)

# ====================== 模型加载与配置 ======================
# 1. 初始化ResNet18模型
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 二分类适配

# 2. 加载并清理权重（移除resnet.前缀）
try:
    state_dict = torch.load(resource_path("best_resnet18_model.pth"), map_location="cpu")
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("resnet."):
            new_key = key[7:]  # 去掉前缀 "resnet."
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    model.eval()  # 推理模式
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = f"模型加载失败：{str(e)}\n请确保 best_resnet18_model.pth 文件与exe同目录！"

# 3. 图片预处理（与训练完全一致）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== 核心识别函数 ======================
def predict_cat_dog(image):
    """单张图片识别核心函数"""
    # 检查模型是否加载成功
    if not model_loaded:
        return f"### ❌ 模型加载失败\n{load_error}"
    
    try:
        # 图片格式转换与预处理
        img = Image.fromarray(np.uint8(image)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
        
        # 模型推理（关闭梯度计算）
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]
            cat_prob, dog_prob = probs[0], probs[1]
        
        # 格式化识别结果
        result = f"""
### 🐱🐶 识别结果
| 类别 | 置信度 |
|------|--------|
| 猫   | {cat_prob:.2%} |
| 狗   | {dog_prob:.2%} |

### 最终判断
{"> 猫" if cat_prob > dog_prob else "> 狗"}（置信度：{max(cat_prob, dog_prob):.2%}）

### 模型信息
ResNet18（轻量化） | 验证集准确率：94.40% | 运行环境：本地CPU
        """
        return result
    
    except Exception as e:
        return f"""
### ❌ 识别失败
错误原因：{str(e)}

请检查：
1. 上传的是否为有效图片（jpg/png格式）
2. 图片是否完整无损坏
        """

def predict_batch(files):
    """批量图片识别函数"""
    if not model_loaded:
        return f"模型加载失败：{load_error}"
    
    results = []
    if not files:
        return "请上传至少一张图片！"
    
    for file in files:
        try:
            img = Image.open(file).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1).numpy()[0]
                cat_prob, dog_prob = probs[0], probs[1]
                pred = "猫" if cat_prob > dog_prob else "狗"
            
            results.append(f"📸 {os.path.basename(file.name)}：{pred}（猫：{cat_prob:.2%} / 狗：{dog_prob:.2%}）")
        except Exception as e:
            results.append(f"❌ {os.path.basename(file.name)}：识别失败 - {str(e)}")
    
    return "\n\n".join(results)

def close_app():
    """关闭应用函数"""
    sys.exit(0)

# ====================== Gradio界面构建（适配桌面应用）======================
with gr.Blocks(
    title="智能猫狗识别系统",
    theme=gr.themes.Soft()
) as demo:
    # 标题区域
    gr.Markdown("""
    # 🐱🐶 智能猫狗识别系统
    ### 基于ResNet18的轻量化识别模型 | 支持单张/批量图片识别
    """)
    
    # 模型状态提示
    if model_loaded:
        gr.Markdown("✅ 模型加载成功 | 验证集准确率：94.40%")
    else:
        gr.Markdown(f"❌ {load_error}")
    
    # 单张识别区域
    gr.Markdown("### 📷 单张图片识别")
    with gr.Row():
        img_input = gr.Image(
            type="numpy",
            label="上传猫狗图片（jpg/png格式）",
            height=400
        )
        result_output = gr.Markdown(
            label="识别结果",
            value="### 请上传图片并点击「开始识别」"
        )
    
    with gr.Row():
        predict_btn = gr.Button("开始识别", variant="primary", size="lg")
        clear_btn = gr.Button("清空", size="lg")
    
    # 批量识别区域
    gr.Markdown("### 📦 批量图片识别")
    with gr.Row():
        batch_input = gr.Files(
            label="上传多张猫狗图片（支持批量选择）",
            file_count="multiple",
            file_types=["image/jpeg", "image/png"]
        )
        batch_output = gr.Textbox(
            label="批量识别结果",
            lines=10,
            value="### 请上传图片并点击「批量识别」"
        )
    
    batch_btn = gr.Button("批量识别", variant="secondary", size="lg")
    
    # 关闭应用按钮
    with gr.Row():
        close_btn = gr.Button("关闭应用", variant="stop", size="lg")
    
    # 绑定按钮事件
    predict_btn.click(predict_cat_dog, inputs=img_input, outputs=result_output)
    clear_btn.click(
        fn=lambda: (None, "### 请上传图片并点击「开始识别」"),
        outputs=[img_input, result_output]
    )
    batch_btn.click(predict_batch, inputs=batch_input, outputs=batch_output)
    close_btn.click(close_app)
    
    # 底部信息
    gr.Markdown("""
    ---
    💻 桌面版应用 | 无需Python环境 | 本地CPU运行
    """)

# ====================== 启动配置（适配桌面打包）======================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",    # 仅本地访问
        server_port=7860,           # 固定端口
        share=False,                # 关闭公网分享（打包后必须关闭）
        inbrowser=True,             # 自动打开浏览器
        prevent_thread_lock=True,   # 防止主线程阻塞（打包关键）
        show_error=True             # 显示错误信息，方便调试
    )