import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(
    base_dir, 
    batch_size=32, 
    img_size=150,  # 保留原默认尺寸150
    use_augmentation=True  # 新增：控制是否启用数据增强（默认启用，保持原逻辑）
):
    # 训练集预处理（核心逻辑不变，仅用参数控制增强开关）
    train_transform_list = [
        transforms.Resize((img_size, img_size)),
    ]
    # 仅在启用增强时添加数据增强操作（默认启用，与原逻辑一致）
    if use_augmentation:
        train_transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    # 统一添加张量转换和归一化（与原逻辑完全一致）
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_transform = transforms.Compose(train_transform_list)
    
    # 验证集/测试集预处理（完全保留原逻辑）
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 增强路径校验（新增：提前检测路径是否存在，避免训练中途报错）
    required_dirs = ['train', 'validation', 'test']
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(
                f"数据集子目录不存在：{dir_path}\n"
                f"请确保{base_dir}下包含{required_dirs}三个子文件夹"
            )
    
    # 加载数据集（逻辑不变，仅优化打印信息）
    train_dataset = datasets.ImageFolder(
        os.path.join(base_dir, 'train'), 
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(base_dir, 'validation'), 
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(base_dir, 'test'), 
        transform=val_test_transform
    )
    
    # 打印加载信息（补充类别信息，便于确认数据正确性）
    print(f"数据加载成功：")
    print(f"训练集: {len(train_dataset)}张 | 验证集: {len(val_dataset)}张 | 测试集: {len(test_dataset)}张")
    print(f"类别映射: {train_dataset.class_to_idx}（0:猫，1:狗）")
    
    # 构建加载器（完全保留原逻辑）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader