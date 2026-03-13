import data_loader

if __name__ == "__main__":
    base_dir = "/home/stu021/cats_and_dogs_small"
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(base_dir, batch_size=20)
    print(f"训练集批次：{len(train_loader)}")
    print(f"验证集批次：{len(val_loader)}")
    print(f"测试集批次：{len(test_loader)}")
