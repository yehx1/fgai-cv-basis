import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 定义SimpleCNN网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*16*16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16*16*16)
        x = self.fc1(x)
        return x

# 创建模拟训练数据
def create_fake_data(num_samples=1000):
    # 假设输入大小为(3, 32, 32)，即RGB图像32x32
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))  # 10个类别
    return images, labels

# 训练模型
def train_model(model, train_loader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])

    for epoch in range(config['epochs']):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {running_loss/len(train_loader)}")

# 保存模型和配置文件
def save_model_and_config(model, config, model_path, config_path):
    torch.save(model.state_dict(), model_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Model and config saved to {model_path} and {config_path}")

# 打包模型和配置
def create_package(model_path, config_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        zipf.write(model_path, os.path.basename(model_path))
        zipf.write(config_path, os.path.basename(config_path))
    print(f"Model and config packed into {output_zip}")

# 解压模型包
def extract_package(package_path, extract_to):
    if not os.path.exists(package_path):
        raise FileNotFoundError(f"Package file not found: {package_path}")
    
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(package_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Package extracted to {extract_to}")

# 加载模型
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = SimpleCNN()  # 重新实例化模型
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

# 加载配置文件
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# 配置参数示例
config = {
    "learning_rate": 0.001,
    "momentum": 0.9,
    "epochs": 5,
    "batch_size": 32,
    "num_samples": 1000
}

# 保存配置到文件
config_path = './config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

# 主程序入口
if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 设置训练目录和模型路径
    model_save_dir = os.path.join(save_dir, 'extracted_model')  # 解压路径
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, 'model.pth')
    config_path = os.path.join(model_save_dir, 'config.json')

    # 创建模型
    model = SimpleCNN()

    # 创建假数据并构建DataLoader
    images, labels = create_fake_data(config['num_samples'])
    train_dataset = TensorDataset(images, labels)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # 训练模型并保存
    train_model(model, train_loader, config)
    save_model_and_config(model, config, model_path, config_path)

    # 打包模型和配置
    zip_file = os.path.join(save_dir, 'model_package_with_config.zip')
    create_package(model_path, config_path, zip_file)

    # 演示解压并加载模型与配置
    extract_to = os.path.join(save_dir, 'unpacked_model')
    extract_package(zip_file, extract_to)

    # 加载解压后的模型与配置
    loaded_model = load_model(os.path.join(extract_to, 'model.pth'))
    loaded_config = load_config(os.path.join(extract_to, 'config.json'))

    print(f"Loaded config: {loaded_config}")
