import torch
import torch.nn as nn
import torch.optim as optim
import time

# 定义一个简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 输出：26x26x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 输出：24x24x64
        self.fc1 = nn.Linear(64 * 24 * 24, 128)  # 输入需要和卷积输出匹配
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义训练函数
def train_model(model, criterion, optimizer, num_epochs=5, use_cudnn=True):
    # 设置 cuDNN 开关
    torch.backends.cudnn.enabled = use_cudnn
    print(f"cuDNN {'启用' if use_cudnn else '禁用'}")

    # 创建随机数据
    inputs = torch.randn(64, 1, 28, 28, device='cuda')  # 假设输入为64个28x28的灰度图像
    targets = torch.randint(0, 10, (64,), device='cuda')  # 64个目标类别

    model.train()
    start_time = time.time()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 强制同步，确保所有计算完成
    torch.cuda.synchronize()

    end_time = time.time()
    return end_time - start_time

if __name__ == '__main__':
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        # 初始化模型、损失函数和优化器
        model = SimpleCNN().cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 预热，包括前向与反向
        cudnn_enabled_time = train_model(model, criterion, optimizer, 1, use_cudnn=True)
        print(f"预热时间: {cudnn_enabled_time:.4f} 秒")

        # 测试启用 cuDNN 的时间
        cudnn_enabled_time = train_model(model, criterion, optimizer, use_cudnn=True)
        print(f"启用 cuDNN 训练时间: {cudnn_enabled_time:.4f} 秒")

        # 测试禁用 cuDNN 的时间
        cudnn_disabled_time = train_model(model, criterion, optimizer, use_cudnn=False)
        print(f"禁用 cuDNN 训练时间: {cudnn_disabled_time:.4f} 秒")
    else:
        print("CUDA 不可用，无法测试 cuDNN 对比。")

    # 输出示例
    # cuDNN 启用
    # Epoch [1/1], Loss: 2.3107
    # 预热时间: 0.3722 秒
    # cuDNN 启用
    # Epoch [1/5], Loss: 2.8140
    # Epoch [2/5], Loss: 2.0239
    # Epoch [3/5], Loss: 1.7684
    # Epoch [4/5], Loss: 1.4894
    # Epoch [5/5], Loss: 1.1285
    # 启用 cuDNN 训练时间: 0.0162 秒
    # cuDNN 禁用
    # Epoch [1/5], Loss: 2.6326
    # Epoch [2/5], Loss: 2.2912
    # Epoch [3/5], Loss: 2.0518
    # Epoch [4/5], Loss: 1.7783
    # Epoch [5/5], Loss: 1.4448
    # 禁用 cuDNN 训练时间: 0.0936 秒
