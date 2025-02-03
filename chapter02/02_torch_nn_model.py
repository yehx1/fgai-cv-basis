import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 自定义激活函数 Swish
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 定义一个使用自定义激活函数的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.swish = Swish()  # 使用自定义的 Swish 激活函数
        self.fc1 = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        x = self.swish(self.conv1(x))  # 使用 Swish 激活函数
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        return x

# 随机生成数据
def generate_random_data(batch_size=64, img_size=(1, 28, 28), num_classes=10):
    inputs = torch.randn(batch_size, *img_size)
    targets = torch.randint(0, num_classes, (batch_size,))
    return inputs, targets

# 训练模型并记录损失
def train_model(model, criterion, optimizer, num_epochs=10):
    loss_history = []
    for epoch in range(num_epochs):
        inputs, targets = generate_random_data()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        loss_history.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return loss_history

# 绘制损失曲线
def plot_loss_curve(loss_history, save_dir):
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig(f'{save_dir}/loss.png')
    plt.show()

if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 初始化模型、损失函数和优化器
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型并获取损失历史
    loss_history = train_model(model, criterion, optimizer, num_epochs=200)

    # 绘制损失曲线
    plot_loss_curve(loss_history, save_dir)

    print('Completed.')
