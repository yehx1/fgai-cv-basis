import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义LeNet-5模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)  # 输入为1通道，输出为6通道，5x5卷积核
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2池化
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 输入为6通道，输出为16通道，5x5卷积核
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2池化
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层1，输入为16x5x5，输出为120
        self.fc2 = nn.Linear(120, 84)  # 全连接层2，输出为84
        self.fc3 = nn.Linear(84, 10)  # 输出层，输出为10类

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 训练模型
def train(model, device, train_loader, optimizer, criterion, epoch, train_losses):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:  # 每100个batch打印一次训练日志
            print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

# 测试模型 (不记录Test Loss)
def test(model, device, test_loader, accuracies):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # 取概率最高的类
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    accuracies.append(accuracy)

    print(f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")

# 绘制训练损失和测试准确率曲线
def plot_metrics(train_losses, accuracies, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    
    # 绘制Train Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.title('Train Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制Test Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Test Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # 保存图像
    plt.savefig(save_path)
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

    # 定义超参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=f'{current_dir}/../01_data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=f'{current_dir}/../01_data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化保存损失和准确率的列表
    train_losses = []
    accuracies = []

    # 开始训练和测试
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, criterion, epoch, train_losses)
        test(model, device, test_loader, accuracies)

    # 保存模型
    torch.save(model.state_dict(), f'{save_dir}/lenet5_mnist.pth')
    print(f'model save path: {save_dir}/lenet5_mnist.pth')

    # 绘制并保存损失和准确率曲线
    plot_metrics(train_losses, accuracies, f'{save_dir}/lenet5_loss_accuracy_curve.png')
