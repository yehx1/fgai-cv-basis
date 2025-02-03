import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义 AlexNet 模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  # MNIST 输入是 1 通道
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # 修正为 256*6*6
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)  # 展开为线性层输入
        x = self.classifier(x)
        return x

# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # 记录损失
        running_loss += loss.item() * data.size(0)  # 按样本数加权
        
        # 计算准确率
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # 计算并记录每个 epoch 的平均损失和准确率
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = 100. * correct / total
    
    print(f'Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%')
    return epoch_loss, epoch_accuracy

# 测试模型
def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # 取概率最高的类
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return accuracy

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
    plt.plot(epochs, accuracies[0], label='Train Accuracy')
    plt.plot(epochs, accuracies[1], label='Test Accuracy')
    plt.title('Train/Test Accuracy over Epochs')
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

    # 超参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # AlexNet 需要输入 224x224 的图像
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据的均值和标准差
    ])

    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(root=f'{current_dir}/../01_data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=f'{current_dir}/../01_data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 用于保存损失和准确率
    train_losses = []
    accuracies = [[], []]

    # 训练和保存曲线的主程序
    for epoch in range(1, num_epochs + 1):
        loss, acc = train(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(loss)
        accuracies[0].append(acc)
        acc = test(model, device, test_loader)
        accuracies[1].append(acc)


    # 保存模型
    torch.save(model.state_dict(), f'{save_dir}/alexnet_mnist.pth')
    print(f'model save path: {save_dir}/alexnet_mnist.pth')

    # 绘制并保存损失和准确率曲线
    plot_metrics(train_losses, accuracies, f'{save_dir}/alexnet_loss_accuracy_curve.png')
