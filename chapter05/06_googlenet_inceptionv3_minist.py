import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# 定义 Inception v3 模型，修改输入通道和全连接层
class InceptionV3(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionV3, self).__init__()
        # 使用预训练的 Inception v3 模型
        self.inception = models.inception_v3(pretrained=True)
        
        # 修改分类器（全连接层）以适应 MNIST 的 10 类分类
        self.inception.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Inception v3 需要 Auxiliary Logits，但在验证和推理时关闭它
        if self.training:
            x, _ = self.inception(x)
        else:
            x = self.inception(x)
        return x

# 训练函数（保持不变）
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

# 测试函数（保持不变）
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

# 绘制训练损失和测试准确率曲线（保持不变）
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

    # 修改数据预处理，将灰度图像复制到3个通道，并调整图像大小为 299x299
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Grayscale(num_output_channels=3),  # 将1通道灰度图像转换为3通道
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(root=f'{current_dir}/../01_data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=f'{current_dir}/../01_data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionV3(num_classes=10).to(device)
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
    torch.save(model.state_dict(), f'{save_dir}/inceptionv3_mnist.pth')
    print(f'model save path: {save_dir}/inceptionv3_mnist.pth')

    # 绘制并保存损失和准确率曲线
    plot_metrics(train_losses, accuracies, f'{save_dir}/inceptionv3_loss_accuracy_curve.png')