import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pytorchcv.model_provider import get_model as ptcv_get_model  # 引入 pytorchcv

# 定义 MobileNetV1 模型，修改输入通道和全连接层
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV1, self).__init__()
        # 加载预训练的 MobileNetV1 模型
        self.mobilenet = ptcv_get_model("mobilenet_w1", pretrained=True)
        
        # 修改第一个卷积层输入通道为1（因为 MNIST 是单通道灰度图像）
        # self.mobilenet.features.init_block.conv.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # 修改分类器（全连接层）以适应 MNIST 的 10 类分类
        self.mobilenet.output = nn.Linear(self.mobilenet.output.in_features, num_classes)

    def forward(self, x):
        x = self.mobilenet(x)
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
        running_loss += loss.item() * data.size(0)
        
        # 计算准确率
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = 100. * correct / total
    
    print(f'Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%')
    return epoch_loss, epoch_accuracy

# 测试函数
def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return accuracy

# 绘制训练损失和测试准确率曲线
def plot_metrics(train_losses, accuracies, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.title('Train Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies[0], label='Train Accuracy')
    plt.plot(epochs, accuracies[1], label='Test Accuracy')
    plt.title('Train/Test Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
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

    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # MobileNetV1 需要输入 224x224 的图像
        transforms.Grayscale(num_output_channels=3),  # 将1通道灰度图像转换为3通道
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据的均值和标准差
    ])

    train_dataset = datasets.MNIST(root=f'{current_dir}/../01_data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=f'{current_dir}/../01_data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV1(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    accuracies = [[], []]

    for epoch in range(1, num_epochs + 1):
        loss, acc = train(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(loss)
        accuracies[0].append(acc)
        acc = test(model, device, test_loader)
        accuracies[1].append(acc)

    torch.save(model.state_dict(), f'{save_dir}/mobilenetv1_mnist.pth')
    print(f'model save path: {save_dir}/mobilenetv1_mnist.pth')

    plot_metrics(train_losses, accuracies, f'{save_dir}/mobilenetv1_loss_accuracy_curve.png')
