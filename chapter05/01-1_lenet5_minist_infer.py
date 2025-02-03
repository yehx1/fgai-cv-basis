import os
import torch
from torchvision import transforms, datasets
from PIL import Image
import torch.nn as nn
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
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 推理函数
def infer(model, device, image_tensor):
    # 进行推理
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不需要计算梯度
        output = model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率的标签
        return pred.item()  # 返回预测的类标签

# 主函数
if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练好的模型
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(f'{current_dir}/../02_models/fgai_trained_models/chapter05/lenet5_mnist.pth'))
    model.eval()  # 设置模型为评估模式

    # 加载训练集
    train_dataset = datasets.MNIST(root=f'{current_dir}/../01_data', train=True, transform=None, download=True)
    
    # 随机选择一个训练样本
    idx = torch.randint(0, len(train_dataset), (1,)).item()  # 随机选择一个索引
    image, label = train_dataset[idx]  # 获取该样本的图像和真实标签

    # 在推理部分进行图像预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 预处理图像
    image_tensor = transform(image).unsqueeze(0).to(device)  # 应用transform，添加batch维度，并转移到设备

    # 调用推理函数
    predicted_label = infer(model, device, image_tensor)
    
    # 打印真实标签与预测标签
    print(f"真实标签: {label}")
    print(f"预测标签: {predicted_label}")

    # 显示图像
    plt.imshow(image, cmap='gray')  # 显示原始图片
    plt.title(f"Real: {label}, Predicted: {predicted_label}")
    plt.axis('off')  # 不显示坐标轴

    # 保存图像
    save_path = os.path.join(save_dir, 'lenet5_infer.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  # 保存图像
    print(f"结果图片已保存至: {save_path}")
    plt.show()
