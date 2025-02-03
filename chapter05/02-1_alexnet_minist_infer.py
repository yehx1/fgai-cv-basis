import os
import torch
import torch.nn as nn
import torch.optim as optim
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
    model = AlexNet(num_classes=10).to(device)
    model.load_state_dict(torch.load(f'{current_dir}/../02_models/fgai_trained_models/chapter05/alexnet_mnist.pth'))
    model.eval()  # 设置模型为评估模式

    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(root=f'{current_dir}/../01_data', train=True, transform=None, download=True)
    
    # 随机选择一个训练样本
    idx = torch.randint(0, len(train_dataset), (1,)).item()  # 随机选择一个索引
    image, label = train_dataset[idx]  # 获取该样本的图像和真实标签

    # 在推理部分进行图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # AlexNet 需要输入 224x224 的图像
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
    save_path = os.path.join(save_dir, 'alexnet_infer.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  # 保存图像
    print(f"结果图片已保存至: {save_path}")
    plt.show()
