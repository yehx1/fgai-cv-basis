import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# 定义LeNet-5模型 (与训练时相同)
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

# 加载训练好的模型
def load_model(model_path):
    model = LeNet5()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 切换到评估模式
    return model

# 使用 TensorBoard 可视化模型
def visualize_model(model, log_dir):
    # 创建一个 SummaryWriter 实例，用于将模型信息记录到 TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # 创建一个虚拟的输入张量，模拟网络的输入
    dummy_input = torch.randn(1, 1, 32, 32)

    # 将模型结构添加到 TensorBoard
    writer.add_graph(model, dummy_input)

    # 关闭 writer
    writer.close()
    print(f"模型结构已保存到 {log_dir} 中，可以使用 TensorBoard 查看。")

if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)


    # 定义模型文件路径
    model_path = f'{current_dir}/../02_models/lenet5_mnist.pth'

    # 加载模型
    model = load_model(model_path)
    print("模型已加载")

    # TensorBoard 日志目录
    log_dir = f'{save_dir}/logs'
    os.makedirs(log_dir, exist_ok=True)

    # 可视化模型
    visualize_model(model, log_dir)

    print(f'可视化已完成，请使用命令 `tensorboard --logdir={log_dir}` 启动 TensorBoard 查看模型结构。')
