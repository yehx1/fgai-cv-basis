import os
import torch
import torch.onnx
import torch.nn as nn

# 定义网络
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

if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    trained_net = SimpleCNN()
    trained_net.load_state_dict(torch.load(f"{save_dir}/simple_cnn.pth"))
    trained_net.eval()

    # 构造一个 dummy_input
    dummy_input = torch.randn(1, 3, 32, 32)

    # 导出
    torch.onnx.export(
        trained_net, 
        dummy_input, 
        f"{save_dir}/simple_cnn.onnx",
        input_names=["input"], 
        output_names=["output"],
        opset_version=11
    )