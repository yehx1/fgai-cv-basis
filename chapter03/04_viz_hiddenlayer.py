# 1. 导入必要的库
import os
import torch
import torch.nn as nn
import hiddenlayer as hl

# 2. 定义神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# 1、运行程序如果提示错误：
# AttributeError: module 'torch.onnx' has no attribute '_optimize_trace'. Did you mean: '_optimize_graph'?
# 请将提示所在文件的_optimize_trace换成_optimize_graph即可。

# 2、如果提示错误：TypeError: 'torch._C.Node' object is not subscriptable
# 参考解决方案：https://github.com/waleedka/hiddenlayer/issues/100
# 在提示错误的pytorch_builder.py文件的import之后添加如下内容：
'''
# From https://github.com/pytorch/pytorch/blob/2efe4d809fdc94501fc38bf429e9a8d4205b51b6/torch/utils/tensorboard/_pytorch_graph.py#L384
def _node_get(node: torch._C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    sel = node.kindOf(key)
    return getattr(node, sel)(key)

torch._C.Node.__getitem__ = _node_get
'''

# 3、如果提示错误：FileNotFoundError: [Errno 2] No such file or directory: PosixPath('dot')
# 那么在系统中安装graphviz:
# apt update && apt install graphviz

if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 实例化模型
    model = SimpleCNN()

    # 3. 创建示例输入
    dummy_input = torch.randn(1, 1, 28, 28)  # 假设输入是 28x28 的灰度图像

    # 4. 生成并展示模型的计算图
    hl_graph = hl.build_graph(model, dummy_input)

    # 保存为图片文件，默认为pdf格式图片，result/viz_hiddenlayer.pdf
    hl_graph.save(f'{save_dir}/viz_hiddenlayer')
