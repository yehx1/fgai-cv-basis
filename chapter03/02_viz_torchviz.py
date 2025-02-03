# 1. 导入必要的库
import os
import torch
from torch import nn
from torchviz import make_dot

# 2. 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    model = SimpleModel()

    # 3. 创建示例输入
    x = torch.randn(1, 10)

    # 4. 获取模型的输出
    y = model(x)

    # 5. 生成计算图
    dot = make_dot(y, params=dict(model.named_parameters()))

    # 6. 保存和展示计算图
    dot.format = 'png'
    dot.render(f'{save_dir}/viz_torchviz')

    # 6. 展示计算图
    # dot.view()
