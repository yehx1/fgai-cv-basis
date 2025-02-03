import torch
import torch.nn as nn

# 定义反卷积层
class DeconvExample(nn.Module):
    def __init__(self):
        super(DeconvExample, self).__init__()
        # 输入通道: 1，输出通道: 1，卷积核大小: 2x2，填充: 1，步长: 1
        self.deconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=0)
        # 手动设置卷积核权重为 2x2
        self.deconv.weight.data.fill_(2)
        # 手动设置偏置为 0
        self.deconv.bias.data.fill_(0)
    def forward(self, x):
        return self.deconv(x)

# 创建输入特征图（2x2）
input_tensor = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)
# 实例化网络并传入输入
deconv_model = DeconvExample()
# 打印输入特征图
print("Input Feature Map:")
print(input_tensor)
# 执行反卷积
output_tensor = deconv_model(input_tensor)
# 打印输出特征图
print("\nOutput Feature Map:")
print(output_tensor)
# 输出输出特征图的尺寸
print("\nOutput Size:", output_tensor.shape)
