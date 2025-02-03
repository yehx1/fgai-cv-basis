import torch
import torch.nn as nn

class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        # 获取输入的批量大小、通道数、高度和宽度
        batch_size, channels, height, width = x.size()

        assert height % self.stride == 0 and width % self.stride == 0, \
            "Height and width should be divisible by stride."

        # 重新组织维度
        new_height = height // self.stride
        new_width = width // self.stride
        new_channels = channels * (self.stride ** 2)

        # 进行维度重组
        x = x.view(batch_size, channels, new_height, self.stride, new_width, self.stride)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, new_channels, new_height, new_width)

        return x
