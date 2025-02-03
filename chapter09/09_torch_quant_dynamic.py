import torch

# 定义一个使用 LSTM 的浮动点模型
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个 LSTM 层，输入大小为 256，隐藏层大小为 512，层数为 2
        self.lstm = torch.nn.LSTM(input_size=256, hidden_size=512, num_layers=2)
        # 定义一个全连接层，用于输出结果
        self.fc = torch.nn.Linear(512, 10)  # 假设输出是一个大小为 10 的分类问题

    def forward(self, x):
        # LSTM 层的输出包括 hidden_state 和 cell_state
        # 由于是多层 LSTM, 我们会得到两个输出: output 和 (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 将 LSTM 的最后一个时间步的隐藏状态传递给全连接层
        out = self.fc(h_n[-1])  # 选择最后一层的隐藏状态
        return out

# 创建一个浮动点模型实例
model_fp32 = M()

# 创建一个动态量化的模型实例
# 将 LSTM 层进行动态量化
model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp32,  # 原始的浮动点模型
    {torch.nn.LSTM, torch.nn.Linear},  # 指定要动态量化的层，这里是 LSTM 层
    dtype=torch.qint8)  # 量化后的数据类型为 int8

# 创建一个输入张量，假设输入是一个形状为 (序列长度, batch_size, 输入大小) 的张量
input_fp32 = torch.randn(10, 32, 256)  # 假设序列长度为 10，batch_size 为 32，输入特征为 256

# 运行量化后的模型
res = model_int8(input_fp32)  # 获取模型的输出
print(res)  # 打印输出结果
print(model_int8)