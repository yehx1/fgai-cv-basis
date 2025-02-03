import torch

# 定义一个浮动点模型，其中一些层可能会被静态量化
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub 用于将浮动点张量转换为量化张量
        self.quant = torch.ao.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)  # 卷积层
        self.relu = torch.nn.ReLU()  # 激活函数 ReLU
        # DeQuantStub 用于将量化张量转换为浮动点张量
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # 手动指定在量化模型中，张量从浮动点转换为量化时的位置
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        # 手动指定在量化模型中，张量从量化转换为浮动点时的位置
        x = self.dequant(x)
        return x

# 创建一个模型实例
model_fp32 = M()

# 模型必须设置为评估模式，以便静态量化逻辑生效
model_fp32.eval()

# 附加一个全局的量化配置（qconfig），该配置包含有关附加哪些观察器的信息
# 对于服务器推理，使用 'x86'；对于移动推理，使用 'qnnpack'
# 其他量化配置，如选择对称或非对称量化，以及 MinMax 或 L2Norm 校准技术，也可以在这里指定。
# 注意：旧的 'fbgemm' 仍然可用，但推荐默认使用 'x86' 进行服务器推理。
# model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')

# 将激活层与前面的层融合，这需要根据模型架构手动完成
# 常见的融合方式包括 `conv + relu` 和 `conv + batchnorm + relu`
model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

# 准备模型进行静态量化。此步骤将插入观察器，
# 在校准过程中观察激活张量
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)

# 校准准备好的模型，以确定激活的量化参数
# 在实际应用中，校准应使用代表性的训练数据集
input_fp32 = torch.randn(4, 1, 4, 4)  # 创建一个模拟输入
model_fp32_prepared(input_fp32)

# 将已观察的模型转换为量化模型。此过程包括：
# 1. 量化权重，
# 2. 计算并存储每个激活张量的缩放因子和偏差值，
# 3. 将关键操作替换为量化实现。
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

# 运行量化后的模型，相关计算将在 int8 精度下进行
res = model_int8(input_fp32)
print(res)
print(model_int8)
