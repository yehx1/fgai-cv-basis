import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
from torch.utils.data import DataLoader

# 定义一个浮点数模型，某些层可能从 QAT 中受益
class M(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # QuantStub 将张量从浮点数转换为量化形式
        self.quant = torch.ao.quantization.QuantStub()
        
        # 定义卷积层，批量归一化和 ReLU 激活函数
        self.conv = torch.nn.Conv2d(1, 16, 3, padding=1)  # 16个输出通道，3x3卷积核
        self.bn = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()
        
        # 定义一个全连接层，用于分类，假设最终的类别数为 10
        self.fc = torch.nn.Linear(16 * 28 * 28, num_classes)  # 将特征图展平并输出 num_classes 类别
        
        # DeQuantStub 将张量从量化形式转换回浮点数
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # 展平特征图为一维向量，准备送入全连接层
        # print(x.shape, x.size(0))
        x = x.reshape(x.size(0), -1)  # 展平
        x = self.fc(x)  # 通过全连接层
        
        x = self.dequant(x)
        return x

# 定义量化感知训练的训练循环
def training_loop(model, num_epochs=10, train_loader=None, criterion=None, optimizer=None):
    """
    量化感知训练的训练循环函数

    参数：
    - model: 量化感知训练模型（需要调用 `prepare_qat` 准备）
    - num_epochs: 训练的总 epoch 数
    - train_loader: 训练数据的 DataLoader
    - criterion: 损失函数（如 CrossEntropyLoss）
    - optimizer: 优化器（如 SGD）
    """
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            
            # 调整 target 的维度，确保是一个 1D 张量
            target = target.view(-1)  # 将 target 扁平化为 1D 张量，大小为 [batch_size]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
    print("训练完成")

# 创建一个模型实例
model_fp32 = M(num_classes=10)

# 模型必须设置为 eval，才能进行融合
model_fp32.eval()

# 附加一个全局 qconfig，包含关于应该使用什么类型的观察器的信息
model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

# 将激活层与前面的层融合，这需要根据模型架构手动完成
model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32,
    [['conv', 'bn', 'relu']])

# 准备模型进行 QAT，这将插入观察器和 fake_quant 操作
# 训练模式下模型将观察权重和激活张量
model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused.train())

# 生成训练数据和标签（示例）
train_data = torch.randn(100, 1, 28, 28)  # 随机生成的训练数据
train_labels = torch.randint(0, 10, (100,))  # 随机生成的标签（1D张量）
train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=32, shuffle=True)

# 确保目标标签是 1D 张量（类别索引）
assert train_labels.dim() == 1  # 检查标签是 1D 张量

# 使用交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()

# 使用 SGD 优化器
optimizer = optim.SGD(model_fp32.parameters(), lr=0.001, momentum=0.9)

# 调用训练循环
training_loop(model_fp32_prepared, num_epochs=5, train_loader=train_loader, criterion=criterion, optimizer=optimizer)

# 将模型转换为量化后的 int8 模型
model_fp32_prepared.eval()
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

# 进行推理，量化模型将在推理时执行 int8 计算
input_fp32 = torch.randn(10, 1, 28, 28)  # 随机生成测试数据
output = model_int8(input_fp32)

print("量化模型的输出:", output)
print(model_int8)
