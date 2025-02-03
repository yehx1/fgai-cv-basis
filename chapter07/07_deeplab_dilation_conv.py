import torch
import torch.nn.functional as F

# 输入矩阵
input_matrix = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 扩展为 [batch_size, channels, height, width]
# 卷积核
kernel = torch.tensor([
    [2, 2],
    [2, 2]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 扩展为 [out_channels, in_channels, kernel_height, kernel_width]
# 空洞率（dilation rate）
dilation_rate = 2
# 执行空洞卷积
output = F.conv2d(input_matrix, kernel, dilation=dilation_rate)
print("输入矩阵：")
print(input_matrix.squeeze().numpy())  # 去掉多余维度以便输出更清晰
print("\n卷积核：")
print(kernel.squeeze().numpy())
print("\n空洞率：", dilation_rate)
print("\n输出矩阵：")
print(output.squeeze().detach().numpy())
