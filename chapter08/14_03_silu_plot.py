import os
import numpy as np
import matplotlib.pyplot as plt

# 定义ReLU函数
def relu(x):
    return np.maximum(0, x)

# 定义SiLU（Swish）函数：SiLU(x) = x * sigmoid(x)
def silu(x):
    return x / (1 + np.exp(-x))


# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
print("当前文件所在目录:", current_dir)
# 创建结果保存目录
save_dir = os.path.join(current_dir, 'result')
os.makedirs(save_dir, exist_ok=True)

# 创建输入数据
x = np.linspace(-6, 6, 200)  # 在-6到6之间生成200个点

# 计算输出
y_relu = relu(x)
y_silu = silu(x)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(x, y_relu, label='ReLU', linewidth=2, color='blue')
plt.plot(x, y_silu, label='SiLU', linewidth=2, color='red')

# 添加网格、标题和标签
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Comparison of ReLU and SiLU Activation Functions', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(fontsize=12)

plt.savefig(f'{save_dir}/silu_vs_relu.png')
