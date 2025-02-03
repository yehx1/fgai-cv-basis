import os
import numpy as np
import matplotlib.pyplot as plt

# 定义 Leaky ReLU 和 ReLU 函数
def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def relu(x):
    return np.maximum(0, x)

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
print("当前文件所在目录:", current_dir)
# 创建结果保存目录
save_dir = os.path.join(current_dir, 'result')
os.makedirs(save_dir, exist_ok=True)

# 创建输入值
x = np.linspace(-10, 10, 400)

# 计算输出值
y_leaky_relu = leaky_relu(x)
y_relu = relu(x)

# 绘制图形
plt.figure(figsize=(8, 6))

plt.plot(x, y_leaky_relu, label="Leaky ReLU (alpha=0.01)", color='blue', linewidth=2)
plt.plot(x, y_relu, label="ReLU", color='red', linestyle='--', linewidth=2)

plt.title("Leaky ReLU vs ReLU")
plt.xlabel("Input (x)")
plt.ylabel("Output (f(x))")
plt.legend()
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# 保存图像到文件
plt.savefig(f'{save_dir}/leaky_relu_vs_relu.png')

# 关闭图形，以避免显示
plt.close()
