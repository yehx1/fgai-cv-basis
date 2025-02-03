import os
import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数
def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

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

# 生成数据点
x = np.linspace(-5, 5, 1000)

# 计算函数值
y_mish = mish(x)
y_leaky_relu = leaky_relu(x)
y_relu = relu(x)

# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制Mish函数
plt.plot(x, y_mish, label='Mish', color='blue')
# 绘制Leaky ReLU函数
plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='green', linestyle='--')
# 绘制ReLU函数
plt.plot(x, y_relu, label='ReLU', color='red', linestyle='-.')

# 图表标题与标签
plt.title('Comparison of Activation Functions (Mish, Leaky ReLU, ReLU)')
plt.xlabel('x')
plt.ylabel('y')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 保存图片
plt.savefig(f'{save_dir}/mish_activation_functions_comparison.png')  # 保存为PNG格式

# # 展示图形
# plt.show()
