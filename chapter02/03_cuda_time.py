import torch
import time

# 定义矩阵大小
matrix_size = 10000

# 创建一个函数来测量运行时间
def measure_time(device):
    # 创建两个随机矩阵
    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)
    
    # 记录开始时间
    start_time = time.time()

    # 执行矩阵相乘
    C = torch.matmul(A, B)

    # 强制同步，确保所有计算完成（尤其在 CUDA 中）
    if device == 'cuda':
        torch.cuda.synchronize()

    # 记录结束时间
    end_time = time.time()

    # 返回运行时间
    return end_time - start_time

if __name__ == '__main__':
    # 测量在 CPU 上的时间
    cpu_time = measure_time('cpu')
    print(f"CPU 运行时间: {cpu_time:.4f} 秒")

    # 如果 CUDA 可用，测量在 GPU 上的时间
    if torch.cuda.is_available():
        cuda_time = measure_time('cuda')
        print(f"CUDA 运行时间: {cuda_time:.4f} 秒")
    else:
        print("CUDA 不可用")

    print(f'GPU运行速度是CPU运行速度的{cpu_time // cuda_time}倍。')

# 输出示例
# CPU 运行时间: 2.2207 秒
# CUDA 运行时间: 0.1791 秒
# GPU运行速度是CPU运行速度的12.0倍。