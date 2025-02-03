import torch
import numpy as np
import torch.nn.functional as F

# 输出torch版本，CUDA和CUDNN可用状态
def torch_version():
    print("PyTorch版本：",torch.__version__)
    print("CUDA是否可用：", torch.cuda.is_available())
    print("cuDNN是否可用：", torch.backends.cudnn.is_available())
    print("CUDA版本：", torch.version.cuda)
    print("cuDNN版本：", torch.backends.cudnn.version())

# Tensor: 张量
def tensor_create():
    # 创建一个张量
    x = torch.rand(5, 3)
    print('x cpu: ', x)
    # 将张量移动到GPU
    if torch.cuda.is_available():
        x = x.cuda()
    print('x cuda: ', x)

# Autograd 自动求导
def autograd_test():
    # 创建一个可求导的张量
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2
    y.backward(torch.tensor([1.0, 1.0, 1.0]))
    print(x.grad)  # 输出：tensor([2., 4., 6.])

    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = x * y + y ** 2
    z.backward()
    print(x.grad)  # 输出：2.0
    print(y.grad)  # 输出：5.0

    input = torch.randn(1, 1, 5, 5, requires_grad=True)
    weight = torch.randn(1, 1, 3, 3, requires_grad=True)
    output = F.conv2d(input, weight)
    output.backward(torch.ones_like(output))
    print(input.grad.size())  # 输出：torch.Size([1, 1, 5, 5])
    print(weight.grad.size())  # 输出：torch.Size([1, 1, 3, 3])

# 基本运算操作
def operation_basis():
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    c = a + b  # 张量相加
    d = torch.matmul(a, b.T)  # 矩阵乘法
    print('d: ', d)

# numpy转换
def np_convert():
    # Tensor转为NumPy数组
    a = torch.ones(5)
    b = a.numpy()
    print('b: ', b)
    # NumPy数组转为Tensor
    c = np.array([1, 2, 3])
    d = torch.from_numpy(c)
    print('d: ', d)


if __name__ == '__main__':
    # 1、torch版本、CUDA、CUDNN状态
    torch_version()

    # 2、Tensor张量创建
    tensor_create()

    # 3、Autograd 自动求导
    autograd_test()

    # 4、基本运算操作
    operation_basis()

    # 5、numpy转换
    np_convert()


