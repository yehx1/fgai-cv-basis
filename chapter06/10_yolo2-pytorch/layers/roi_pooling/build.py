import os
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from setuptools import setup

sources = ['src/roi_pooling.c']
headers = ['src/roi_pooling.h']
defines = []
extra_objects = []

# 检查是否使用 CUDA
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/roi_pooling_cuda.c']
    headers += ['src/roi_pooling_cuda.h']
    defines += [('WITH_CUDA', None)]
    extra_objects += [os.path.join('src', 'cuda', 'roi_pooling_kernel.cu')]

# 设置扩展模块
extension = (
    CUDAExtension(
        name='_ext.roi_pooling',
        sources=sources,
        define_macros=defines,
        extra_objects=extra_objects,
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
    ) if torch.cuda.is_available() else
    CppExtension(
        name='_ext.roi_pooling',
        sources=sources,
        define_macros=defines,
        extra_compile_args=['-O3']
    )
)

# 设置编译
setup(
    name='_ext.roi_pooling',
    ext_modules=[extension],
    cmdclass={'build_ext': BuildExtension}
)
