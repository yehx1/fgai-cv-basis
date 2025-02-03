import os
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from setuptools import setup


sources = ['src/reorg_cpu.c']
headers = ['src/reorg_cpu.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/reorg_cuda.c']
    headers += ['src/reorg_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
# print(this_file)
extra_objects = ['src/reorg_cuda_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

# 设置扩展模块
extension = (
    CUDAExtension(
        name='_ext.reorg_layer',
        sources=sources,
        define_macros=defines,
        extra_objects=extra_objects,
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
    ) if torch.cuda.is_available() else
    CppExtension(
        name='_ext.reorg_layer',
        sources=sources,
        define_macros=defines,
        extra_compile_args=['-O3']
    )
)

# 设置编译
setup(
    name='_ext.reorg_layer',
    ext_modules=[extension],
    cmdclass={'build_ext': BuildExtension}
)
# if __name__ == '__main__':
#     ffi.build()
