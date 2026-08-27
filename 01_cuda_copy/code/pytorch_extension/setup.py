from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='copy_kernel',
    ext_modules=[
        CUDAExtension('copy_kernel', ['copy_kernel.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
