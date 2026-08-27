from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rmsnorm_kernel',
    ext_modules=[
        CUDAExtension('rmsnorm_kernel', ['rmsnorm_kernel.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
