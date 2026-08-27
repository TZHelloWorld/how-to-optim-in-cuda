from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='reduce_kernel',
    ext_modules=[
        CUDAExtension('reduce_kernel', ['reduce_kernel.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
