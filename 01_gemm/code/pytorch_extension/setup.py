from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gemm_kernel',
    ext_modules=[
        CUDAExtension('gemm_kernel', ['gemm_kernel.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
