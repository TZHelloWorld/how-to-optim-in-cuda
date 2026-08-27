from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="layernorm_ext",
    ext_modules=[
        CUDAExtension(
            name="layernorm_ext",
            sources=["layernorm_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-arch=sm_70"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
