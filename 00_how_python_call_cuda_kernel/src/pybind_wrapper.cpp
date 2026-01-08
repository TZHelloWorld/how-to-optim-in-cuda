#include <pybind11/pybind11.h>

extern "C" void launch_cuda_hello();

PYBIND11_MODULE(cuda_hello, m) {
    m.def("hello", &launch_cuda_hello, "A function that launches a CUDA kernel to print Hello");
}