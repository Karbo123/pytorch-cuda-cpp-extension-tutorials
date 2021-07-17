#include <cuda_runtime.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.dtype()==torch::kFloat32, #x " must be float32")
#define CHECK_1DARRAY(x) TORCH_CHECK(x.ndimension()==1, #x " must be 1d array")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x); CHECK_1DARRAY(x)

__global__ void cuSaxpy(float* x, float* y, float* z, 
                        float a, const int size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        z[i] = a * x[i] + y[i];
    }
}

torch::Tensor saxpy(torch::Tensor x, torch::Tensor y, float a)
{
    CHECK_INPUT(x); CHECK_INPUT(y);
    TORCH_CHECK(x.size(0) == y.size(0), "size mismatch");

    auto z = torch::empty_like(x);
    const int size = x.size(0);
    const int threads = std::min(size, 1024);
    const int blocks  = std::ceil(size / 1024.0f);
    cuSaxpy <<<blocks, threads>>> (x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), a, size);
    return z;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("saxpy", &saxpy, "SAXPY (CUDA)");
}
