#include <cuda_runtime.h>
#include <cstddef>
#include <cstdlib>

#include "kernels/vecAdd.cuh"
#include "helper.cuh"


int main() {
    int N = 1 << 20;
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    std::size_t bytes = N * sizeof(float);

    checkCudaErrors(cudaMallocManaged(&A, bytes));
    init(A, N);
    checkCudaErrors(cudaMallocManaged(&B, bytes));
    init(B, N);
    checkCudaErrors(cudaMallocManaged(&C, bytes));
    checkCudaErrors(cudaMemset(C, 0.0, bytes));
    solve(A, B, C, N);
}