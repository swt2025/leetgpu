#include <cuda_runtime.h>
#include <iostream>
#include "benchmark/cuda_timer.cuh"
#include "kernels/vecAdd.cuh"

__global__ void vector_add_0(const float* A, const float* B, float* C, int N) {
    // 当前线程的线性id
    int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadIndex < N) {
        C[threadIndex] = A[threadIndex] + B[threadIndex];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float time_ms = benchmark_cuda_kernel([&](){vector_add_0<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);});
    std::cout << "Spend " << time_ms << "ms" << std::endl;
}