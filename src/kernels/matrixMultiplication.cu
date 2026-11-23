#include "kernels/matrixMultiplication.cuh"
#include "helper.cuh"
#include "benchmark/cuda_timer.cuh"
#include <cuda/cmath>

__global__ void matrix_multiplication_kernel_0(
    const float* A, const float* B, float* C, int M, int N, int K) {
        // 每个线程对应C的一个元素
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (col >= K || row >= M) {
            return ;
        }

        float sum = 0.0f;

        for(int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
}

extern "C" void solveMatrixMultiplication(
    const float* A, const float* B, float* C, 
    int M, int N, int K) {
        dim3 threadsPerBlock(16,16);

        // 需要保证每个线程对应一个C的元素
        dim3 blocksPerGrid(cuda::ceil_div(K, threadsPerBlock.x),cuda::ceil_div(M, threadsPerBlock.y));

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        float time_ms = benchmark_cuda_kernel([&](){
            matrix_multiplication_kernel_0<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
            checkCudaErrors(cudaGetLastError());
        });

        std::cout << "Spend " << time_ms << "ms" << std::endl;
    }