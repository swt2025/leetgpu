#include <cuda_runtime.h>
#include <cstddef>
#include <cstdlib>

#include "kernels/vecAdd.cuh"
#include "kernels/matrixMultiplication.cuh"
#include "helper.cuh"

void execVecAdd()
{
    int N = 1 << 20;
    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;
    std::size_t bytes = N * sizeof(float);

    checkCudaErrors(cudaMallocManaged(&A, bytes));
    init(A, N);
    checkCudaErrors(cudaMallocManaged(&B, bytes));
    init(B, N);
    checkCudaErrors(cudaMallocManaged(&C, bytes));
    checkCudaErrors(cudaMemset(C, 0.0, bytes));
    solveVecAdd(A, B, C, N);
}

void execMatrixMultiplication()
{
    int M = 100;
    int N = 200;
    int K = 100;
    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;

    checkCudaErrors(cudaMallocManaged(&A, M * N * sizeof(float)));
    init(A, M * N);
    checkCudaErrors(cudaMallocManaged(&B, N * K * sizeof(float)));
    init(B, N * K);
    checkCudaErrors(cudaMallocManaged(&C, M * K * sizeof(float)));
    checkCudaErrors(cudaMemset(C, 0.0, M * K * sizeof(float)));
    solveMatrixMultiplication(A, B, C, M, N, K);
}

int main()
{
    execVecAdd();
    execMatrixMultiplication();
}