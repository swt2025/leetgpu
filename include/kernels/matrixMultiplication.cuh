#pragma once
#include <cuda_runtime.h>


/**
 * @breif 计算两个矩阵A(M,N)和B(N,K)的乘积C(M,K)
 */
__global__ void matrix_multiplication_kernel_0(const float* A, const float* B, float* C, int M, int N, int K);

extern "C" void solveMatrixMultiplication(const float* A, const float* B, float* C, int M, int N, int K);