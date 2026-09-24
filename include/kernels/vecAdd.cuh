#pragma once
#include <cuda_runtime.h>

__global__ void vector_add_0(const float* A, const float* B, float* C, int N);

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N);