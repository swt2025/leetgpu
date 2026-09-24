#pragma once
#include <cuda_runtime.h>
#include <stddef.h>
#include <iostream>
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
void check(T result, char const* const func, const char* const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code = %d(%s) \"%s\"\n",
            file,
            line,
            static_cast<unsigned int>(result),
            cudaGetErrorName(result),
            func);
        exit(EXIT_FAILURE);
    }
}

void getDeviceProperties();

void init(float* A, int N);