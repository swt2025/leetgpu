#pragma once
#include <cuda_runtime.h>
#include "helper.cuh"
template <typename Func>
float benchmark_cuda_kernel(Func launch, int warmup = 10, int repeat = 100) {
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    checkCudaErrors(cudaDeviceSynchronize());
    float ms = 0;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&end_event));
    checkCudaErrors(cudaEventRecord(start_event));
    for (int i = 0; i < repeat; ++i) {
        launch();
    }
    checkCudaErrors(cudaEventRecord(end_event));
    checkCudaErrors(cudaEventSynchronize(end_event));
    checkCudaErrors(cudaEventElapsedTime(&ms, start_event, end_event));
    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(end_event));

    return ms / repeat;
}
