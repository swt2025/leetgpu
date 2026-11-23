#include "helper.cuh"

void init(float* A, int N) {
    std::srand(0);
    for (int i = 0; i < N; ++i) {
        A[i] = std::rand() % 100;
    }
}

void printfDeviceProperties() {
    cudaDeviceProp prop{};
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

    std::cout   << "GPU: " << prop.name << '\n'
                << "Compute Capability: " << prop.major << '.' << prop.minor << '\n'
                << "Global Memory(MiB): " << prop.totalGlobalMem << '\n'
                << "SM count: " << prop.multiProcessorCount << '\n'
                << "Threads per warp: " << prop.warpSize << '\n'
                << "Max threads per block: " << prop.maxThreadsPerBlock << '\n'
                << "Shared memory per block(KiB): " << prop.sharedMemPerBlock / 1024.0 << '\n';    
}