#include "helper.cuh"

void init(float* A, int N) {
    std::srand(0);
    for (int i = 0; i < N; ++i) {
        A[i] = std::rand() % 100;
    }
}

void getDeviceProperties() {
    // TO DO
}