cmake_minimum_required(VERSION 3.18)
project(LeetGPU LANGUAGES CUDA)

add_executable(matrixMultiplication main.cu src/helper.cu src/kernels/matrixMultiplication.cu)
target_include_directories(matrixMultiplication PRIVATE include)