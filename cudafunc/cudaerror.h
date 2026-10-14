#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

inline void __cuda_check_errors(
    const char *filename,
    const int line_number)
{
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("CUDA error %d at %s:%d: %s\n",
               (int)err, filename, line_number, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __cuda_safe_call(
    cudaError_t err,
    const char *filename,
    const int line_number)
{
    if (err != cudaSuccess) {
        printf("CUDA error %d at %s:%d: %s\n",
               (int)err, filename, line_number, cudaGetErrorString(err));
        exit(-1);
    }
}

#define CUDA_CHECK_ERROR() __cuda_check_errors(__FILE__, __LINE__)
#define CUDA_SAFE_CALL(err) __cuda_safe_call(err, __FILE__, __LINE__)
