#ifndef CUDAATOMIC_H
#define CUDAATOMIC_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sys/time.h>

/// Sets up atomic addition function for older architecture (les than
/// CUDA 6.0

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else

__device__ int myAtomicAdd(int* address, int val)
{
    int old = *address, assumed;

    do {
        assumed = old;
        old = atomicCAS(address, assumed, val + assumed);
    } while (assumed != old);

    return old;
}

__device__ double myAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    
    unsigned long long int old = *address_as_ull, assumed;

    do{ assumed = old;
        old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +__longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#endif // CUDAATOMIC_H