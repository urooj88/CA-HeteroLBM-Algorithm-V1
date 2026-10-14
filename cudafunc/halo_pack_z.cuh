#pragma once

#include "../3dsetup/typedefs.h"

// ============================================================
// GPU pack all 19 directions for a fixed z-plane into buffer
// buffer layout: buf[l*(Nnx*Nny) + (j*Nnx + i)]
// ============================================================
template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void packZPlaneAllDirs(
    const VALUE_TYPE* __restrict__ f_post,
    VALUE_TYPE* __restrict__ buf,
    INDEX_TYPE Nnx, INDEX_TYPE Nny, INDEX_TYPE Nnz,
    INDEX_TYPE kplane)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t planeSize = (size_t)Nnx * (size_t)Nny;
    if (tid >= planeSize) return;

    INDEX_TYPE i = (INDEX_TYPE)(tid % (size_t)Nnx);
    INDEX_TYPE j = (INDEX_TYPE)(tid / (size_t)Nnx);

    size_t ixyz = (size_t)i +
                  (size_t)j * (size_t)Nnx +
                  (size_t)kplane * (size_t)Nnx * (size_t)Nny;

    size_t Ntotal = (size_t)Nnx * (size_t)Nny * (size_t)Nnz;

    #pragma unroll
    for (int l = 0; l < 19; l++) {
        buf[(size_t)l * planeSize + tid] = f_post[ixyz + (size_t)l * Ntotal];
    }
}

// ============================================================
// GPU unpack all 19 directions from buffer into a fixed z-plane
// ============================================================
template<typename VALUE_TYPE, typename INDEX_TYPE>
__global__ void unpackZPlaneAllDirs(
    const VALUE_TYPE* __restrict__ buf,
    VALUE_TYPE* __restrict__ f_post,
    INDEX_TYPE Nnx, INDEX_TYPE Nny, INDEX_TYPE Nnz,
    INDEX_TYPE kplane)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t planeSize = (size_t)Nnx * (size_t)Nny;
    if (tid >= planeSize) return;

    INDEX_TYPE i = (INDEX_TYPE)(tid % (size_t)Nnx);
    INDEX_TYPE j = (INDEX_TYPE)(tid / (size_t)Nnx);

    size_t ixyz = (size_t)i +
                  (size_t)j * (size_t)Nnx +
                  (size_t)kplane * (size_t)Nnx * (size_t)Nny;

    size_t Ntotal = (size_t)Nnx * (size_t)Nny * (size_t)Nnz;

    #pragma unroll
    for (int l = 0; l < 19; l++) {
        f_post[ixyz + (size_t)l * Ntotal] = buf[(size_t)l * planeSize + tid];
    }
}

// ============================================================
// Host helpers for CPU worker halo exchange
// These are plain inline functions so the same header supports
// both GPU ranks and the CPU rank in the heterogeneous version.
// ============================================================
template<typename VALUE_TYPE, typename INDEX_TYPE>
inline void packZPlaneAllDirs_host(
    const VALUE_TYPE* f_post,
    VALUE_TYPE* buf,
    INDEX_TYPE Nnx, INDEX_TYPE Nny, INDEX_TYPE Nnz,
    INDEX_TYPE kplane)
{
    const size_t planeSize = (size_t)Nnx * (size_t)Nny;
    const size_t Ntotal    = (size_t)Nnx * (size_t)Nny * (size_t)Nnz;

    for (INDEX_TYPE j = 0; j < Nny; ++j) {
        for (INDEX_TYPE i = 0; i < Nnx; ++i) {
            const size_t tid = (size_t)j * (size_t)Nnx + (size_t)i;
            const size_t ixyz = (size_t)i +
                                (size_t)j * (size_t)Nnx +
                                (size_t)kplane * (size_t)Nnx * (size_t)Nny;

            for (int l = 0; l < 19; ++l) {
                buf[(size_t)l * planeSize + tid] =
                    f_post[ixyz + (size_t)l * Ntotal];
            }
        }
    }
}

template<typename VALUE_TYPE, typename INDEX_TYPE>
inline void unpackZPlaneAllDirs_host(
    const VALUE_TYPE* buf,
    VALUE_TYPE* f_post,
    INDEX_TYPE Nnx, INDEX_TYPE Nny, INDEX_TYPE Nnz,
    INDEX_TYPE kplane)
{
    const size_t planeSize = (size_t)Nnx * (size_t)Nny;
    const size_t Ntotal    = (size_t)Nnx * (size_t)Nny * (size_t)Nnz;

    for (INDEX_TYPE j = 0; j < Nny; ++j) {
        for (INDEX_TYPE i = 0; i < Nnx; ++i) {
            const size_t tid = (size_t)j * (size_t)Nnx + (size_t)i;
            const size_t ixyz = (size_t)i +
                                (size_t)j * (size_t)Nnx +
                                (size_t)kplane * (size_t)Nnx * (size_t)Nny;

            for (int l = 0; l < 19; ++l) {
                f_post[ixyz + (size_t)l * Ntotal] =
                    buf[(size_t)l * planeSize + tid];
            }
        }
    }
}
