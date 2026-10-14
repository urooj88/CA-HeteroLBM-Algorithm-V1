#pragma once

#include <cmath>
#include <cstring>
#include <algorithm>
#include <omp.h>

#include "../3dsetup/typedefs.h"
#include "../3dbase/hetero_constants.h"

namespace hetero_host {

// ------------------------------------------------------------
// Basic helpers
// ------------------------------------------------------------
inline bool in_bounds(INDEX_TYPE i, INDEX_TYPE j, INDEX_TYPE k,
                      INDEX_TYPE Nnx, INDEX_TYPE Nny, INDEX_TYPE Nnz) {
    return (i >= 0 && i < Nnx &&
            j >= 0 && j < Nny &&
            k >= 0 && k < Nnz);
}

inline size_t scalar_idx(INDEX_TYPE i, INDEX_TYPE j, INDEX_TYPE k,
                         INDEX_TYPE Nnx, INDEX_TYPE Nny) {
    return hetero_const::scalar_index(i, j, k, Nnx, Nny);
}

inline size_t pdf_idx(INDEX_TYPE l, INDEX_TYPE i, INDEX_TYPE j, INDEX_TYPE k,
                      INDEX_TYPE Nnx, INDEX_TYPE Nny, INDEX_TYPE Nnz) {
    return hetero_const::pdf_index(l, i, j, k, Nnx, Nny, Nnz);
}

// ------------------------------------------------------------
// Initialization
// ------------------------------------------------------------
template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void initialiseDenVelBasic_host(
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *rho,
    VALUE_TYPE_T rhoinit,
    VALUE_TYPE_T *vx,
    VALUE_TYPE_T vxinit,
    VALUE_TYPE_T *vy,
    VALUE_TYPE_T vyinit,
    VALUE_TYPE_T *vz,
    VALUE_TYPE_T vzinit)
{
    const size_t Ntot = static_cast<size_t>(Nnx) *
                        static_cast<size_t>(Nny) *
                        static_cast<size_t>(Nnz);

    #pragma omp parallel for
    for (size_t idx = 0; idx < Ntot; ++idx) {
        rho[idx] = rhoinit;
        vx[idx]  = vxinit;
        vy[idx]  = vyinit;
        vz[idx]  = vzinit;
    }
}

template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void initialiseEDF_host(
    VALUE_TYPE_T *f,
    VALUE_TYPE_T *f_post,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *rho,
    VALUE_TYPE_T rhoinit,
    INDEX_TYPE_T simtype,
    VALUE_TYPE_T tau,
    VALUE_TYPE_T *vx,
    VALUE_TYPE_T *vy,
    VALUE_TYPE_T *vz)
{
    (void)tau;

    #pragma omp parallel for collapse(3)
    for (INDEX_TYPE_T k = 0; k < Nnz; ++k) {
        for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
            for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
                const size_t ixyz = scalar_idx(i, j, k, Nnx, Nny);
                const VALUE_TYPE_T U2 =
                    vx[ixyz] * vx[ixyz] +
                    vy[ixyz] * vy[ixyz] +
                    vz[ixyz] * vz[ixyz];

                for (INDEX_TYPE_T l = 0; l < hetero_const::QH; ++l) {
                    const VALUE_TYPE_T cu =
                        hetero_const::cx[l] * vx[ixyz] +
                        hetero_const::cy[l] * vy[ixyz] +
                        hetero_const::cz[l] * vz[ixyz];

                    VALUE_TYPE_T feq = VALUE_TYPE_T(0);
                    if (simtype == 0) {
                        feq = hetero_const::w[l] * rho[ixyz] *
                              (VALUE_TYPE_T(1.0) + VALUE_TYPE_T(3.0) * cu +
                               VALUE_TYPE_T(4.5) * cu * cu -
                               VALUE_TYPE_T(1.5) * U2);
                    } else {
                        feq = hetero_const::w[l] *
                              (rho[ixyz] + VALUE_TYPE_T(3.0) * rhoinit * cu +
                               VALUE_TYPE_T(4.5) * rhoinit * cu * cu -
                               VALUE_TYPE_T(1.5) * rhoinit * U2);
                    }

                    const size_t idx = pdf_idx(l, i, j, k, Nnx, Nny, Nnz);
                    f[idx] = feq;
                    f_post[idx] = feq;
                }
            }
        }
    }
}

// ------------------------------------------------------------
// BGK collision
// ------------------------------------------------------------
template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void collBGK_host(
    VALUE_TYPE_T *f,
    VALUE_TYPE_T *f_post,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *rho,
    VALUE_TYPE_T rhoinit,
    INDEX_TYPE_T simtype,
    VALUE_TYPE_T tau,
    VALUE_TYPE_T *vx,
    VALUE_TYPE_T *vy,
    VALUE_TYPE_T *vz)
{
    #pragma omp parallel for collapse(3)
    for (INDEX_TYPE_T k = 0; k < Nnz; ++k) {
        for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
            for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
                const size_t ixyz = scalar_idx(i, j, k, Nnx, Nny);
                const VALUE_TYPE_T U2 =
                    vx[ixyz] * vx[ixyz] +
                    vy[ixyz] * vy[ixyz] +
                    vz[ixyz] * vz[ixyz];

                for (INDEX_TYPE_T l = 0; l < hetero_const::QH; ++l) {
                    const size_t index = pdf_idx(l, i, j, k, Nnx, Nny, Nnz);

                    const VALUE_TYPE_T cu =
                        hetero_const::cx[l] * vx[ixyz] +
                        hetero_const::cy[l] * vy[ixyz] +
                        hetero_const::cz[l] * vz[ixyz];

                    VALUE_TYPE_T feq = VALUE_TYPE_T(0);
                    if (simtype == 0) {
                        feq = hetero_const::w[l] * rho[ixyz] *
                              (VALUE_TYPE_T(1.0) + VALUE_TYPE_T(3.0) * cu +
                               VALUE_TYPE_T(4.5) * cu * cu -
                               VALUE_TYPE_T(1.5) * U2);
                    } else {
                        feq = hetero_const::w[l] *
                              (rho[ixyz] + VALUE_TYPE_T(3.0) * rhoinit * cu +
                               VALUE_TYPE_T(4.5) * rhoinit * cu * cu -
                               VALUE_TYPE_T(1.5) * rhoinit * U2);
                    }

                    f_post[index] = f[index] - (f[index] - feq) / tau;
                }
            }
        }
    }
}

// ------------------------------------------------------------
// MRT collision fallback
// For your current input Relaxtion_type: 0, this is not used.
// ------------------------------------------------------------
template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void collMRT_host(
    VALUE_TYPE_T *f,
    VALUE_TYPE_T *f_post,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *rho,
    VALUE_TYPE_T rhoinit,
    INDEX_TYPE_T simtype,
    VALUE_TYPE_T tau_fallback,
    VALUE_TYPE_T *vx,
    VALUE_TYPE_T *vy,
    VALUE_TYPE_T *vz)
{
    collBGK_host<VALUE_TYPE_T, INDEX_TYPE_T>(
        f, f_post, Nnx, Nny, Nnz, rho, rhoinit,
        simtype, tau_fallback, vx, vy, vz
    );
}

// ------------------------------------------------------------
// Periodic
// ------------------------------------------------------------
template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void periodicFb_host(
    VALUE_TYPE_T *f_post,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz)
{
    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
        for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
            {
                const INDEX_TYPE_T k = 0;
                const INDEX_TYPE_T ks = Nnz - 2;

                f_post[pdf_idx(5,  i, j, k,  Nnx, Nny, Nnz)] = f_post[pdf_idx(5,  i, j, ks, Nnx, Nny, Nnz)];
                f_post[pdf_idx(9,  i, j, k,  Nnx, Nny, Nnz)] = f_post[pdf_idx(9,  i, j, ks, Nnx, Nny, Nnz)];
                f_post[pdf_idx(11, i, j, k,  Nnx, Nny, Nnz)] = f_post[pdf_idx(11, i, j, ks, Nnx, Nny, Nnz)];
                f_post[pdf_idx(16, i, j, k,  Nnx, Nny, Nnz)] = f_post[pdf_idx(16, i, j, ks, Nnx, Nny, Nnz)];
                f_post[pdf_idx(18, i, j, k,  Nnx, Nny, Nnz)] = f_post[pdf_idx(18, i, j, ks, Nnx, Nny, Nnz)];
            }

            {
                const INDEX_TYPE_T k = Nnz - 1;
                const INDEX_TYPE_T ks = 1;

                f_post[pdf_idx(6,  i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(6,  i, j, ks, Nnx, Nny, Nnz)];
                f_post[pdf_idx(10, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(10, i, j, ks, Nnx, Nny, Nnz)];
                f_post[pdf_idx(12, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(12, i, j, ks, Nnx, Nny, Nnz)];
                f_post[pdf_idx(15, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(15, i, j, ks, Nnx, Nny, Nnz)];
                f_post[pdf_idx(17, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(17, i, j, ks, Nnx, Nny, Nnz)];
            }
        }
    }
}

template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void periodicLr_host(
    VALUE_TYPE_T *f_post,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz)
{
    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE_T k = 0; k < Nnz; ++k) {
        for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
            {
                const INDEX_TYPE_T i = 0;
                const INDEX_TYPE_T is = Nnx - 2;

                f_post[pdf_idx(1,  i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(1,  is, j, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(7,  i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(7,  is, j, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(9,  i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(9,  is, j, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(13, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(13, is, j, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(15, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(15, is, j, k, Nnx, Nny, Nnz)];
            }

            {
                const INDEX_TYPE_T i = Nnx - 1;
                const INDEX_TYPE_T is = 1;

                f_post[pdf_idx(2,  i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(2,  is, j, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(8,  i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(8,  is, j, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(10, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(10, is, j, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(14, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(14, is, j, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(16, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(16, is, j, k, Nnx, Nny, Nnz)];
            }
        }
    }
}

template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void periodicTb_host(
    VALUE_TYPE_T *f_post,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz)
{
    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE_T k = 0; k < Nnz; ++k) {
        for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
            {
                const INDEX_TYPE_T j = 0;
                const INDEX_TYPE_T js = Nny - 2;

                f_post[pdf_idx(3,  i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(3,  i, js, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(7,  i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(7,  i, js, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(11, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(11, i, js, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(14, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(14, i, js, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(17, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(17, i, js, k, Nnx, Nny, Nnz)];
            }

            {
                const INDEX_TYPE_T j = Nny - 1;
                const INDEX_TYPE_T js = 1;

                f_post[pdf_idx(4,  i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(4,  i, js, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(8,  i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(8,  i, js, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(12, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(12, i, js, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(13, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(13, i, js, k, Nnx, Nny, Nnz)];
                f_post[pdf_idx(18, i, j, k, Nnx, Nny, Nnz)] = f_post[pdf_idx(18, i, js, k, Nnx, Nny, Nnz)];
            }
        }
    }
}

// ------------------------------------------------------------
// Streaming
// ------------------------------------------------------------
template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void streaming_host(
    VALUE_TYPE_T *f,
    VALUE_TYPE_T *f_post,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz)
{
    #pragma omp parallel for collapse(3)
    for (INDEX_TYPE_T k = 0; k < Nnz; ++k) {
        for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
            for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
                for (INDEX_TYPE_T l = 0; l < hetero_const::QH; ++l) {
                    const INDEX_TYPE_T id = i - hetero_const::cx[l];
                    const INDEX_TYPE_T jd = j - hetero_const::cy[l];
                    const INDEX_TYPE_T kd = k - hetero_const::cz[l];

                    if (in_bounds(id, jd, kd, Nnx, Nny, Nnz)) {
                        const size_t dst = pdf_idx(l, i,  j,  k,  Nnx, Nny, Nnz);
                        const size_t src = pdf_idx(l, id, jd, kd, Nnx, Nny, Nnz);
                        f[dst] = f_post[src];
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------
// Density and velocity
// ------------------------------------------------------------
template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void denVel_host(
    VALUE_TYPE_T *f,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *rho,
    VALUE_TYPE_T rhoinit,
    INDEX_TYPE_T simtype,
    VALUE_TYPE_T *vx,
    VALUE_TYPE_T *vy,
    VALUE_TYPE_T *vz)
{
    #pragma omp parallel for collapse(3)
    for (INDEX_TYPE_T k = 0; k < Nnz; ++k) {
        for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
            for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
                const size_t ixyz = scalar_idx(i, j, k, Nnx, Nny);

                const VALUE_TYPE_T f0  = f[pdf_idx(0,  i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f1  = f[pdf_idx(1,  i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f2  = f[pdf_idx(2,  i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f3  = f[pdf_idx(3,  i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f4  = f[pdf_idx(4,  i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f5  = f[pdf_idx(5,  i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f6  = f[pdf_idx(6,  i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f7  = f[pdf_idx(7,  i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f8  = f[pdf_idx(8,  i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f9  = f[pdf_idx(9,  i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f10 = f[pdf_idx(10, i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f11 = f[pdf_idx(11, i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f12 = f[pdf_idx(12, i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f13 = f[pdf_idx(13, i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f14 = f[pdf_idx(14, i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f15 = f[pdf_idx(15, i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f16 = f[pdf_idx(16, i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f17 = f[pdf_idx(17, i,j,k,Nnx,Nny,Nnz)];
                const VALUE_TYPE_T f18 = f[pdf_idx(18, i,j,k,Nnx,Nny,Nnz)];

                rho[ixyz] = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 +
                            f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18;

                if (simtype == 0) {
                    vx[ixyz] = (f1 + f7 + f9 + f13 + f15 -
                                f2 - f8 - f10 - f14 - f16) / rho[ixyz];

                    vy[ixyz] = (f3 + f7 + f11 + f14 + f17 -
                                f4 - f8 - f12 - f13 - f18) / rho[ixyz];

                    vz[ixyz] = (f5 + f9 + f11 + f16 + f18 -
                                f6 - f10 - f12 - f15 - f17) / rho[ixyz];
                } else {
                    vx[ixyz] = (f1 + f7 + f9 + f13 + f15 -
                                f2 - f8 - f10 - f14 - f16) / rhoinit;

                    vy[ixyz] = (f3 + f7 + f11 + f14 + f17 -
                                f4 - f8 - f12 - f13 - f18) / rhoinit;

                    vz[ixyz] = (f5 + f9 + f11 + f16 + f18 -
                                f6 - f10 - f12 - f15 - f17) / rhoinit;
                }
            }
        }
    }
}

// ------------------------------------------------------------
// Error
// ------------------------------------------------------------
template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void error_host(
    VALUE_TYPE_T *e1,
    VALUE_TYPE_T *e2,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *vxerr,
    VALUE_TYPE_T *vyerr,
    VALUE_TYPE_T *vzerr,
    VALUE_TYPE_T *vx,
    VALUE_TYPE_T *vy,
    VALUE_TYPE_T *vz)
{
    #pragma omp parallel for collapse(3)
    for (INDEX_TYPE_T k = 0; k < Nnz; ++k) {
        for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
            for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
                const size_t idx = scalar_idx(i, j, k, Nnx, Nny);

                const VALUE_TYPE_T dvx = vx[idx] - vxerr[idx];
                const VALUE_TYPE_T dvy = vy[idx] - vyerr[idx];
                const VALUE_TYPE_T dvz = vz[idx] - vzerr[idx];

                e1[idx] = std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
                e2[idx] = std::sqrt(vx[idx] * vx[idx] +
                                    vy[idx] * vy[idx] +
                                    vz[idx] * vz[idx]);

                vxerr[idx] = vx[idx];
                vyerr[idx] = vy[idx];
                vzerr[idx] = vz[idx];
            }
        }
    }
}

template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void interior_error_sums_host(
    VALUE_TYPE_T &sum1,
    VALUE_TYPE_T &sum2,
    VALUE_TYPE_T *e1,
    VALUE_TYPE_T *e2,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T startX,
    INDEX_TYPE_T startY,
    INDEX_TYPE_T startZ,
    INDEX_TYPE_T nxI,
    INDEX_TYPE_T nyI,
    INDEX_TYPE_T nzI)
{
    sum1 = VALUE_TYPE_T(0);
    sum2 = VALUE_TYPE_T(0);

    #pragma omp parallel for collapse(3) reduction(+:sum1,sum2)
    for (INDEX_TYPE_T kk = 0; kk < nzI; ++kk) {
        for (INDEX_TYPE_T jj = 0; jj < nyI; ++jj) {
            for (INDEX_TYPE_T ii = 0; ii < nxI; ++ii) {
                const INDEX_TYPE_T i = startX + ii;
                const INDEX_TYPE_T j = startY + jj;
                const INDEX_TYPE_T k = startZ + kk;
                const size_t idx = scalar_idx(i, j, k, Nnx, Nny);
                sum1 += e1[idx];
                sum2 += e2[idx];
            }
        }
    }
}

// ------------------------------------------------------------
// Bounceback helper
// ------------------------------------------------------------
template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
inline void apply_bounce_pair(
    VALUE_TYPE_T *f,
    VALUE_TYPE_T *f_post,
    VALUE_TYPE_T rho,
    VALUE_TYPE_T u, VALUE_TYPE_T v, VALUE_TYPE_T wv,
    INDEX_TYPE_T l_to,
    INDEX_TYPE_T l_from,
    INDEX_TYPE_T i, INDEX_TYPE_T j, INDEX_TYPE_T k,
    INDEX_TYPE_T Nnx, INDEX_TYPE_T Nny, INDEX_TYPE_T Nnz)
{
    const size_t to   = pdf_idx(l_to,   i, j, k, Nnx, Nny, Nnz);
    const size_t from = pdf_idx(l_from, i, j, k, Nnx, Nny, Nnz);

    f[to] = f_post[from] +
            VALUE_TYPE_T(6.0) * rho * hetero_const::w[l_to] *
            (hetero_const::cx[l_to] * u +
             hetero_const::cy[l_to] * v +
             hetero_const::cz[l_to] * wv);
}

// ------------------------------------------------------------
// Bounceback walls
// ------------------------------------------------------------
template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void halfwayBouncebackBack_host(
    VALUE_TYPE_T *f,
    VALUE_TYPE_T *f_post,
    VALUE_TYPE_T latvxback,
    VALUE_TYPE_T latvyback,
    VALUE_TYPE_T latvzback,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *rho)
{
    const INDEX_TYPE_T k = 0;

    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
        for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
            const size_t idx = scalar_idx(i, j, k, Nnx, Nny);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxback, latvyback, latvzback, 5,  6,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxback, latvyback, latvzback, 9,  10, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxback, latvyback, latvzback, 11, 12, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxback, latvyback, latvzback, 16, 15, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxback, latvyback, latvzback, 18, 17, i, j, k, Nnx, Nny, Nnz);
        }
    }
}

template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void halfwayBouncebackFront_host(
    VALUE_TYPE_T *f,
    VALUE_TYPE_T *f_post,
    VALUE_TYPE_T latvxfront,
    VALUE_TYPE_T latvyfront,
    VALUE_TYPE_T latvzfront,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *rho)
{
    const INDEX_TYPE_T k = Nnz - 1;

    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
        for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
            const size_t idx = scalar_idx(i, j, k, Nnx, Nny);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxfront, latvyfront, latvzfront, 6,  5,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxfront, latvyfront, latvzfront, 10, 9,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxfront, latvyfront, latvzfront, 12, 11, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxfront, latvyfront, latvzfront, 15, 16, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxfront, latvyfront, latvzfront, 17, 18, i, j, k, Nnx, Nny, Nnz);
        }
    }
}

template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void halfwayBouncebackBottom_host(
    VALUE_TYPE_T *f,
    VALUE_TYPE_T *f_post,
    VALUE_TYPE_T latvxbottom,
    VALUE_TYPE_T latvybottom,
    VALUE_TYPE_T latvzbottom,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *rho)
{
    const INDEX_TYPE_T j = 0;

    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE_T k = 0; k < Nnz; ++k) {
        for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
            const size_t idx = scalar_idx(i, j, k, Nnx, Nny);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxbottom, latvybottom, latvzbottom, 3,  4,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxbottom, latvybottom, latvzbottom, 7,  8,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxbottom, latvybottom, latvzbottom, 11, 12, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxbottom, latvybottom, latvzbottom, 14, 13, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxbottom, latvybottom, latvzbottom, 17, 18, i, j, k, Nnx, Nny, Nnz);
        }
    }
}

template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void halfwayBouncebackTop_host(
    VALUE_TYPE_T *f,
    VALUE_TYPE_T *f_post,
    VALUE_TYPE_T latvxtop,
    VALUE_TYPE_T latvytop,
    VALUE_TYPE_T latvztop,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *rho)
{
    const INDEX_TYPE_T j = Nny - 1;

    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE_T k = 0; k < Nnz; ++k) {
        for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
            const size_t idx = scalar_idx(i, j, k, Nnx, Nny);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxtop, latvytop, latvztop, 4,  3,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxtop, latvytop, latvztop, 8,  7,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxtop, latvytop, latvztop, 12, 11, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxtop, latvytop, latvztop, 13, 14, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxtop, latvytop, latvztop, 18, 17, i, j, k, Nnx, Nny, Nnz);
        }
    }
}

template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void halfwayBouncebackLeft_host(
    VALUE_TYPE_T *f,
    VALUE_TYPE_T *f_post,
    VALUE_TYPE_T latvxleft,
    VALUE_TYPE_T latvyleft,
    VALUE_TYPE_T latvzleft,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *rho)
{
    const INDEX_TYPE_T i = 0;

    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE_T k = 0; k < Nnz; ++k) {
        for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
            const size_t idx = scalar_idx(i, j, k, Nnx, Nny);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxleft, latvyleft, latvzleft, 1,  2,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxleft, latvyleft, latvzleft, 7,  8,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxleft, latvyleft, latvzleft, 9,  10, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxleft, latvyleft, latvzleft, 13, 14, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxleft, latvyleft, latvzleft, 15, 16, i, j, k, Nnx, Nny, Nnz);
        }
    }
}

template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void halfwayBouncebackRight_host(
    VALUE_TYPE_T *f,
    VALUE_TYPE_T *f_post,
    VALUE_TYPE_T latvxright,
    VALUE_TYPE_T latvyright,
    VALUE_TYPE_T latvzright,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    VALUE_TYPE_T *rho)
{
    const INDEX_TYPE_T i = Nnx - 1;

    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE_T k = 0; k < Nnz; ++k) {
        for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
            const size_t idx = scalar_idx(i, j, k, Nnx, Nny);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxright, latvyright, latvzright, 2,  1,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxright, latvyright, latvzright, 8,  7,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxright, latvyright, latvzright, 10, 9,  i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxright, latvyright, latvzright, 14, 13, i, j, k, Nnx, Nny, Nnz);
            apply_bounce_pair<VALUE_TYPE_T, INDEX_TYPE_T>(f, f_post, rho[idx], latvxright, latvyright, latvzright, 16, 15, i, j, k, Nnx, Nny, Nnz);
        }
    }
}

// ------------------------------------------------------------
// Halo pack / unpack on host
// ------------------------------------------------------------
template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void packZPlaneAllDirs_host(
    const VALUE_TYPE_T* f_post,
    VALUE_TYPE_T* buf,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    INDEX_TYPE_T kplane)
{
    const size_t planeSize = static_cast<size_t>(Nnx) * static_cast<size_t>(Nny);
    const size_t Ntotal = static_cast<size_t>(Nnx) * static_cast<size_t>(Nny) * static_cast<size_t>(Nnz);

    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
        for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
            const size_t tid = static_cast<size_t>(j) * static_cast<size_t>(Nnx) + static_cast<size_t>(i);
            const size_t ixyz = scalar_idx(i, j, kplane, Nnx, Nny);
            for (int l = 0; l < 19; ++l) {
                buf[static_cast<size_t>(l) * planeSize + tid] =
                    f_post[ixyz + static_cast<size_t>(l) * Ntotal];
            }
        }
    }
}

template<typename VALUE_TYPE_T, typename INDEX_TYPE_T>
void unpackZPlaneAllDirs_host(
    const VALUE_TYPE_T* buf,
    VALUE_TYPE_T* f_post,
    INDEX_TYPE_T Nnx,
    INDEX_TYPE_T Nny,
    INDEX_TYPE_T Nnz,
    INDEX_TYPE_T kplane)
{
    const size_t planeSize = static_cast<size_t>(Nnx) * static_cast<size_t>(Nny);
    const size_t Ntotal = static_cast<size_t>(Nnx) * static_cast<size_t>(Nny) * static_cast<size_t>(Nnz);

    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE_T j = 0; j < Nny; ++j) {
        for (INDEX_TYPE_T i = 0; i < Nnx; ++i) {
            const size_t tid = static_cast<size_t>(j) * static_cast<size_t>(Nnx) + static_cast<size_t>(i);
            const size_t ixyz = scalar_idx(i, j, kplane, Nnx, Nny);
            for (int l = 0; l < 19; ++l) {
                f_post[ixyz + static_cast<size_t>(l) * Ntotal] =
                    buf[static_cast<size_t>(l) * planeSize + tid];
            }
        }
    }
}

} // namespace hetero_host
