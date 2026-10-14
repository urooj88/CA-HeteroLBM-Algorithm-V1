#ifndef HETERO_OPENH_CPU_BACKEND_H
#define HETERO_OPENH_CPU_BACKEND_H

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <omp.h>

#include "hetero_openh_runtime.h"

// host-side copies of device constants
static int h_Q = 19;
static int h_cx[19];
static int h_cy[19];
static int h_cz[19];
static double h_w[19];
static int h_D[19];
static double h_s[19];

static inline void set_host_lbm_constants(
    const int cx_in[19],
    const int cy_in[19],
    const int cz_in[19],
    const double w_in[19],
    const int D_in[19],
    const double s_in[19]
) {
    h_Q = 19;
    for (int q = 0; q < 19; q++) {
        h_cx[q] = cx_in[q];
        h_cy[q] = cy_in[q];
        h_cz[q] = cz_in[q];
        h_w[q]  = w_in[q];
        h_D[q]  = D_in[q];
        h_s[q]  = s_in[q];
    }
}

static inline size_t cpu_scalar_index(
    int i, int j, int k,
    int Nnx, int Nny
) {
    return (size_t)i + (size_t)j * (size_t)Nnx + (size_t)k * (size_t)Nnx * (size_t)Nny;
}

static inline size_t cpu_qindex(
    int q, size_t scalar_idx, size_t Ntotal
) {
    return scalar_idx + (size_t)q * Ntotal;
}

static inline void cpu_initialise_denvel_basic_host(
    WorkerState& ws,
    double rhoinit,
    double vxinit,
    double vyinit,
    double vzinit
) {
    #pragma omp parallel for
    for (long long idx = 0; idx < (long long)ws.geom.Ntotal; idx++) {
        ws.rho[idx] = rhoinit;
        ws.vx[idx]  = vxinit;
        ws.vy[idx]  = vyinit;
        ws.vz[idx]  = vzinit;
        ws.vxerr[idx] = 0.0;
        ws.vyerr[idx] = 0.0;
        ws.vzerr[idx] = 0.0;
        ws.e1[idx] = 0.0;
        ws.e2[idx] = 0.0;
    }
}

static inline void cpu_initialise_edf_placeholder(
    WorkerState& ws
) {
    #pragma omp parallel for
    for (long long idx = 0; idx < (long long)ws.geom.Ntotal; idx++) {
        const double rho = ws.rho[idx];
        const double ux  = ws.vx[idx];
        const double uy  = ws.vy[idx];
        const double uz  = ws.vz[idx];
        const double u2  = ux * ux + uy * uy + uz * uz;

        for (int q = 0; q < h_Q; q++) {
            const double cu = h_cx[q] * ux + h_cy[q] * uy + h_cz[q] * uz;
            const double feq = h_w[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
            ws.f[cpu_qindex(q, (size_t)idx, ws.geom.Ntotal)] = feq;
            ws.f_post[cpu_qindex(q, (size_t)idx, ws.geom.Ntotal)] = feq;
        }
    }
}

static inline void cpu_collide_bgk(
    WorkerState& ws,
    double rhoinit,
    int simtype,
    double tau
) {
    const int Nnx = ws.geom.Nnx;
    const int Nny = ws.geom.Nny;
    const int Nnz = ws.geom.Nnz;
    const size_t Ntotal = ws.geom.Ntotal;

    #pragma omp parallel for collapse(3)
    for (int k = 0; k < Nnz; k++) {
        for (int j = 0; j < Nny; j++) {
            for (int i = 0; i < Nnx; i++) {
                const size_t ixyz = cpu_scalar_index(i, j, k, Nnx, Nny);

                for (int q = 0; q < h_Q; q++) {
                    const size_t index = cpu_qindex(q, ixyz, Ntotal);

                    const double cu = h_cx[q] * ws.vx[ixyz] + h_cy[q] * ws.vy[ixyz] + h_cz[q] * ws.vz[ixyz];
                    const double u2 = ws.vx[ixyz] * ws.vx[ixyz] +
                                      ws.vy[ixyz] * ws.vy[ixyz] +
                                      ws.vz[ixyz] * ws.vz[ixyz];

                    double feq = 0.0;
                    if (simtype == 0) {
                        feq = h_w[q] * ws.rho[ixyz] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
                    } else {
                        feq = h_w[q] * (ws.rho[ixyz] + 3.0 * rhoinit * cu + 4.5 * rhoinit * cu * cu - 1.5 * rhoinit * u2);
                    }

                    ws.f_post[index] = ws.f[index] - (ws.f[index] - feq) / tau;
                }
            }
        }
    }
}

static inline void cpu_apply_periodic_x(
    WorkerState& ws
) {
    const int Nnx = ws.geom.Nnx;
    const int Nny = ws.geom.Nny;
    const int Nnz = ws.geom.Nnz;
    const size_t Ntotal = ws.geom.Ntotal;

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < Nnz; k++) {
        for (int j = 0; j < Nny; j++) {
            const size_t left  = cpu_scalar_index(0,      j, k, Nnx, Nny);
            const size_t right = cpu_scalar_index(Nnx-1,  j, k, Nnx, Nny);
            const size_t srcL  = cpu_scalar_index(Nnx-2,  j, k, Nnx, Nny);
            const size_t srcR  = cpu_scalar_index(1,      j, k, Nnx, Nny);

            for (int q = 0; q < h_Q; q++) {
                if (h_cx[q] > 0) {
                    ws.f_post[cpu_qindex(q, left, Ntotal)] = ws.f_post[cpu_qindex(q, srcL, Ntotal)];
                }
                if (h_cx[q] < 0) {
                    ws.f_post[cpu_qindex(q, right, Ntotal)] = ws.f_post[cpu_qindex(q, srcR, Ntotal)];
                }
            }
        }
    }
}

static inline void cpu_apply_periodic_y(
    WorkerState& ws
) {
    const int Nnx = ws.geom.Nnx;
    const int Nny = ws.geom.Nny;
    const int Nnz = ws.geom.Nnz;
    const size_t Ntotal = ws.geom.Ntotal;

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < Nnz; k++) {
        for (int i = 0; i < Nnx; i++) {
            const size_t bottom = cpu_scalar_index(i, 0,      k, Nnx, Nny);
            const size_t top    = cpu_scalar_index(i, Nny-1,  k, Nnx, Nny);
            const size_t srcB   = cpu_scalar_index(i, Nny-2,  k, Nnx, Nny);
            const size_t srcT   = cpu_scalar_index(i, 1,      k, Nnx, Nny);

            for (int q = 0; q < h_Q; q++) {
                if (h_cy[q] > 0) {
                    ws.f_post[cpu_qindex(q, bottom, Ntotal)] = ws.f_post[cpu_qindex(q, srcB, Ntotal)];
                }
                if (h_cy[q] < 0) {
                    ws.f_post[cpu_qindex(q, top, Ntotal)] = ws.f_post[cpu_qindex(q, srcT, Ntotal)];
                }
            }
        }
    }
}

static inline void cpu_streaming(
    WorkerState& ws
) {
    const int Nnx = ws.geom.Nnx;
    const int Nny = ws.geom.Nny;
    const int Nnz = ws.geom.Nnz;
    const size_t Ntotal = ws.geom.Ntotal;

    #pragma omp parallel for collapse(3)
    for (int k = 0; k < Nnz; k++) {
        for (int j = 0; j < Nny; j++) {
            for (int i = 0; i < Nnx; i++) {
                const size_t ixyz = cpu_scalar_index(i, j, k, Nnx, Nny);

                for (int q = 0; q < h_Q; q++) {
                    const int id = i - h_cx[q];
                    const int jd = j - h_cy[q];
                    const int kd = k - h_cz[q];

                    if (id >= 0 && id < Nnx &&
                        jd >= 0 && jd < Nny &&
                        kd >= 0 && kd < Nnz) {
                        const size_t src = cpu_scalar_index(id, jd, kd, Nnx, Nny);
                        ws.f[cpu_qindex(q, ixyz, Ntotal)] = ws.f_post[cpu_qindex(q, src, Ntotal)];
                    }
                }
            }
        }
    }
}

static inline void cpu_apply_bounceback_x(
    WorkerState& ws,
    double latvxleft, double latvyleft, double latvzleft,
    double latvxright, double latvyright, double latvzright
) {
    const int Nnx = ws.geom.Nnx;
    const int Nny = ws.geom.Nny;
    const int Nnz = ws.geom.Nnz;
    const size_t Ntotal = ws.geom.Ntotal;

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < Nnz; k++) {
        for (int j = 0; j < Nny; j++) {
            {
                const int i = 0;
                const size_t idx = cpu_scalar_index(i, j, k, Nnx, Nny);
                const double rho = ws.rho[idx];

                ws.f[cpu_qindex(1, idx, Ntotal)]  = ws.f_post[cpu_qindex(2, idx, Ntotal)]  + 6.0 * rho * h_w[1]  * (h_cx[1]  * latvxleft  + h_cy[1]  * latvyleft  + h_cz[1]  * latvzleft);
                ws.f[cpu_qindex(7, idx, Ntotal)]  = ws.f_post[cpu_qindex(8, idx, Ntotal)]  + 6.0 * rho * h_w[7]  * (h_cx[7]  * latvxleft  + h_cy[7]  * latvyleft  + h_cz[7]  * latvzleft);
                ws.f[cpu_qindex(9, idx, Ntotal)]  = ws.f_post[cpu_qindex(10, idx, Ntotal)] + 6.0 * rho * h_w[9]  * (h_cx[9]  * latvxleft  + h_cy[9]  * latvyleft  + h_cz[9]  * latvzleft);
                ws.f[cpu_qindex(13, idx, Ntotal)] = ws.f_post[cpu_qindex(14, idx, Ntotal)] + 6.0 * rho * h_w[13] * (h_cx[13] * latvxleft  + h_cy[13] * latvyleft  + h_cz[13] * latvzleft);
                ws.f[cpu_qindex(15, idx, Ntotal)] = ws.f_post[cpu_qindex(16, idx, Ntotal)] + 6.0 * rho * h_w[15] * (h_cx[15] * latvxleft  + h_cy[15] * latvyleft  + h_cz[15] * latvzleft);
            }

            {
                const int i = Nnx - 1;
                const size_t idx = cpu_scalar_index(i, j, k, Nnx, Nny);
                const double rho = ws.rho[idx];

                ws.f[cpu_qindex(2, idx, Ntotal)]  = ws.f_post[cpu_qindex(1, idx, Ntotal)]  + 6.0 * rho * h_w[2]  * (h_cx[2]  * latvxright + h_cy[2]  * latvyright + h_cz[2]  * latvzright);
                ws.f[cpu_qindex(8, idx, Ntotal)]  = ws.f_post[cpu_qindex(7, idx, Ntotal)]  + 6.0 * rho * h_w[8]  * (h_cx[8]  * latvxright + h_cy[8]  * latvyright + h_cz[8]  * latvzright);
                ws.f[cpu_qindex(10, idx, Ntotal)] = ws.f_post[cpu_qindex(9, idx, Ntotal)]  + 6.0 * rho * h_w[10] * (h_cx[10] * latvxright + h_cy[10] * latvyright + h_cz[10] * latvzright);
                ws.f[cpu_qindex(14, idx, Ntotal)] = ws.f_post[cpu_qindex(13, idx, Ntotal)] + 6.0 * rho * h_w[14] * (h_cx[14] * latvxright + h_cy[14] * latvyright + h_cz[14] * latvzright);
                ws.f[cpu_qindex(16, idx, Ntotal)] = ws.f_post[cpu_qindex(15, idx, Ntotal)] + 6.0 * rho * h_w[16] * (h_cx[16] * latvxright + h_cy[16] * latvyright + h_cz[16] * latvzright);
            }
        }
    }
}

static inline void cpu_apply_bounceback_y(
    WorkerState& ws,
    double latvxbottom, double latvybottom, double latvzbottom,
    double latvxtop, double latvytop, double latvztop
) {
    const int Nnx = ws.geom.Nnx;
    const int Nny = ws.geom.Nny;
    const int Nnz = ws.geom.Nnz;
    const size_t Ntotal = ws.geom.Ntotal;

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < Nnz; k++) {
        for (int i = 0; i < Nnx; i++) {
            {
                const int j = 0;
                const size_t idx = cpu_scalar_index(i, j, k, Nnx, Nny);
                const double rho = ws.rho[idx];

                ws.f[cpu_qindex(3, idx, Ntotal)]  = ws.f_post[cpu_qindex(4, idx, Ntotal)]  + 6.0 * rho * h_w[3]  * (h_cx[3]  * latvxbottom + h_cy[3]  * latvybottom + h_cz[3]  * latvzbottom);
                ws.f[cpu_qindex(7, idx, Ntotal)]  = ws.f_post[cpu_qindex(8, idx, Ntotal)]  + 6.0 * rho * h_w[7]  * (h_cx[7]  * latvxbottom + h_cy[7]  * latvybottom + h_cz[7]  * latvzbottom);
                ws.f[cpu_qindex(11, idx, Ntotal)] = ws.f_post[cpu_qindex(12, idx, Ntotal)] + 6.0 * rho * h_w[11] * (h_cx[11] * latvxbottom + h_cy[11] * latvybottom + h_cz[11] * latvzbottom);
                ws.f[cpu_qindex(14, idx, Ntotal)] = ws.f_post[cpu_qindex(13, idx, Ntotal)] + 6.0 * rho * h_w[14] * (h_cx[14] * latvxbottom + h_cy[14] * latvybottom + h_cz[14] * latvzbottom);
                ws.f[cpu_qindex(17, idx, Ntotal)] = ws.f_post[cpu_qindex(18, idx, Ntotal)] + 6.0 * rho * h_w[17] * (h_cx[17] * latvxbottom + h_cy[17] * latvybottom + h_cz[17] * latvzbottom);
            }

            {
                const int j = Nny - 1;
                const size_t idx = cpu_scalar_index(i, j, k, Nnx, Nny);
                const double rho = ws.rho[idx];

                ws.f[cpu_qindex(4, idx, Ntotal)]  = ws.f_post[cpu_qindex(3, idx, Ntotal)]  + 6.0 * rho * h_w[4]  * (h_cx[4]  * latvxtop + h_cy[4]  * latvytop + h_cz[4]  * latvztop);
                ws.f[cpu_qindex(8, idx, Ntotal)]  = ws.f_post[cpu_qindex(7, idx, Ntotal)]  + 6.0 * rho * h_w[8]  * (h_cx[8]  * latvxtop + h_cy[8]  * latvytop + h_cz[8]  * latvztop);
                ws.f[cpu_qindex(12, idx, Ntotal)] = ws.f_post[cpu_qindex(11, idx, Ntotal)] + 6.0 * rho * h_w[12] * (h_cx[12] * latvxtop + h_cy[12] * latvytop + h_cz[12] * latvztop);
                ws.f[cpu_qindex(13, idx, Ntotal)] = ws.f_post[cpu_qindex(14, idx, Ntotal)] + 6.0 * rho * h_w[13] * (h_cx[13] * latvxtop + h_cy[13] * latvytop + h_cz[13] * latvztop);
                ws.f[cpu_qindex(18, idx, Ntotal)] = ws.f_post[cpu_qindex(17, idx, Ntotal)] + 6.0 * rho * h_w[18] * (h_cx[18] * latvxtop + h_cy[18] * latvytop + h_cz[18] * latvztop);
            }
        }
    }
}

static inline void cpu_apply_bounceback_z_if_global_boundary(
    WorkerState& ws,
    bool ownsBack,
    bool ownsFront,
    double latvxback, double latvyback, double latvzback,
    double latvxfront, double latvyfront, double latvzfront
) {
    const int Nnx = ws.geom.Nnx;
    const int Nny = ws.geom.Nny;
    const int Nnz = ws.geom.Nnz;
    const size_t Ntotal = ws.geom.Ntotal;

    if (ownsBack) {
        const int k = 0;
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < Nny; j++) {
            for (int i = 0; i < Nnx; i++) {
                const size_t idx = cpu_scalar_index(i, j, k, Nnx, Nny);
                const double rho = ws.rho[idx];

                ws.f[cpu_qindex(5, idx, Ntotal)]  = ws.f_post[cpu_qindex(6, idx, Ntotal)]  + 6.0 * rho * h_w[5]  * (h_cx[5]  * latvxback + h_cy[5]  * latvyback + h_cz[5]  * latvzback);
                ws.f[cpu_qindex(9, idx, Ntotal)]  = ws.f_post[cpu_qindex(10, idx, Ntotal)] + 6.0 * rho * h_w[9]  * (h_cx[9]  * latvxback + h_cy[9]  * latvyback + h_cz[9]  * latvzback);
                ws.f[cpu_qindex(11, idx, Ntotal)] = ws.f_post[cpu_qindex(12, idx, Ntotal)] + 6.0 * rho * h_w[11] * (h_cx[11] * latvxback + h_cy[11] * latvyback + h_cz[11] * latvzback);
                ws.f[cpu_qindex(16, idx, Ntotal)] = ws.f_post[cpu_qindex(15, idx, Ntotal)] + 6.0 * rho * h_w[16] * (h_cx[16] * latvxback + h_cy[16] * latvyback + h_cz[16] * latvzback);
                ws.f[cpu_qindex(18, idx, Ntotal)] = ws.f_post[cpu_qindex(17, idx, Ntotal)] + 6.0 * rho * h_w[18] * (h_cx[18] * latvxback + h_cy[18] * latvyback + h_cz[18] * latvzback);
            }
        }
    }

    if (ownsFront) {
        const int k = Nnz - 1;
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < Nny; j++) {
            for (int i = 0; i < Nnx; i++) {
                const size_t idx = cpu_scalar_index(i, j, k, Nnx, Nny);
                const double rho = ws.rho[idx];

                ws.f[cpu_qindex(6, idx, Ntotal)]  = ws.f_post[cpu_qindex(5, idx, Ntotal)]  + 6.0 * rho * h_w[6]  * (h_cx[6]  * latvxfront + h_cy[6]  * latvyfront + h_cz[6]  * latvzfront);
                ws.f[cpu_qindex(10, idx, Ntotal)] = ws.f_post[cpu_qindex(9, idx, Ntotal)]  + 6.0 * rho * h_w[10] * (h_cx[10] * latvxfront + h_cy[10] * latvyfront + h_cz[10] * latvzfront);
                ws.f[cpu_qindex(12, idx, Ntotal)] = ws.f_post[cpu_qindex(11, idx, Ntotal)] + 6.0 * rho * h_w[12] * (h_cx[12] * latvxfront + h_cy[12] * latvyfront + h_cz[12] * latvzfront);
                ws.f[cpu_qindex(15, idx, Ntotal)] = ws.f_post[cpu_qindex(16, idx, Ntotal)] + 6.0 * rho * h_w[15] * (h_cx[15] * latvxfront + h_cy[15] * latvyfront + h_cz[15] * latvzfront);
                ws.f[cpu_qindex(17, idx, Ntotal)] = ws.f_post[cpu_qindex(18, idx, Ntotal)] + 6.0 * rho * h_w[17] * (h_cx[17] * latvxfront + h_cy[17] * latvyfront + h_cz[17] * latvzfront);
            }
        }
    }
}

static inline void cpu_denvel(
    WorkerState& ws,
    double rhoinit,
    int simtype
) {
    const int Nnx = ws.geom.Nnx;
    const int Nny = ws.geom.Nny;
    const int Nnz = ws.geom.Nnz;
    const size_t Ntotal = ws.geom.Ntotal;

    #pragma omp parallel for collapse(3)
    for (int k = 0; k < Nnz; k++) {
        for (int j = 0; j < Nny; j++) {
            for (int i = 0; i < Nnx; i++) {
                const size_t ixyz = cpu_scalar_index(i, j, k, Nnx, Nny);

                const size_t i0  = cpu_qindex(0,  ixyz, Ntotal);
                const size_t i1  = cpu_qindex(1,  ixyz, Ntotal);
                const size_t i2  = cpu_qindex(2,  ixyz, Ntotal);
                const size_t i3  = cpu_qindex(3,  ixyz, Ntotal);
                const size_t i4  = cpu_qindex(4,  ixyz, Ntotal);
                const size_t i5  = cpu_qindex(5,  ixyz, Ntotal);
                const size_t i6  = cpu_qindex(6,  ixyz, Ntotal);
                const size_t i7  = cpu_qindex(7,  ixyz, Ntotal);
                const size_t i8  = cpu_qindex(8,  ixyz, Ntotal);
                const size_t i9  = cpu_qindex(9,  ixyz, Ntotal);
                const size_t i10 = cpu_qindex(10, ixyz, Ntotal);
                const size_t i11 = cpu_qindex(11, ixyz, Ntotal);
                const size_t i12 = cpu_qindex(12, ixyz, Ntotal);
                const size_t i13 = cpu_qindex(13, ixyz, Ntotal);
                const size_t i14 = cpu_qindex(14, ixyz, Ntotal);
                const size_t i15 = cpu_qindex(15, ixyz, Ntotal);
                const size_t i16 = cpu_qindex(16, ixyz, Ntotal);
                const size_t i17 = cpu_qindex(17, ixyz, Ntotal);
                const size_t i18 = cpu_qindex(18, ixyz, Ntotal);

                ws.rho[ixyz] =
                    ws.f[i0]  + ws.f[i1]  + ws.f[i2]  + ws.f[i3]  + ws.f[i4]  +
                    ws.f[i5]  + ws.f[i6]  + ws.f[i7]  + ws.f[i8]  + ws.f[i9]  +
                    ws.f[i10] + ws.f[i11] + ws.f[i12] + ws.f[i13] + ws.f[i14] +
                    ws.f[i15] + ws.f[i16] + ws.f[i17] + ws.f[i18];

                if (simtype == 0) {
                    ws.vx[ixyz] = (ws.f[i1] + ws.f[i7] + ws.f[i9] + ws.f[i13] + ws.f[i15]
                                 - ws.f[i2] - ws.f[i8] - ws.f[i10] - ws.f[i14] - ws.f[i16]) / ws.rho[ixyz];

                    ws.vy[ixyz] = (ws.f[i3] + ws.f[i7] + ws.f[i11] + ws.f[i14] + ws.f[i17]
                                 - ws.f[i4] - ws.f[i8] - ws.f[i12] - ws.f[i13] - ws.f[i18]) / ws.rho[ixyz];

                    ws.vz[ixyz] = (ws.f[i5] + ws.f[i9] + ws.f[i11] + ws.f[i16] + ws.f[i18]
                                 - ws.f[i6] - ws.f[i10] - ws.f[i12] - ws.f[i15] - ws.f[i17]) / ws.rho[ixyz];
                } else {
                    ws.vx[ixyz] = (ws.f[i1] + ws.f[i7] + ws.f[i9] + ws.f[i13] + ws.f[i15]
                                 - ws.f[i2] - ws.f[i8] - ws.f[i10] - ws.f[i14] - ws.f[i16]) / rhoinit;

                    ws.vy[ixyz] = (ws.f[i3] + ws.f[i7] + ws.f[i11] + ws.f[i14] + ws.f[i17]
                                 - ws.f[i4] - ws.f[i8] - ws.f[i12] - ws.f[i13] - ws.f[i18]) / rhoinit;

                    ws.vz[ixyz] = (ws.f[i5] + ws.f[i9] + ws.f[i11] + ws.f[i16] + ws.f[i18]
                                 - ws.f[i6] - ws.f[i10] - ws.f[i12] - ws.f[i15] - ws.f[i17]) / rhoinit;
                }
            }
        }
    }
}

static inline void cpu_compute_local_error(
    WorkerState& ws,
    int periodicinx,
    int periodiciny
) {
    const int Nnx = ws.geom.Nnx;
    const int Nny = ws.geom.Nny;
    const int Nnz = ws.geom.Nnz;

    const int startX = (periodicinx == 1) ? 1 : 0;
    const int endX   = (periodicinx == 1) ? (Nnx - 1) : Nnx;
    const int startY = (periodiciny == 1) ? 1 : 0;
    const int endY   = (periodiciny == 1) ? (Nny - 1) : Nny;
    const int startZ = 1;
    const int endZ   = (int)ws.part.localNzInt + 1;

    double sum1 = 0.0;
    double sum2 = 0.0;

    #pragma omp parallel for collapse(3) reduction(+:sum1,sum2)
    for (int k = startZ; k < endZ; k++) {
        for (int j = startY; j < endY; j++) {
            for (int i = startX; i < endX; i++) {
                const size_t idx = cpu_scalar_index(i, j, k, Nnx, Nny);

                const double ex = ws.vx[idx] - ws.vxerr[idx];
                const double ey = ws.vy[idx] - ws.vyerr[idx];
                const double ez = ws.vz[idx] - ws.vzerr[idx];

                const double e1 = std::sqrt(ex * ex + ey * ey + ez * ez);
                const double e2 = std::sqrt(ws.vx[idx] * ws.vx[idx] +
                                            ws.vy[idx] * ws.vy[idx] +
                                            ws.vz[idx] * ws.vz[idx]);

                ws.e1[idx] = e1;
                ws.e2[idx] = e2;

                ws.vxerr[idx] = ws.vx[idx];
                ws.vyerr[idx] = ws.vy[idx];
                ws.vzerr[idx] = ws.vz[idx];

                sum1 += e1;
                sum2 += e2;
            }
        }
    }

    ws.local_err_num = sum1;
    ws.local_err_den = sum2;
}

static inline void cpu_extract_center_velocity_if_owned(
    WorkerState& ws,
    int M2, int N2, int O2
) {
    ws.center_vx = 0.0;
    ws.center_vy = 0.0;
    ws.center_vz = 0.0;

    const long long gk0 = ws.part.localStartK;
    const long long gk1 = ws.part.localStartK + ws.part.localNzInt - 1;

    if ((long long)O2 < gk0 || (long long)O2 > gk1) return;

    const int localK = (int)((long long)O2 - gk0) + 1;
    const size_t idx = cpu_scalar_index(M2, N2, localK, ws.geom.Nnx, ws.geom.Nny);

    ws.center_vx = ws.vx[idx];
    ws.center_vy = ws.vy[idx];
    ws.center_vz = ws.vz[idx];
}

#endif
