

using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sys/time.h>
#include <algorithm>
#include <string>
#include <sstream>
#include <filesystem>
#include <numeric>
#include <type_traits>

#include <mpi.h>
#include <omp.h>

#include "../3dbase/constants.h"
#include "../3dbase/denvel.h"
#include "../3dbase/error.h"
#include "../3dbase/linkwise.h"
#include "../3dbase/mrt.h"
#include "../3dbase/periodic.h"
#include "../3dbase/srt.h"
#include "../3dbase/streaming.h"
#include "../3dbase/wetnode.h"

#include "../3dbase/hetero_constants.h"
#include "../3dbase/hetero_host_kernels.h"

#include "../3dgeneral/fileops.h"
#include "../3dgeneral/plotting.h"
#include "../3dgeneral/timing.h"

#include "../3dsetup/data.h"
#include "../3dsetup/initialisation.h"
#include "../3dsetup/nondim.h"
#include "../3dsetup/typedefs.h"

#include "../cudafunc/cudaerror.h"
#include "../cudafunc/cudainfo.h"
#include "../cudafunc/nvtx.h"
#include "../cudafunc/halo_pack_z.cuh"

#include "runtime_resource_control.h"

namespace fs = std::filesystem;

// ------------------------------------------------------------
// GPU constant memory
// ------------------------------------------------------------
__constant__ INDEX_TYPE Q;
__constant__ INDEX_TYPE cx[19];
__constant__ INDEX_TYPE cy[19];
__constant__ INDEX_TYPE cz[19];
__constant__ VALUE_TYPE w[19];
__constant__ INDEX_TYPE D[19];
__constant__ VALUE_TYPE s[19];

// ------------------------------------------------------------
// Worker types
// ------------------------------------------------------------
enum WorkerType {
    WORKER_GPU = 0,
    WORKER_CPU = 1
};

// ------------------------------------------------------------
// Decomposition
// ------------------------------------------------------------
struct ZDecomp {
    int rank = 0;
    int nranks = 1;

    INDEX_TYPE globalStartK = 0;
    INDEX_TYPE globalNzInt  = 0;

    INDEX_TYPE localNzInt   = 0;
    INDEX_TYPE localStartK  = 0;

    int prevRank = MPI_PROC_NULL;
    int nextRank = MPI_PROC_NULL;

    int workerType = WORKER_GPU;
    int gpuId = -1;
    double weight = 1.0;
};

static std::vector<int> parse_gpu_list(const std::string& s) {
    std::vector<int> g;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) g.push_back(std::stoi(tok));
    }
    return g;
}

static std::vector<int> gpu_worker_ranks(const std::vector<int>& workerTypes) {
    std::vector<int> ranks;
    for (int r = 0; r < (int)workerTypes.size(); ++r) {
        if (workerTypes[r] == WORKER_GPU) ranks.push_back(r);
    }
    return ranks;
}

struct FixedHeteroProfile {
    bool active = false;
    std::string name;
    int cpuThreads = 0;
    std::vector<INDEX_TYPE> zCounts;
};

static FixedHeteroProfile fixed_hetero_profile_for_problem(
    INDEX_TYPE Ncx,
    INDEX_TYPE Ncy,
    INDEX_TYPE Ncz,
    const std::vector<int>& workerTypes,
    int cpuPlaneOverride)
{
    FixedHeteroProfile profile;
    if (workerTypes.size() != 3 || !(Ncx == Ncy && Ncy == Ncz)) {
        return profile;
    }

    const std::vector<int> gpuRanks = gpu_worker_ranks(workerTypes);
    if (gpuRanks.size() != 2) return profile;

    int cpuRank = -1;
    for (int r = 0; r < (int)workerTypes.size(); ++r) {
        if (workerTypes[r] == WORKER_CPU) {
            cpuRank = r;
            break;
        }
    }
    if (cpuRank < 0) return profile;

    profile.zCounts.assign(workerTypes.size(), 0);

    if (cpuPlaneOverride > 0) {
        INDEX_TYPE cpuPlanes = (INDEX_TYPE)cpuPlaneOverride;
        if (cpuPlanes > Ncz - 2) cpuPlanes = Ncz - 2;
        if (cpuPlanes < 1) cpuPlanes = 1;

        const INDEX_TYPE gpuPlanes = Ncz - cpuPlanes;
        const INDEX_TYPE gpu0Planes = gpuPlanes / 2;
        const INDEX_TYPE gpu1Planes = gpuPlanes - gpu0Planes;

        profile.active = true;
        profile.name = "override-cpu-planes-" + std::to_string((long long)cpuPlanes);
        profile.cpuThreads = 0;
        profile.zCounts[gpuRanks[0]] = gpu0Planes;
        profile.zCounts[gpuRanks[1]] = gpu1Planes;
        profile.zCounts[cpuRank] = cpuPlanes;
        return profile;
    }

    if (Ncx == 100) {
        profile.active = true;
        profile.name = "fixed-100";
        profile.cpuThreads = 8;
        profile.zCounts[gpuRanks[0]] = 50;
        profile.zCounts[gpuRanks[1]] = 48;
        profile.zCounts[cpuRank] = 2;
    } else if (Ncx == 150) {
        profile.active = true;
        profile.name = "fixed-150";
        profile.cpuThreads = 16;
        profile.zCounts[gpuRanks[0]] = 75;
        profile.zCounts[gpuRanks[1]] = 72;
        profile.zCounts[cpuRank] = 3;
    } else if (Ncx == 200) {
        profile.active = true;
        profile.name = "fixed-200";
        profile.cpuThreads = 32;
        profile.zCounts[gpuRanks[0]] = 99;
        profile.zCounts[gpuRanks[1]] = 97;
        profile.zCounts[cpuRank] = 4;
    } else if (Ncx == 250) {
        profile.active = true;
        profile.name = "fixed-250";
        profile.cpuThreads = 32;
        profile.zCounts[gpuRanks[0]] = 119;
        profile.zCounts[gpuRanks[1]] = 126;
        profile.zCounts[cpuRank] = 5;
    }

    return profile;
}

static ZDecomp decompose_z_fixed_counts(
    int rank,
    int nranks,
    INDEX_TYPE Nnz_global,
    INDEX_TYPE periodicinz,
    const std::vector<INDEX_TYPE>& counts,
    const std::vector<int>& workerTypes,
    const std::vector<int>& gpuIds)
{
    ZDecomp d{};
    d.rank = rank;
    d.nranks = nranks;

    if (periodicinz == 1) {
        d.globalStartK = 1;
        d.globalNzInt  = Nnz_global - 2;
    } else {
        d.globalStartK = 0;
        d.globalNzInt  = Nnz_global;
    }

    d.workerType = workerTypes[rank];
    d.gpuId      = gpuIds[rank];
    d.weight     = 1.0;

    INDEX_TYPE assigned = 0;
    for (INDEX_TYPE c : counts) assigned += c;
    if ((int)counts.size() != nranks || assigned != d.globalNzInt) {
        d.localNzInt = 0;
        return d;
    }

    INDEX_TYPE offset = 0;
    for (int r = 0; r < rank; r++) offset += counts[r];

    d.localNzInt  = counts[rank];
    d.localStartK = d.globalStartK + offset;

    if (nranks == 1) {
        d.prevRank = MPI_PROC_NULL;
        d.nextRank = MPI_PROC_NULL;
    } else if (periodicinz == 1) {
        d.prevRank = (rank - 1 + nranks) % nranks;
        d.nextRank = (rank + 1) % nranks;
    } else {
        d.prevRank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        d.nextRank = (rank == nranks - 1) ? MPI_PROC_NULL : rank + 1;
    }

    return d;
}

static ZDecomp decompose_z_weighted(
    int rank,
    int nranks,
    INDEX_TYPE Nnz_global,
    INDEX_TYPE periodicinz,
    const std::vector<double>& weights,
    const std::vector<int>& workerTypes,
    const std::vector<int>& gpuIds)
{
    ZDecomp d{};
    d.rank = rank;
    d.nranks = nranks;

    if (periodicinz == 1) {
        d.globalStartK = 1;
        d.globalNzInt  = Nnz_global - 2;
    } else {
        d.globalStartK = 0;
        d.globalNzInt  = Nnz_global;
    }

    d.workerType = workerTypes[rank];
    d.gpuId      = gpuIds[rank];
    d.weight     = weights[rank];

    std::vector<INDEX_TYPE> counts(nranks, 0);
    std::vector<double> normalizedWeights(nranks, 0.0);

    double weightSum = 0.0;
    for (double wgt : weights) {
        if (wgt > 0.0) weightSum += wgt;
    }
    if (weightSum <= 0.0) weightSum = 1.0;
    for (int r = 0; r < nranks; r++) {
        normalizedWeights[r] = std::max(0.0, static_cast<double>(weights[r])) / weightSum;
    }

    INDEX_TYPE assigned = 0;
    for (int r = 0; r < nranks; r++) {
        counts[r] = static_cast<INDEX_TYPE>(normalizedWeights[r] * static_cast<double>(d.globalNzInt));
        assigned += counts[r];
    }

    // Keep each hetero worker present, but let the measured relative speed
    // weights choose the actual CPU/GPU Z-plane share.
    if (nranks == 3) {
        for (int r = 0; r < nranks; r++) {
            if (workerTypes[r] != WORKER_CPU) continue;

            const INDEX_TYPE cpuMinPlanes = std::min<INDEX_TYPE>(d.globalNzInt, 1);

            if (counts[r] < cpuMinPlanes) {
                INDEX_TYPE need = cpuMinPlanes - counts[r];
                counts[r] += need;
                assigned += need;

                while (assigned > d.globalNzInt) {
                    int donor = -1;
                    for (int rr = 0; rr < nranks; rr++) {
                        if (rr == r || counts[rr] <= 1) continue;
                        if (donor < 0 || counts[rr] > counts[donor]) donor = rr;
                    }
                    if (donor < 0) break;
                    counts[donor]--;
                    assigned--;
                }
            }
            break;
        }
    }

    while (assigned < d.globalNzInt) {
        int best = 0;
        for (int r = 1; r < nranks; r++) {
            if (normalizedWeights[r] > normalizedWeights[best]) best = r;
        }
        counts[best]++;
        assigned++;
    }

    INDEX_TYPE offset = 0;
    for (int r = 0; r < rank; r++) offset += counts[r];

    d.localNzInt  = counts[rank];
    d.localStartK = d.globalStartK + offset;

    if (nranks == 1) {
        d.prevRank = MPI_PROC_NULL;
        d.nextRank = MPI_PROC_NULL;
    } else if (periodicinz == 1) {
        d.prevRank = (rank - 1 + nranks) % nranks;
        d.nextRank = (rank + 1) % nranks;
    } else {
        d.prevRank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        d.nextRank = (rank == nranks - 1) ? MPI_PROC_NULL : rank + 1;
    }

    return d;
}

static int choose_auto_cpu_threads(
    INDEX_TYPE Ncx,
    INDEX_TYPE Ncy,
    INDEX_TYPE Ncz,
    INDEX_TYPE cpuLocalNz,
    int availableThreads)
{
    int byPlanes = 16;
    if (cpuLocalNz <= 2) {
        byPlanes = 16;
    } else if (cpuLocalNz <= 4) {
        byPlanes = 16;
    } else if (cpuLocalNz <= 8) {
        byPlanes = 24;
    } else if (cpuLocalNz <= 16) {
        byPlanes = 32;
    } else {
        byPlanes = 64;
    }

    const INDEX_TYPE maxDim = std::max(Ncx, std::max(Ncy, Ncz));
    int bySize = 16;
    if (maxDim <= 100) {
        bySize = 8;
    } else if (maxDim <= 150) {
        bySize = 16;
    } else if (maxDim <= 200) {
        bySize = 32;
    } else {
        bySize = 64;
    }

    int selected = std::min(byPlanes, bySize);
    if (availableThreads > 0) selected = std::min(selected, availableThreads);
    return std::max(1, selected);
}

static int choose_auto_cpu_thread_limit_for_problem(
    INDEX_TYPE Ncx,
    INDEX_TYPE Ncy,
    INDEX_TYPE Ncz)
{
    const INDEX_TYPE maxDim = std::max(Ncx, std::max(Ncy, Ncz));
    if (maxDim <= 100) return 8;
    if (maxDim <= 150) return 16;
    if (maxDim <= 200) return 32;
    return 64;
}

static std::vector<double> cap_cpu_weight_for_problem(
    const std::vector<double>& rawWeights,
    const std::vector<int>& workerTypes,
    INDEX_TYPE Ncx,
    INDEX_TYPE Ncy,
    INDEX_TYPE Ncz)
{
    std::vector<double> weights = rawWeights;
    if (weights.size() != workerTypes.size()) return weights;

    double sum = 0.0;
    for (double& w : weights) {
        if (w < 0.0) w = 0.0;
        sum += w;
    }
    if (sum <= 0.0) {
        const double uniform = 1.0 / std::max<int>(1, (int)weights.size());
        for (double& w : weights) w = uniform;
    } else {
        for (double& w : weights) w /= sum;
    }

    const INDEX_TYPE maxDim = std::max(Ncx, std::max(Ncy, Ncz));
    double cpuCap = 0.02;
    if (maxDim > 250) {
        cpuCap = 0.03;
    }

    for (size_t r = 0; r < workerTypes.size(); ++r) {
        if (workerTypes[r] != WORKER_CPU) continue;

        weights[r] = std::min(cpuCap, weights[r]);

        double gpuSum = 0.0;
        for (size_t rr = 0; rr < workerTypes.size(); ++rr) {
            if (rr != r && workerTypes[rr] == WORKER_GPU) gpuSum += weights[rr];
        }

        if (gpuSum > 0.0) {
            const double remaining = 1.0 - weights[r];
            for (size_t rr = 0; rr < workerTypes.size(); ++rr) {
                if (rr != r && workerTypes[rr] == WORKER_GPU) {
                    weights[rr] = remaining * (weights[rr] / gpuSum);
                }
            }
        }
        break;
    }

    return weights;
}

// ------------------------------------------------------------
// Memory container
// ------------------------------------------------------------
struct LocalFields {
    VALUE_TYPE *f      = nullptr;
    VALUE_TYPE *f_post = nullptr;
    VALUE_TYPE *vx     = nullptr;
    VALUE_TYPE *vy     = nullptr;
    VALUE_TYPE *vz     = nullptr;
    VALUE_TYPE *rho    = nullptr;
    VALUE_TYPE *vxerr  = nullptr;
    VALUE_TYPE *vyerr  = nullptr;
    VALUE_TYPE *vzerr  = nullptr;
    VALUE_TYPE *e1     = nullptr;
    VALUE_TYPE *e2     = nullptr;

    std::vector<VALUE_TYPE> h_f;
    std::vector<VALUE_TYPE> h_f_post;
    std::vector<VALUE_TYPE> h_vx;
    std::vector<VALUE_TYPE> h_vy;
    std::vector<VALUE_TYPE> h_vz;
    std::vector<VALUE_TYPE> h_rho;
    std::vector<VALUE_TYPE> h_vxerr;
    std::vector<VALUE_TYPE> h_vyerr;
    std::vector<VALUE_TYPE> h_vzerr;
    std::vector<VALUE_TYPE> h_e1;
    std::vector<VALUE_TYPE> h_e2;
};

static void allocate_cpu_fields(LocalFields& fld, size_t Ntotal) {
    fld.h_f.resize(Ntotal * 19);
    fld.h_f_post.resize(Ntotal * 19);
    fld.h_vx.resize(Ntotal);
    fld.h_vy.resize(Ntotal);
    fld.h_vz.resize(Ntotal);
    fld.h_rho.resize(Ntotal);
    fld.h_vxerr.resize(Ntotal);
    fld.h_vyerr.resize(Ntotal);
    fld.h_vzerr.resize(Ntotal);
    fld.h_e1.resize(Ntotal);
    fld.h_e2.resize(Ntotal);

    fld.f      = fld.h_f.data();
    fld.f_post = fld.h_f_post.data();
    fld.vx     = fld.h_vx.data();
    fld.vy     = fld.h_vy.data();
    fld.vz     = fld.h_vz.data();
    fld.rho    = fld.h_rho.data();
    fld.vxerr  = fld.h_vxerr.data();
    fld.vyerr  = fld.h_vyerr.data();
    fld.vzerr  = fld.h_vzerr.data();
    fld.e1     = fld.h_e1.data();
    fld.e2     = fld.h_e2.data();
}

static void allocate_gpu_fields(LocalFields& fld, size_t Ntotal) {
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&fld.f,      Ntotal * 19 * sizeof(VALUE_TYPE)));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&fld.f_post, Ntotal * 19 * sizeof(VALUE_TYPE)));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&fld.vx,     Ntotal * sizeof(VALUE_TYPE)));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&fld.vy,     Ntotal * sizeof(VALUE_TYPE)));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&fld.vz,     Ntotal * sizeof(VALUE_TYPE)));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&fld.rho,    Ntotal * sizeof(VALUE_TYPE)));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&fld.vxerr,  Ntotal * sizeof(VALUE_TYPE)));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&fld.vyerr,  Ntotal * sizeof(VALUE_TYPE)));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&fld.vzerr,  Ntotal * sizeof(VALUE_TYPE)));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&fld.e1,     Ntotal * sizeof(VALUE_TYPE)));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&fld.e2,     Ntotal * sizeof(VALUE_TYPE)));
}

static void free_gpu_fields(LocalFields& fld) {
    if (fld.f)      CUDA_SAFE_CALL(cudaFree(fld.f));
    if (fld.f_post) CUDA_SAFE_CALL(cudaFree(fld.f_post));
    if (fld.vx)     CUDA_SAFE_CALL(cudaFree(fld.vx));
    if (fld.vy)     CUDA_SAFE_CALL(cudaFree(fld.vy));
    if (fld.vz)     CUDA_SAFE_CALL(cudaFree(fld.vz));
    if (fld.rho)    CUDA_SAFE_CALL(cudaFree(fld.rho));
    if (fld.vxerr)  CUDA_SAFE_CALL(cudaFree(fld.vxerr));
    if (fld.vyerr)  CUDA_SAFE_CALL(cudaFree(fld.vyerr));
    if (fld.vzerr)  CUDA_SAFE_CALL(cudaFree(fld.vzerr));
    if (fld.e1)     CUDA_SAFE_CALL(cudaFree(fld.e1));
    if (fld.e2)     CUDA_SAFE_CALL(cudaFree(fld.e2));
}

static std::vector<double> normalize_worker_weights(
    const std::vector<double>& rawWeights,
    const std::vector<int>& workerTypes)
{
    std::vector<double> weights = rawWeights;
    double sum = 0.0;
    for (double& w : weights) {
        if (w < 0.0) w = 0.0;
        sum += w;
    }
    if (sum <= 0.0) {
        const double uniform = 1.0 / std::max<int>(1, (int)weights.size());
        for (double& w : weights) w = uniform;
    } else {
        for (double& w : weights) w /= sum;
    }

    if (weights.size() == workerTypes.size()) {
        for (size_t r = 0; r < workerTypes.size(); ++r) {
            if (workerTypes[r] != WORKER_CPU) continue;

            const double cpuTarget = 0.02;
            const double cpuCap = 0.04;
            weights[r] = std::max(cpuTarget, std::min(cpuCap, weights[r]));

            double gpuSum = 0.0;
            for (size_t rr = 0; rr < workerTypes.size(); ++rr) {
                if (rr != r && workerTypes[rr] == WORKER_GPU) gpuSum += weights[rr];
            }
            if (gpuSum > 0.0) {
                const double remaining = 1.0 - weights[r];
                for (size_t rr = 0; rr < workerTypes.size(); ++rr) {
                    if (rr != r && workerTypes[rr] == WORKER_GPU) {
                        weights[rr] = remaining * (weights[rr] / gpuSum);
                    }
                }
            }
            break;
        }
    }

    sum = 0.0;
    for (double& w : weights) {
        if (w < 0.0) w = 0.0;
        sum += w;
    }
    if (sum > 0.0) {
        for (double& w : weights) w /= sum;
    }
    return weights;
}

static std::vector<double> stabilize_relative_weights(
    const std::vector<double>& measuredSpeeds,
    const std::vector<double>& defaultWeights,
    const std::vector<int>& workerTypes)
{
    std::vector<double> measured = normalize_worker_weights(measuredSpeeds, workerTypes);
    std::vector<double> defaults = normalize_worker_weights(defaultWeights, workerTypes);
    std::vector<double> blended(measured.size(), 0.0);

    const double measuredFactor = 0.60;
    const double defaultFactor  = 0.40;

    for (size_t i = 0; i < blended.size(); ++i) {
        blended[i] = measuredFactor * measured[i] + defaultFactor * defaults[i];
    }

    for (size_t i = 0; i < blended.size(); ++i) {
        if (workerTypes[i] == WORKER_GPU) {
            blended[i] = std::max(0.20, std::min(0.70, blended[i]));
        } else if (workerTypes[i] == WORKER_CPU) {
            blended[i] = std::max(0.02, std::min(0.04, blended[i]));
        }
    }

    double sum = 0.0;
    for (double w : blended) sum += w;
    if (sum > 0.0) {
        for (double& w : blended) w /= sum;
    }

    // Re-apply floors after normalization so no rank collapses to an unusable slab.
    double floorSum = 0.0;
    for (size_t i = 0; i < blended.size(); ++i) {
        if (workerTypes[i] == WORKER_GPU) {
            blended[i] = std::max(0.20, blended[i]);
        } else if (workerTypes[i] == WORKER_CPU) {
            blended[i] = std::max(0.02, blended[i]);
        }
        floorSum += blended[i];
    }

    if (floorSum > 0.0) {
        for (double& w : blended) w /= floorSum;
    }

    return blended;
}

static double calibrate_relative_worker_speed(
    bool isGPUWorker,
    INDEX_TYPE NnxCal,
    INDEX_TYPE NnyCal,
    INDEX_TYPE NnzCal,
    VALUE_TYPE rhoinit,
    VALUE_TYPE vxinit,
    VALUE_TYPE vyinit,
    VALUE_TYPE vzinit,
    VALUE_TYPE tau,
    INDEX_TYPE simtype,
    INDEX_TYPE relaxtype)
{
    const int calibrationIters = 5;
    const size_t NtotalCal = (size_t)NnxCal * (size_t)NnyCal * (size_t)NnzCal;

    LocalFields calFld;
    if (isGPUWorker) allocate_gpu_fields(calFld, NtotalCal);
    else             allocate_cpu_fields(calFld, NtotalCal);

    if (isGPUWorker) {
        dim3 block(8, 8, 8);
        dim3 grid((NnxCal + block.x - 1) / block.x,
                  (NnyCal + block.y - 1) / block.y,
                  (NnzCal + block.z - 1) / block.z);

        initialiseDenVelBasic<<<grid, block>>>(NnxCal, NnyCal, NnzCal, calFld.rho, rhoinit,
                                                calFld.vx, vxinit, calFld.vy, vyinit, calFld.vz, vzinit);
        CUDA_CHECK_ERROR();
        initialiseEDF<<<grid, block>>>(calFld.f, calFld.f_post, NnxCal, NnyCal, NnzCal,
                                       calFld.rho, rhoinit, simtype, tau, calFld.vx, calFld.vy, calFld.vz);
        CUDA_CHECK_ERROR();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        double t0 = MPI_Wtime();
        for (int iter = 0; iter < calibrationIters; ++iter) {
            if (relaxtype == 0) {
                collBGK<<<grid, block>>>(calFld.f, calFld.f_post, NnxCal, NnyCal, NnzCal,
                                         calFld.rho, rhoinit, simtype, tau, calFld.vx, calFld.vy, calFld.vz);
            } else {
                collMRT<<<grid, block>>>(calFld.f, calFld.f_post, NnxCal, NnyCal, NnzCal,
                                         calFld.rho, rhoinit, simtype, calFld.vx, calFld.vy, calFld.vz);
            }
            CUDA_CHECK_ERROR();
            streaming<<<grid, block>>>(calFld.f, calFld.f_post, NnxCal, NnyCal, NnzCal);
            CUDA_CHECK_ERROR();
            denVel<<<grid, block>>>(calFld.f, NnxCal, NnyCal, NnzCal,
                                    calFld.rho, rhoinit, simtype, calFld.vx, calFld.vy, calFld.vz);
            CUDA_CHECK_ERROR();
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        double t1 = MPI_Wtime();

        free_gpu_fields(calFld);
        const double elapsed = std::max(1e-12, t1 - t0);
        return (double)(NnxCal * NnyCal * (NnzCal - 2)) * calibrationIters / elapsed;
    }

    hetero_host::initialiseDenVelBasic_host<VALUE_TYPE, INDEX_TYPE>(NnxCal, NnyCal, NnzCal, calFld.rho, rhoinit,
                                                                     calFld.vx, vxinit, calFld.vy, vyinit, calFld.vz, vzinit);
    hetero_host::initialiseEDF_host<VALUE_TYPE, INDEX_TYPE>(calFld.f, calFld.f_post, NnxCal, NnyCal, NnzCal,
                                                             calFld.rho, rhoinit, simtype, tau, calFld.vx, calFld.vy, calFld.vz);

    double t0 = MPI_Wtime();
    for (int iter = 0; iter < calibrationIters; ++iter) {
        if (relaxtype == 0) {
            hetero_host::collBGK_host<VALUE_TYPE, INDEX_TYPE>(calFld.f, calFld.f_post, NnxCal, NnyCal, NnzCal,
                                                              calFld.rho, rhoinit, simtype, tau, calFld.vx, calFld.vy, calFld.vz);
        } else {
            hetero_host::collMRT_host<VALUE_TYPE, INDEX_TYPE>(calFld.f, calFld.f_post, NnxCal, NnyCal, NnzCal,
                                                              calFld.rho, rhoinit, simtype, tau, calFld.vx, calFld.vy, calFld.vz);
        }
        hetero_host::streaming_host<VALUE_TYPE, INDEX_TYPE>(calFld.f, calFld.f_post, NnxCal, NnyCal, NnzCal);
        hetero_host::denVel_host<VALUE_TYPE, INDEX_TYPE>(calFld.f, NnxCal, NnyCal, NnzCal,
                                                         calFld.rho, rhoinit, simtype, calFld.vx, calFld.vy, calFld.vz);
    }
    double t1 = MPI_Wtime();

    const double elapsed = std::max(1e-12, t1 - t0);
    return (double)(NnxCal * NnyCal * (NnzCal - 2)) * calibrationIters / elapsed;
}

static void streaming_host_zrange(
    VALUE_TYPE *f,
    VALUE_TYPE *f_post,
    INDEX_TYPE Nnx,
    INDEX_TYPE Nny,
    INDEX_TYPE Nnz,
    INDEX_TYPE kBegin,
    INDEX_TYPE kEnd)
{
    if (kBegin >= kEnd) return;

    #pragma omp parallel for collapse(3)
    for (INDEX_TYPE k = kBegin; k < kEnd; ++k) {
        for (INDEX_TYPE j = 0; j < Nny; ++j) {
            for (INDEX_TYPE i = 0; i < Nnx; ++i) {
                for (INDEX_TYPE l = 0; l < hetero_const::QH; ++l) {
                    const INDEX_TYPE id = i - hetero_const::cx[l];
                    const INDEX_TYPE jd = j - hetero_const::cy[l];
                    const INDEX_TYPE kd = k - hetero_const::cz[l];

                    if (id >= 0 && id < Nnx &&
                        jd >= 0 && jd < Nny &&
                        kd >= 0 && kd < Nnz) {
                        const size_t dst = hetero_const::pdf_index(l, i,  j,  k,  Nnx, Nny, Nnz);
                        const size_t src = hetero_const::pdf_index(l, id, jd, kd, Nnx, Nny, Nnz);
                        f[dst] = f_post[src];
                    }
                }
            }
        }
    }
}

template <typename T, typename I>
__global__ void streaming_zrange_kernel(
    T *f,
    const T *f_post,
    I Nnx,
    I Nny,
    I Nnz,
    I kBegin,
    I kEnd)
{
    I idx = blockIdx.x * blockDim.x + threadIdx.x;

    I total = Nnx * Nny * (kEnd - kBegin);
    if (idx >= total) return;

    I localK = idx / (Nnx * Nny);
    I rem = idx % (Nnx * Nny);
    I j = rem / Nnx;
    I i = rem % Nnx;
    I k = kBegin + localK;

    for (I l = 0; l < 19; ++l) {
        I id = i - cx[l];
        I jd = j - cy[l];
        I kd = k - cz[l];

        if (id >= 0 && id < Nnx &&
            jd >= 0 && jd < Nny &&
            kd >= 0 && kd < Nnz) {

            size_t dst =
                (size_t)l * Nnx * Nny * Nnz +
                (size_t)k * Nnx * Nny +
                (size_t)j * Nnx + i;

            size_t src =
                (size_t)l * Nnx * Nny * Nnz +
                (size_t)kd * Nnx * Nny +
                (size_t)jd * Nnx + id;

            f[dst] = f_post[src];
        }
    }
}

static constexpr int ZDIRS = 10;
static constexpr int ZHALF_DIRS = 5;
static constexpr int ZDIR_SET_PLUS = 1;
static constexpr int ZDIR_SET_MINUS = -1;

template <typename T, typename I>
__global__ void packZPlaneZDirs(
    const T *f_post,
    T *sendBuf,
    I Nnx,
    I Nny,
    I Nnz,
    I kPlane)
{
    const int zDirs[ZDIRS] = {
        5, 6, 9, 10, 11, 12, 15, 16, 17, 18
    };

    I p = blockIdx.x * blockDim.x + threadIdx.x;
    I planeSize = Nnx * Nny;

    if (p >= planeSize) return;

    I j = p / Nnx;
    I i = p % Nnx;

    for (int q = 0; q < ZDIRS; ++q) {
        int l = zDirs[q];

        size_t src =
            (size_t)l * Nnx * Nny * Nnz +
            (size_t)kPlane * Nnx * Nny +
            (size_t)j * Nnx + i;

        size_t dst =
            (size_t)q * planeSize + p;

        sendBuf[dst] = f_post[src];
    }
}

template <typename T, typename I>
__global__ void unpackZPlaneZDirs(
    const T *recvBuf,
    T *f_post,
    I Nnx,
    I Nny,
    I Nnz,
    I kGhost)
{
    const int zDirs[ZDIRS] = {
        5, 6, 9, 10, 11, 12, 15, 16, 17, 18
    };

    I p = blockIdx.x * blockDim.x + threadIdx.x;
    I planeSize = Nnx * Nny;

    if (p >= planeSize) return;

    I j = p / Nnx;
    I i = p % Nnx;

    for (int q = 0; q < ZDIRS; ++q) {
        int l = zDirs[q];

        size_t src =
            (size_t)q * planeSize + p;

        size_t dst =
            (size_t)l * Nnx * Nny * Nnz +
            (size_t)kGhost * Nnx * Nny +
            (size_t)j * Nnx + i;

        f_post[dst] = recvBuf[src];
    }
}

template <typename T, typename I>
__global__ void packZPlaneHalfDirs(
    const T *f_post,
    T *sendBuf,
    I Nnx,
    I Nny,
    I Nnz,
    I kPlane,
    int dirSet)
{
    const int zPlus[ZHALF_DIRS]  = {5, 9, 11, 16, 18};
    const int zMinus[ZHALF_DIRS] = {6, 10, 12, 15, 17};

    I p = blockIdx.x * blockDim.x + threadIdx.x;
    I planeSize = Nnx * Nny;

    if (p >= planeSize) return;

    I j = p / Nnx;
    I i = p % Nnx;

    for (int q = 0; q < ZHALF_DIRS; ++q) {
        int l = (dirSet == ZDIR_SET_PLUS) ? zPlus[q] : zMinus[q];

        size_t src =
            (size_t)l * Nnx * Nny * Nnz +
            (size_t)kPlane * Nnx * Nny +
            (size_t)j * Nnx + i;

        size_t dst =
            (size_t)q * planeSize + p;

        sendBuf[dst] = f_post[src];
    }
}

template <typename T, typename I>
__global__ void unpackZPlaneHalfDirs(
    const T *recvBuf,
    T *f_post,
    I Nnx,
    I Nny,
    I Nnz,
    I kGhost,
    int dirSet)
{
    const int zPlus[ZHALF_DIRS]  = {5, 9, 11, 16, 18};
    const int zMinus[ZHALF_DIRS] = {6, 10, 12, 15, 17};

    I p = blockIdx.x * blockDim.x + threadIdx.x;
    I planeSize = Nnx * Nny;

    if (p >= planeSize) return;

    I j = p / Nnx;
    I i = p % Nnx;

    for (int q = 0; q < ZHALF_DIRS; ++q) {
        int l = (dirSet == ZDIR_SET_PLUS) ? zPlus[q] : zMinus[q];

        size_t src =
            (size_t)q * planeSize + p;

        size_t dst =
            (size_t)l * Nnx * Nny * Nnz +
            (size_t)kGhost * Nnx * Nny +
            (size_t)j * Nnx + i;

        f_post[dst] = recvBuf[src];
    }
}

template <typename T, typename I>
static void packZPlaneZDirs_host(
    const T *f_post,
    T *sendBuf,
    I Nnx,
    I Nny,
    I Nnz,
    I kPlane)
{
    const int zDirs[ZDIRS] = {
        5, 6, 9, 10, 11, 12, 15, 16, 17, 18
    };

    I planeSize = Nnx * Nny;

    #pragma omp parallel for
    for (I p = 0; p < planeSize; ++p) {
        I j = p / Nnx;
        I i = p % Nnx;

        for (int q = 0; q < ZDIRS; ++q) {
            int l = zDirs[q];

            size_t src =
                (size_t)l * Nnx * Nny * Nnz +
                (size_t)kPlane * Nnx * Nny +
                (size_t)j * Nnx + i;

            size_t dst =
                (size_t)q * planeSize + p;

            sendBuf[dst] = f_post[src];
        }
    }
}

template <typename T, typename I>
static void packZPlaneHalfDirs_host(
    const T *f_post,
    T *sendBuf,
    I Nnx,
    I Nny,
    I Nnz,
    I kPlane,
    int dirSet)
{
    const int zPlus[ZHALF_DIRS]  = {5, 9, 11, 16, 18};
    const int zMinus[ZHALF_DIRS] = {6, 10, 12, 15, 17};

    I planeSize = Nnx * Nny;

    #pragma omp parallel for
    for (I p = 0; p < planeSize; ++p) {
        I j = p / Nnx;
        I i = p % Nnx;

        for (int q = 0; q < ZHALF_DIRS; ++q) {
            int l = (dirSet == ZDIR_SET_PLUS) ? zPlus[q] : zMinus[q];

            size_t src =
                (size_t)l * Nnx * Nny * Nnz +
                (size_t)kPlane * Nnx * Nny +
                (size_t)j * Nnx + i;

            size_t dst =
                (size_t)q * planeSize + p;

            sendBuf[dst] = f_post[src];
        }
    }
}

template <typename T, typename I>
static void unpackZPlaneZDirs_host(
    const T *recvBuf,
    T *f_post,
    I Nnx,
    I Nny,
    I Nnz,
    I kGhost)
{
    const int zDirs[ZDIRS] = {
        5, 6, 9, 10, 11, 12, 15, 16, 17, 18
    };

    I planeSize = Nnx * Nny;

    #pragma omp parallel for
    for (I p = 0; p < planeSize; ++p) {
        I j = p / Nnx;
        I i = p % Nnx;

        for (int q = 0; q < ZDIRS; ++q) {
            int l = zDirs[q];

            size_t src =
                (size_t)q * planeSize + p;

            size_t dst =
                (size_t)l * Nnx * Nny * Nnz +
                (size_t)kGhost * Nnx * Nny +
                (size_t)j * Nnx + i;

            f_post[dst] = recvBuf[src];
        }
    }
}

template <typename T, typename I>
static void unpackZPlaneHalfDirs_host(
    const T *recvBuf,
    T *f_post,
    I Nnx,
    I Nny,
    I Nnz,
    I kGhost,
    int dirSet)
{
    const int zPlus[ZHALF_DIRS]  = {5, 9, 11, 16, 18};
    const int zMinus[ZHALF_DIRS] = {6, 10, 12, 15, 17};

    I planeSize = Nnx * Nny;

    #pragma omp parallel for
    for (I p = 0; p < planeSize; ++p) {
        I j = p / Nnx;
        I i = p % Nnx;

        for (int q = 0; q < ZHALF_DIRS; ++q) {
            int l = (dirSet == ZDIR_SET_PLUS) ? zPlus[q] : zMinus[q];

            size_t src =
                (size_t)q * planeSize + p;

            size_t dst =
                (size_t)l * Nnx * Nny * Nnz +
                (size_t)kGhost * Nnx * Nny +
                (size_t)j * Nnx + i;

            f_post[dst] = recvBuf[src];
        }
    }
}

static void initialise_error_history(size_t Ntotal, VALUE_TYPE* vx, VALUE_TYPE* vy, VALUE_TYPE* vz,
                                     VALUE_TYPE* vxerr, VALUE_TYPE* vyerr, VALUE_TYPE* vzerr,
                                     VALUE_TYPE* e1, VALUE_TYPE* e2)
{
    #pragma omp parallel for
    for (size_t i = 0; i < Ntotal; ++i) {
        vxerr[i] = vx[i];
        vyerr[i] = vy[i];
        vzerr[i] = vz[i];
        e1[i] = 0.0;
        e2[i] = 0.0;
    }
}

static long long estimate_local_rank_bytes(size_t Ntotal, size_t haloElems, bool isGPUWorker)
{
    long long bytes = 0;
    bytes += (long long)(Ntotal * 19 * sizeof(VALUE_TYPE)); // f
    bytes += (long long)(Ntotal * 19 * sizeof(VALUE_TYPE)); // f_post
    bytes += (long long)(Ntotal * sizeof(VALUE_TYPE));      // vx
    bytes += (long long)(Ntotal * sizeof(VALUE_TYPE));      // vy
    bytes += (long long)(Ntotal * sizeof(VALUE_TYPE));      // vz
    bytes += (long long)(Ntotal * sizeof(VALUE_TYPE));      // rho
    bytes += (long long)(Ntotal * sizeof(VALUE_TYPE));      // vxerr
    bytes += (long long)(Ntotal * sizeof(VALUE_TYPE));      // vyerr
    bytes += (long long)(Ntotal * sizeof(VALUE_TYPE));      // vzerr
    bytes += (long long)(Ntotal * sizeof(VALUE_TYPE));      // e1
    bytes += (long long)(Ntotal * sizeof(VALUE_TYPE));      // e2

    if (isGPUWorker) {
        bytes += 4LL * (long long)(haloElems * sizeof(VALUE_TYPE)); // d_send/recv
        bytes += 4LL * (long long)(haloElems * sizeof(VALUE_TYPE)); // host staging buffers
    } else {
        bytes += 4LL * (long long)(haloElems * sizeof(VALUE_TYPE)); // host halo buffers
    }

    return bytes;
}

static void ordered_rank_print(int rank, int nranks, const std::string& msg)
{
    for (int r = 0; r < nranks; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            std::printf("%s", msg.c_str());
            std::fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

static int cuda_cores_per_sm_from_device_prop(const cudaDeviceProp& prop)
{
    // CUDA does not expose CUDA core count directly. This maps known NVIDIA
    // architectures to FP32 CUDA cores per SM; L40S/Ada (sm_89) uses 128.
    switch (prop.major) {
        case 2: return (prop.minor == 1) ? 48 : 32;
        case 3: return 192;
        case 5: return 128;
        case 6: return (prop.minor == 0) ? 64 : 128;
        case 7: return (prop.minor == 0 || prop.minor == 5) ? 64 : 128;
        case 8: return (prop.minor == 0) ? 64 : 128;
        case 9: return 128;
        default: return 128;
    }
}

static MPI_Datatype mpi_value_type()
{
    if constexpr (std::is_same<VALUE_TYPE, double>::value) {
        return MPI_DOUBLE;
    } else if constexpr (std::is_same<VALUE_TYPE, float>::value) {
        return MPI_FLOAT;
    } else {
        return MPI_DOUBLE;
    }
}

static void append_rank_timing_summary_to_file(
    int rank,
    bool isGPUWorker,
    int gpuid,
    const ZDecomp& zdec,
    INDEX_TYPE Nnx,
    INDEX_TYPE Nny,
    double t_compute,
    double t_comm,
    double t_io,
    double wallLocal,
    INDEX_TYPE step)
{
    std::filesystem::create_directories("perf_outputs");

    std::ofstream fout("perf_outputs/per_rank_timing_summary.txt", std::ios::app);
    if (!fout.is_open()) return;

    double total_local_measured = t_compute + t_comm + t_io;
    if (total_local_measured <= 0.0) total_local_measured = 1e-12;

    double comm_percent_local = 100.0 * t_comm / total_local_measured;
    double avg_step_local = (step > 0) ? (wallLocal / (double)step) : 0.0;

    size_t local_slab_cells = (size_t)Nnx * (size_t)Nny * (size_t)zdec.localNzInt;

    fout << "[Rank " << rank << "] "
         << (isGPUWorker ? "GPU" : "CPU");
    if (isGPUWorker) fout << " gpu=" << gpuid;
    fout << "\n";
    fout << "  local slab size (Z planes): " << zdec.localNzInt << "\n";
    fout << "  local slab cells          : " << local_slab_cells << "\n";
    fout << "  compute time             : " << t_compute << " s\n";
    fout << "  communication time       : " << t_comm << " s\n";
    fout << "  communication percentage : " << comm_percent_local << " %\n";
    fout << "  average time per step    : " << avg_step_local << " s\n";
    fout << "  wall time (local rank)   : " << wallLocal << " s\n";
    fout << "--------------------------------------------------\n";
    fout.close();
}

// ------------------------------------------------------------
// Output gather
// ------------------------------------------------------------
static void gather_and_write_outputs(
    int rank, int nranks,
    const ZDecomp& zdec,
    INDEX_TYPE Ncx, INDEX_TYPE Ncy, INDEX_TYPE Ncz,
    INDEX_TYPE NnxG, INDEX_TYPE NnyG, INDEX_TYPE NnzG,
    INDEX_TYPE periodicinx, INDEX_TYPE periodiciny, INDEX_TYPE periodicinz,
    VALUE_TYPE physlengthx, VALUE_TYPE physlengthy, VALUE_TYPE physlengthz,
    VALUE_TYPE convfactvelcv, VALUE_TYPE convfactpresscp,
    VALUE_TYPE physvxtop,
    INDEX_TYPE step,
    INDEX_TYPE Nnx, INDEX_TYPE Nny, INDEX_TYPE Nnz,
    bool isGPUWorker,
    VALUE_TYPE *rho, VALUE_TYPE *vx, VALUE_TYPE *vy, VALUE_TYPE *vz,
    std::string &filename3dvelocities,
    std::string &filename3dpressure,
    std::string &filename3dxvelvsanalytical,
    std::ofstream &f3dv,
    std::ofstream &f3dp,
    std::ofstream &f3dxvelvsa)
{
    if (isGPUWorker) {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    if (nranks == 1) {
        if (rank != 0) return;

        tecplot3DLinkwiseVelocities(
            convfactvelcv, filename3dvelocities, f3dv,
            Ncx, Ncy, Ncz, NnxG, NnyG, NnzG,
            periodicinx, periodiciny, periodicinz,
            physlengthx, physlengthy, physlengthz,
            step, vx, vy, vz
        );

        tecplot3DLinkwisePressure(
            convfactpresscp, filename3dpressure, f3dp,
            Ncx, Ncy, Ncz, NnxG, NnyG, NnzG,
            periodicinx, periodiciny, periodicinz,
            physlengthx, physlengthy, physlengthz,
            rho, step
        );

        tecplot3DLinkwiseXvelocityVsAnalyticalPlaneCouette(
            convfactvelcv, filename3dxvelvsanalytical, f3dxvelvsa,
            Ncx, Ncy, Ncz, NnxG, NnyG, NnzG,
            periodicinx, periodiciny, periodicinz,
            physlengthx, physlengthy, physlengthz,
            physvxtop, step, vx
        );
        return;
    }

    const INDEX_TYPE localNzInt = zdec.localNzInt;
    const size_t planeSize = (size_t)Nnx * (size_t)Nny;
    const size_t sendElems = (size_t)localNzInt * planeSize;

    std::vector<VALUE_TYPE> send_rho(sendElems);
    std::vector<VALUE_TYPE> send_vx(sendElems);
    std::vector<VALUE_TYPE> send_vy(sendElems);
    std::vector<VALUE_TYPE> send_vz(sendElems);

    #pragma omp parallel for collapse(2)
    for (INDEX_TYPE lk = 1; lk <= localNzInt; lk++) {
        for (INDEX_TYPE j = 0; j < Nny; j++) {
            for (INDEX_TYPE i = 0; i < Nnx; i++) {
                size_t local_ixyz = (size_t)i + (size_t)j * (size_t)Nnx + (size_t)lk * (size_t)Nnx * (size_t)Nny;
                size_t outIdx = (size_t)(lk - 1) * planeSize + (size_t)j * (size_t)Nnx + (size_t)i;
                send_rho[outIdx] = rho[local_ixyz];
                send_vx[outIdx]  = vx[local_ixyz];
                send_vy[outIdx]  = vy[local_ixyz];
                send_vz[outIdx]  = vz[local_ixyz];
            }
        }
    }

    std::vector<int> counts, displs;
    std::vector<VALUE_TYPE> recv_rho, recv_vx, recv_vy, recv_vz;

    int localCount = (int)sendElems;
    std::vector<int> gatheredCounts(nranks, 0);
    MPI_Gather(&localCount, 1, MPI_INT,
               rank == 0 ? gatheredCounts.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        counts = gatheredCounts;
        displs.resize(nranks, 0);
        for (int r = 1; r < nranks; ++r) displs[r] = displs[r - 1] + counts[r - 1];

        const INDEX_TYPE globalNzInt  = (periodicinz == 1) ? (NnzG - 2) : NnzG;
        recv_rho.resize((size_t)globalNzInt * planeSize);
        recv_vx .resize((size_t)globalNzInt * planeSize);
        recv_vy .resize((size_t)globalNzInt * planeSize);
        recv_vz .resize((size_t)globalNzInt * planeSize);
    }

    MPI_Gatherv(send_rho.data(), (int)send_rho.size(), MPI_DOUBLE,
                rank == 0 ? recv_rho.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(send_vx.data(), (int)send_vx.size(), MPI_DOUBLE,
                rank == 0 ? recv_vx.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(send_vy.data(), (int)send_vy.size(), MPI_DOUBLE,
                rank == 0 ? recv_vy.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(send_vz.data(), (int)send_vz.size(), MPI_DOUBLE,
                rank == 0 ? recv_vz.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) return;

    const size_t NtotalG = (size_t)NnxG * (size_t)NnyG * (size_t)NnzG;
    std::vector<VALUE_TYPE> rhoG(NtotalG, 0.0);
    std::vector<VALUE_TYPE> vxG (NtotalG, 0.0);
    std::vector<VALUE_TYPE> vyG (NtotalG, 0.0);
    std::vector<VALUE_TYPE> vzG (NtotalG, 0.0);

    const INDEX_TYPE globalStartK = (periodicinz == 1) ? 1 : 0;
    const INDEX_TYPE globalNzInt  = (periodicinz == 1) ? (NnzG - 2) : NnzG;

    for (INDEX_TYPE kz = 0; kz < globalNzInt; kz++) {
        INDEX_TYPE gk = globalStartK + kz;
        for (INDEX_TYPE j = 0; j < NnyG; j++) {
            for (INDEX_TYPE i = 0; i < NnxG; i++) {
                size_t inIdx = (size_t)kz * planeSize + (size_t)j * (size_t)NnxG + (size_t)i;
                size_t gixyz = (size_t)i + (size_t)j * (size_t)NnxG + (size_t)gk * (size_t)NnxG * (size_t)NnyG;
                rhoG[gixyz] = recv_rho[inIdx];
                vxG[gixyz]  = recv_vx[inIdx];
                vyG[gixyz]  = recv_vy[inIdx];
                vzG[gixyz]  = recv_vz[inIdx];
            }
        }
    }

    tecplot3DLinkwiseVelocities(
        convfactvelcv, filename3dvelocities, f3dv,
        Ncx, Ncy, Ncz, NnxG, NnyG, NnzG,
        periodicinx, periodiciny, periodicinz,
        physlengthx, physlengthy, physlengthz,
        step, vxG.data(), vyG.data(), vzG.data()
    );

    tecplot3DLinkwisePressure(
        convfactpresscp, filename3dpressure, f3dp,
        Ncx, Ncy, Ncz, NnxG, NnyG, NnzG,
        periodicinx, periodiciny, periodicinz,
        physlengthx, physlengthy, physlengthz,
        rhoG.data(), step
    );

    tecplot3DLinkwiseXvelocityVsAnalyticalPlaneCouette(
        convfactvelcv, filename3dxvelvsanalytical, f3dxvelvsa,
        Ncx, Ncy, Ncz, NnxG, NnyG, NnzG,
        periodicinx, periodiciny, periodicinz,
        physlengthx, physlengthy, physlengthz,
        physvxtop, step, vxG.data()
    );
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (rank == 0) {
        std::filesystem::create_directories("perf_outputs");
        std::ofstream clearTiming("perf_outputs/per_rank_timing_summary.txt", std::ios::trunc);
        clearTiming.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    rrc::RuntimeResourceOptions rrc_opt;
    try {
        rrc_opt = rrc::parse_runtime_resource_options(argc, argv);
    } catch (const std::exception& ex) {
        if (rank == 0) {
            std::cerr << "Runtime resource option error: " << ex.what() << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 91);
    }

    rrc::TopologyHandle topoHandle;
    hwloc_topology_t topo = nullptr;
    try {
        topo = topoHandle.get();
    } catch (const std::exception& ex) {
        if (rank == 0) {
            std::cerr << "Topology initialization error: " << ex.what() << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 92);
    }

    std::string cli_gpu = "";
    std::string cli_mode = "";
    std::string cli_hetero_layout = "gpu-gpu-cpu";
    bool cli_gpu_aware_mpi = false;
    int cli_hetero_cpu_planes = -1;
    int cli_nx = -1, cli_ny = -1, cli_nz = -1;

    for (int a = 1; a < argc; a++) {
        std::string arg = argv[a];
        if (arg == "--gpu" && a + 1 < argc) cli_gpu = argv[++a];
        else if (arg == "--mode" && a + 1 < argc) cli_mode = argv[++a];
        else if (arg == "--hetero-layout" && a + 1 < argc) cli_hetero_layout = argv[++a];
        else if (arg == "--gpu-aware-mpi") cli_gpu_aware_mpi = true;
        else if (arg == "--hetero-cpu-planes" && a + 1 < argc) cli_hetero_cpu_planes = std::stoi(argv[++a]);
        else if (arg == "--nx" && a + 1 < argc) cli_nx = std::stoi(argv[++a]);
        else if (arg == "--ny" && a + 1 < argc) cli_ny = std::stoi(argv[++a]);
        else if (arg == "--nz" && a + 1 < argc) cli_nz = std::stoi(argv[++a]);
    }

    if (rank != 0) {
        std::cout.setstate(std::ios_base::failbit);
        std::cerr.setstate(std::ios_base::failbit);
    }

    if (rank == 0) {
        cout << "*******************************************************" << endl;
        cout << "      CULBIBBS: Heterogeneous 3D Plane Couette         " << endl;
        cout << "*******************************************************" << endl;
        cout << "MPI ranks: " << nranks << endl;
        cout << "Runtime resource layer:" << endl;
        cout << "  bind mode      : " << rrc::bind_mode_to_cstr(rrc_opt.bind_mode) << endl;
        cout << "  core type      : " << rrc::core_kind_to_cstr(rrc_opt.core_kind) << endl;
        cout << "  cpu threads    : " << (rrc_opt.cpu_threads > 0 ? std::to_string(rrc_opt.cpu_threads) : std::string("auto")) << endl;
        cout << "  cpu numa       : " << rrc_opt.cpu_numa << endl;
        cout << "  print topology : " << (rrc_opt.print_topology ? "on" : "off") << endl;
        cout << "  print binding  : " << (rrc_opt.print_binding ? "on" : "off") << endl;
        cout << "  print monitor  : " << (rrc_opt.print_monitor ? "on" : "off") << endl;
        cout << "  monitor every  : " << rrc_opt.monitor_every << endl;
        cout << "  gpu-aware mpi  : " << (cli_gpu_aware_mpi ? "on" : "off") << endl;
        if (!rrc_opt.gpu_host_cores.empty()) {
            cout << "  manual gpu host cores: " << rrc::join_int_vector(rrc_opt.gpu_host_cores) << endl;
        }
        if (!rrc_opt.cpu_cores.empty()) {
            cout << "  manual cpu cores     : " << rrc::join_int_vector(rrc_opt.cpu_cores) << endl;
        }
        if (rrc_opt.print_topology) {
            rrc::print_topology_summary(std::cout, topo);
        }
    }

    string inputname = "input/input_plane_couette_3D_CUDA_paramcalc_virtual_nodes.txt";

    VALUE_TYPE physlengthx, physlengthy, physlengthz;
    INDEX_TYPE Ncx, Ncy, Ncz, periodicinx, periodiciny, periodicinz;
    VALUE_TYPE physrho, physvisc, convfactrhocrho;
    VALUE_TYPE physvxback, physvxbottom, physvxfront, physvxleft, physvxright, physvxtop;
    VALUE_TYPE physvyback, physvybottom, physvyfront, physvyleft, physvyright, physvytop;
    VALUE_TYPE physvzback, physvzbottom, physvzfront, physvzleft, physvzright, physvztop;
    VALUE_TYPE tau;
    INDEX_TYPE relaxtype, stype, simtype, delxdim, multitimestepoutput;
    VALUE_TYPE physchardimtouse, physcharvel;
    INDEX_TYPE remoteoutput, objectplaced;
    VALUE_TYPE physobjectcritdim, physobjectlocx, physobjectlocy, physobjectlocz;
    INDEX_TYPE ibnumobjects, ibicosubdivisions;
    vector<VALUE_TYPE> ibphysobjectloc;

    INDEX_TYPE runonichec = 0;
    std::string remoteoutputfolderichecint, remoteoutputfolderichecbatch;
    std::string remoteoutputfolderaltint, remoteoutputfolderaltbatch;

    INDEX_TYPE remotebatch = 0;
    string remotebatchname = "remote_batch.txt";
    readInRemoteBatch<VALUE_TYPE, INDEX_TYPE>(remotebatchname, remotebatch);

    readInData(
        convfactrhocrho, delxdim, ibicosubdivisions, ibnumobjects, ibphysobjectloc,
        inputname, multitimestepoutput, Ncx, Ncy, Ncz, objectplaced,
        periodicinx, periodiciny, periodicinz,
        physchardimtouse, physcharvel,
        physlengthx, physlengthy, physlengthz,
        physobjectcritdim, physobjectlocx, physobjectlocy, physobjectlocz,
        physrho,
        physvxback, physvxbottom, physvxfront, physvxleft, physvxright, physvxtop,
        physvyback, physvybottom, physvyfront, physvyleft, physvyright, physvytop,
        physvzback, physvzbottom, physvzfront, physvzleft, physvzright, physvztop,
        physvisc, relaxtype, remoteoutput, simtype, stype, tau
    );

    HeteroRuntimeConfig<VALUE_TYPE, INDEX_TYPE> hcfg;
    readHeteroRuntimeConfig<VALUE_TYPE, INDEX_TYPE>(inputname, hcfg);

    if (cli_nx > 0 && cli_ny > 0 && cli_nz > 0) {
        if (rank == 0) cout << "Runtime override: Ncells = " << cli_nx << " x " << cli_ny << " x " << cli_nz << endl;
        Ncx = cli_nx; Ncy = cli_ny; Ncz = cli_nz;
    }

    if (hcfg.heterogeneous_mode != 1) {
        if (rank == 0) {
            std::cerr << "ERROR: This updated file expects Heterogeneous_mode: 1 in the input file." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 100);
    }

    if (nranks != 1 && nranks != 2 && nranks != 3) {
        if (rank == 0) {
            std::cerr << "ERROR: Run with 1, 2, or 3 MPI ranks." << std::endl;
            std::cerr << "np=1 -> rank0 GPU0/specified GPU" << std::endl;
            std::cerr << "np=2 -> rank0 GPU0, rank1 GPU1" << std::endl;
            std::cerr << "np=3 -> rank0 GPU0, rank1 GPU1, rank2 CPU" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 101);
    }

    std::vector<int> workerTypes;
    std::vector<int> gpuIds;
    std::vector<double> weights;

    if (nranks == 1) {
        workerTypes = { WORKER_GPU };
        gpuIds      = { hcfg.gpu0_id };
        weights     = { 1.0 };
    } else if (nranks == 2) {
        workerTypes = { WORKER_GPU, WORKER_GPU };
        gpuIds      = { hcfg.gpu0_id, hcfg.gpu1_id };

        weights = {
            (double)hcfg.gpu0_weight,
            (double)hcfg.gpu1_weight
        };
    } else {
        if (cli_hetero_layout == "gpu-gpu-cpu" || cli_hetero_layout == "current") {
            workerTypes = { WORKER_GPU, WORKER_GPU, WORKER_CPU };
            gpuIds      = { hcfg.gpu0_id, hcfg.gpu1_id, -1 };
            weights     = { (double)hcfg.gpu0_weight, (double)hcfg.gpu1_weight, (double)hcfg.cpu_weight };
            cli_hetero_layout = "gpu-gpu-cpu";
        } else if (cli_hetero_layout == "cpu-gpu-gpu") {
            workerTypes = { WORKER_CPU, WORKER_GPU, WORKER_GPU };
            gpuIds      = { -1, hcfg.gpu0_id, hcfg.gpu1_id };
            weights     = { (double)hcfg.cpu_weight, (double)hcfg.gpu0_weight, (double)hcfg.gpu1_weight };
        } else {
            if (rank == 0) {
                std::cerr << "ERROR: Unknown --hetero-layout '" << cli_hetero_layout
                          << "'. Use gpu-gpu-cpu or cpu-gpu-gpu." << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 102);
        }
    }

    const std::vector<double> configuredWeights = weights;
    const bool enable_relative_speed_load_balance = true;

    if (!cli_gpu.empty()) {
        auto cli_gpus = parse_gpu_list(cli_gpu);
        if (nranks == 1 && !cli_gpus.empty()) {
            gpuIds[0] = cli_gpus[0];
        } else {
            const std::vector<int> gpuRanks = gpu_worker_ranks(workerTypes);
            if ((int)cli_gpus.size() >= (int)gpuRanks.size()) {
                for (size_t i = 0; i < gpuRanks.size(); ++i) {
                    gpuIds[gpuRanks[i]] = cli_gpus[i];
                }
            }
        }
    }

    const bool isGPUWorker = (workerTypes[rank] == WORKER_GPU);
    const bool isCPUWorker = (workerTypes[rank] == WORKER_CPU);

    const bool auto_cpu_threads = (rrc_opt.cpu_threads <= 0);
    int effective_cpu_threads = auto_cpu_threads
        ? choose_auto_cpu_thread_limit_for_problem(Ncx, Ncy, Ncz)
        : rrc_opt.cpu_threads;
    if (isCPUWorker) {
        omp_set_num_threads(std::max(1, effective_cpu_threads));
    }

    int gpuid = -1;
    if (isGPUWorker) {
        int devCount = 0;
        cudaGetDeviceCount(&devCount);
        if (devCount <= 0) {
            if (rank == 0) std::cerr << "ERROR: No CUDA devices found.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int chosenGpu = gpuIds[rank];
        if (chosenGpu < 0 || chosenGpu >= devCount) {
            if (rank == 0) {
                std::cerr << "ERROR: Requested GPU " << chosenGpu
                          << " but CUDA reports " << devCount << " device(s).\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        CUDA_SAFE_CALL(cudaSetDevice(chosenGpu));
        cudaGetDevice(&gpuid);
    }

    // ------------------------------------------------------------
    // Runtime resource control: build deterministic binding plan
    // ------------------------------------------------------------
    rrc::BindingDecision bind_plan;
    std::vector<int> reserved_gpu_host_lcores;
    const std::vector<int> gpuRanksForBinding = gpu_worker_ranks(workerTypes);

    if (rrc_opt.bind_mode == rrc::BindMode::Auto) {
        rrc::TopologySummary topo_summary = rrc::query_topology_summary(topo);
        std::vector<int> gpu_host_for_rank(nranks, -1);

        for (int gpuRank : gpuRanksForBinding) {
            std::vector<int> candidates = rrc::query_gpu_nearby_logical_cores_from_nvidia_smi(gpuIds[gpuRank]);
            if (candidates.empty()) candidates = topo_summary.logical_os_ids;

            int chosen = -1;
            for (int c : candidates) {
                if (std::find(reserved_gpu_host_lcores.begin(), reserved_gpu_host_lcores.end(), c) == reserved_gpu_host_lcores.end()) {
                    chosen = c;
                    break;
                }
            }
            if (chosen < 0 && !candidates.empty()) {
                chosen = candidates.front();
            }
            gpu_host_for_rank[gpuRank] = chosen;
            if (chosen >= 0 &&
                std::find(reserved_gpu_host_lcores.begin(), reserved_gpu_host_lcores.end(), chosen) == reserved_gpu_host_lcores.end()) {
                reserved_gpu_host_lcores.push_back(chosen);
            }
        }

        const std::vector<int> reserved_gpu_host_pcores =
            rrc::logical_to_physical_list(topo, reserved_gpu_host_lcores);

        if (isGPUWorker) {
            int chosen_host = gpu_host_for_rank[rank];
            if (chosen_host >= 0) {
                bind_plan.selected_logical_cores = {chosen_host};
                bind_plan.selected_physical_cores = rrc::logical_to_physical_list(topo, bind_plan.selected_logical_cores);
                bind_plan.selected_numa = rrc::find_numa_for_logical_core(topo, chosen_host);
                bind_plan.source = "auto gpu-near host core";
            } else {
                bind_plan.source = "auto no valid gpu host core";
            }
        } else {
            std::vector<int> cpu_pool;
            if (rrc_opt.cpu_numa >= 0) {
                cpu_pool = rrc::get_numa_logical_cores(topo, rrc_opt.cpu_numa);
                bind_plan.selected_numa = rrc_opt.cpu_numa;
                bind_plan.source = "auto cpu numa pool";
            } else {
                cpu_pool = topo_summary.logical_os_ids;
                bind_plan.source = "auto cpu all logical cores";
            }

            for (int c : cpu_pool) {
                const std::vector<int> c_phys_vec = rrc::logical_to_physical_list(topo, {c});
                const int c_phys = c_phys_vec.empty() ? -1 : c_phys_vec.front();
                const bool shares_gpu_host_physical =
                    std::find(reserved_gpu_host_pcores.begin(), reserved_gpu_host_pcores.end(), c_phys) != reserved_gpu_host_pcores.end();
                if (!shares_gpu_host_physical) {
                    bind_plan.selected_logical_cores.push_back(c);
                }
            }

            if (effective_cpu_threads > 0 && (int)bind_plan.selected_logical_cores.size() > effective_cpu_threads) {
                bind_plan.selected_logical_cores.resize(effective_cpu_threads);
            }

            bind_plan.selected_physical_cores = rrc::logical_to_physical_list(topo, bind_plan.selected_logical_cores);
            if (bind_plan.selected_numa < 0 && !bind_plan.selected_logical_cores.empty()) {
                bind_plan.selected_numa = rrc::find_numa_for_logical_core(topo, bind_plan.selected_logical_cores.front());
            }
        }
    } else {
        bind_plan = rrc::decide_binding_for_rank(
            topo, rrc_opt, rank, isGPUWorker, (isGPUWorker ? gpuid : -1), reserved_gpu_host_lcores
        );
    }

    if (rrc_opt.bind_mode != rrc::BindMode::Off && !bind_plan.selected_logical_cores.empty()) {
        rrc::apply_binding_plan(topo, bind_plan, false);
    }

    if (isCPUWorker && !bind_plan.selected_logical_cores.empty()) {
        effective_cpu_threads = std::max(1, std::min(effective_cpu_threads, (int)bind_plan.selected_logical_cores.size()));
        omp_set_num_threads(effective_cpu_threads);
    }

    if (rrc_opt.print_binding) {
        std::ostringstream bss;
        rrc::print_binding_report(bss, rank, isGPUWorker ? "GPU" : "CPU", (isGPUWorker ? gpuid : -1), bind_plan, topo);
        ordered_rank_print(rank, nranks, bss.str());
    }

    VALUE_TYPE physchardim = physchardimtouse;

    INDEX_TYPE NnxG = (periodicinx == 1) ? (Ncx + 2) : Ncx;
    INDEX_TYPE NnyG = (periodiciny == 1) ? (Ncy + 2) : Ncy;
    INDEX_TYPE NnzG = (periodicinz == 1) ? (Ncz + 2) : Ncz;

    ZDecomp zdec{};
    INDEX_TYPE Nnx = NnxG;
    INDEX_TYPE Nny = NnyG;
    INDEX_TYPE Nnz = 0;

    VALUE_TYPE rhoinit, latvisc, physdelxcl, physdeltct, convfactpresscp;
    VALUE_TYPE convfactvelcv;
    VALUE_TYPE latvxback, latvxbottom, latvxfront, latvxleft, latvxright, latvxtop;
    VALUE_TYPE latvyback, latvybottom, latvyfront, latvyleft, latvyright, latvytop;
    VALUE_TYPE latvzback, latvzbottom, latvzfront, latvzleft, latvzright, latvztop;
    VALUE_TYPE latre, vxinit, vyinit, vzinit, timestepsfor1s;
    VALUE_TYPE numoflatupdates;

    string codename = "plane_couette_3D_CUDA_paramcalc_virtual_nodes";
    string testcasename = "plane_couette_";
    string foldername = "plane_couette";

    if (rank == 0) {
        std::string outdir = "../../local_output/" + foldername + "/" + codename;
        fs::create_directories(outdir);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    INDEX_TYPE countsearch = 0;
    string filenameinputdata;
    ofstream finputdata;

    if (rank == 0) {
        filenameinputdata =
            "../../local_output/" + foldername + "/" + codename + "/" + testcasename + to_string(0) + "_inputdata.dat";

        string inputdata = "_inputdata";
        countsearch = fileExists<VALUE_TYPE, INDEX_TYPE>(
            codename, countsearch, inputdata, filenameinputdata,
            foldername, remotebatch, remoteoutput,
            remoteoutputfolderichecint, remoteoutputfolderichecbatch,
            remoteoutputfolderaltint, remoteoutputfolderaltbatch,
            runonichec, testcasename
        );

        cout << "\nFilename for input data: " << filenameinputdata << endl;
    }
    MPI_Bcast(&countsearch, 1, MPI_INT, 0, MPI_COMM_WORLD);

    string filename3dvelocities, filename3dpressure, filename3dxvelvsanalytical;
    ofstream f3dv, f3dp, f3dxvelvsa;

    if (rank == 0) {
        filename3dvelocities =
            "../../local_output/" + foldername + "/" + codename + "/" + testcasename +
            to_string(countsearch) + "_tecplot_3d_velocities.dat";

        filename3dpressure =
            "../../local_output/" + foldername + "/" + codename + "/" + testcasename +
            to_string(countsearch) + "_tecplot_3d_pressure.dat";

        filename3dxvelvsanalytical =
            "../../local_output/" + foldername + "/" + codename + "/" + testcasename +
            to_string(countsearch) + "_tecplot_3d_xvelvsanalytical.dat";

        cout << "\nFilename 1: " << filename3dvelocities << endl;
        cout << "\nFilename 2: " << filename3dpressure << endl;
        cout << "\nFilename 3: " << filename3dxvelvsanalytical << endl << endl;
    }

    paramCalc(
        convfactpresscp, convfactrhocrho, convfactvelcv, delxdim,
        filenameinputdata, finputdata, latvisc,
        latvxback, latvxbottom, latvxfront, latvxleft, latvxright, latvxtop,
        latvyback, latvybottom, latvyfront, latvyleft, latvyright, latvytop,
        latvzback, latvzbottom, latvzfront, latvzleft, latvzright, latvztop,
        latre, Ncx, Ncy, Ncz, numoflatupdates,
        NnxG, NnyG, NnzG, physchardim, physcharvel,
        physdelxcl, physdeltct, physlengthx, physlengthy, physlengthz,
        physrho, physvisc,
        physvxback, physvxbottom, physvxfront, physvxleft, physvxright, physvxtop,
        physvyback, physvybottom, physvyfront, physvyleft, physvyright, physvytop,
        physvzback, physvzbottom, physvzfront, physvzleft, physvzright, physvztop,
        relaxtype, rhoinit, tau, timestepsfor1s, vxinit, vyinit, vzinit
    );

    if (isGPUWorker) {
        INDEX_TYPE Q_h = 19;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(Q, &Q_h, sizeof(INDEX_TYPE)));

        INDEX_TYPE cx_h[19] = {
             0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0
        };
        INDEX_TYPE cy_h[19] = {
             0,  0,  0,  1, -1,  0,  0,  1, -1,  0,  0,  1, -1, -1,  1,  0,  0,  1, -1
        };
        INDEX_TYPE cz_h[19] = {
             0,  0,  0,  0,  0,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0, -1,  1, -1,  1
        };
        VALUE_TYPE w_h[19] = {
            VALUE_TYPE(1.0/3.0),
            VALUE_TYPE(1.0/18.0), VALUE_TYPE(1.0/18.0),
            VALUE_TYPE(1.0/18.0), VALUE_TYPE(1.0/18.0),
            VALUE_TYPE(1.0/18.0), VALUE_TYPE(1.0/18.0),
            VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0),
            VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0),
            VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0),
            VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0),
            VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0),
            VALUE_TYPE(1.0/36.0), VALUE_TYPE(1.0/36.0)
        };
        INDEX_TYPE D_h[19] = {
            19, 2394, 252, 10, 40, 10, 40, 10, 40, 36, 72, 12, 24, 4, 4, 4, 8, 8, 8
        };
        VALUE_TYPE s_h[19];
        hetero_const::fill_s(tau, stype, s_h);

        CUDA_SAFE_CALL(cudaMemcpyToSymbol(cx, cx_h, 19 * sizeof(INDEX_TYPE)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(cy, cy_h, 19 * sizeof(INDEX_TYPE)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(cz, cz_h, 19 * sizeof(INDEX_TYPE)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(w,  w_h,  19 * sizeof(VALUE_TYPE)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(D,  D_h,  19 * sizeof(INDEX_TYPE)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(s,  s_h,  19 * sizeof(VALUE_TYPE)));
    }

    const FixedHeteroProfile fixedProfile = fixed_hetero_profile_for_problem(
        Ncx, Ncy, Ncz, workerTypes, cli_hetero_cpu_planes);

    weights = normalize_worker_weights(configuredWeights, workerTypes);
    if (fixedProfile.active) {
        if (rank == 0) {
            cout << "Fixed hetero profile:" << endl;
            cout << "  profile      : " << fixedProfile.name << endl;
            if (cli_hetero_cpu_planes > 0) {
                cout << "  cpu planes override : " << cli_hetero_cpu_planes << endl;
            }
            cout << "  cpu threads  : " << fixedProfile.cpuThreads << endl;
            for (int r = 0; r < nranks; r++) {
                cout << "  rank " << r << " fixed Z planes=" << fixedProfile.zCounts[r] << endl;
            }
        }
    } else if (enable_relative_speed_load_balance) {
        const INDEX_TYPE NnxCal = std::min<INDEX_TYPE>(NnxG, 64);
        const INDEX_TYPE NnyCal = std::min<INDEX_TYPE>(NnyG, 64);
        const INDEX_TYPE NnzCal = std::max<INDEX_TYPE>(4, std::min<INDEX_TYPE>(NnzG, 18));

        const double localSpeed = calibrate_relative_worker_speed(
            isGPUWorker, NnxCal, NnyCal, NnzCal,
            rhoinit, vxinit, vyinit, vzinit, tau, simtype, relaxtype);

        std::vector<double> measuredSpeeds(nranks, 0.0);
        MPI_Allgather(&localSpeed, 1, MPI_DOUBLE,
                      measuredSpeeds.data(), 1, MPI_DOUBLE,
                      MPI_COMM_WORLD);

        weights = stabilize_relative_weights(measuredSpeeds, configuredWeights, workerTypes);
        weights = cap_cpu_weight_for_problem(weights, workerTypes, Ncx, Ncy, Ncz);

        if (rank == 0) {
            cout << "Relative-speed load balancing weights:" << endl;
            for (int r = 0; r < nranks; r++) {
                cout << "  rank " << r << " speed=" << measuredSpeeds[r]
                     << " weight=" << weights[r] << endl;
            }
        }
    }

    zdec = fixedProfile.active
        ? decompose_z_fixed_counts(rank, nranks, NnzG, periodicinz, fixedProfile.zCounts, workerTypes, gpuIds)
        : decompose_z_weighted(rank, nranks, NnzG, periodicinz, weights, workerTypes, gpuIds);
    if (isCPUWorker) {
        int thread_limit = effective_cpu_threads;
        if (auto_cpu_threads && fixedProfile.active && fixedProfile.cpuThreads > 0) {
            thread_limit = fixedProfile.cpuThreads;
        }
        if (!bind_plan.selected_logical_cores.empty()) {
            thread_limit = std::min(thread_limit, (int)bind_plan.selected_logical_cores.size());
        }

        if (auto_cpu_threads && fixedProfile.active && fixedProfile.cpuThreads > 0) {
            effective_cpu_threads = std::max(1, thread_limit);
        } else if (auto_cpu_threads) {
            effective_cpu_threads = choose_auto_cpu_threads(Ncx, Ncy, Ncz, zdec.localNzInt, thread_limit);
        } else {
            effective_cpu_threads = std::max(1, thread_limit);
        }
        omp_set_num_threads(effective_cpu_threads);
    }
    if (zdec.localNzInt <= 0) {
        if (rank == 0) cerr << "ERROR: Invalid weighted decomposition." << endl;
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    Nnz = zdec.localNzInt + 2;

    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < nranks; r++) {
        if (rank == r) {
            if (workerTypes[rank] == WORKER_GPU) {
                printf("[Rank %d] GPU worker on GPU %d | localStartK=%d localNzInt=%d\n",
                       rank, gpuIds[rank], zdec.localStartK, zdec.localNzInt);
            } else {
                printf("[Rank %d] CPU worker | localStartK=%d localNzInt=%d | CPU_threads=%d%s\n",
                       rank, zdec.localStartK, zdec.localNzInt, effective_cpu_threads,
                       auto_cpu_threads ? " (auto)" : " (manual)");
            }
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        cout << "*******************************************************" << endl;
        cout << "Run layout:" << endl;
        if (!cli_mode.empty()) {
            cout << "  requested mode: " << cli_mode << endl;
        }
        if (nranks == 3) {
            cout << "  hetero layout : " << cli_hetero_layout << endl;
        }
        for (int r = 0; r < nranks; ++r) {
            if (workerTypes[r] == WORKER_GPU) {
                cout << "  rank " << r << " -> GPU " << gpuIds[r] << endl;
            } else {
                cout << "  rank " << r << " -> CPU" << endl;
            }
        }
    }

    size_t Ntotal = (size_t)Nnx * (size_t)Nny * (size_t)Nnz;
    LocalFields fld;

    if (isGPUWorker) allocate_gpu_fields(fld, Ntotal);
    else             allocate_cpu_fields(fld, Ntotal);

    const size_t planeSize = (size_t)Nnx * (size_t)Nny;
    const bool useHalfZHalo = (nranks == 3);
    const int haloDirs = useHalfZHalo ? ZHALF_DIRS : ZDIRS;
    const size_t haloElems = planeSize * (size_t)haloDirs;

    long long local_est_bytes = estimate_local_rank_bytes(Ntotal, haloElems, isGPUWorker);
    {
        std::ostringstream rss;
        rss << "[Rank " << rank << "] role=" << (isGPUWorker ? "GPU" : "CPU")
            << " local_estimated_memory=" << rrc::human_bytes(local_est_bytes)
            << " local_cells=" << Ntotal
            << " halo_dirs=" << haloDirs
            << " halo_elems=" << haloElems << "\n";
        ordered_rank_print(rank, nranks, rss.str());
    }

    VALUE_TYPE *d_send_first = nullptr, *d_send_last = nullptr, *d_recv_prev = nullptr, *d_recv_next = nullptr;
    cudaStream_t haloStream = nullptr;
    cudaStream_t computeStream = nullptr;
    std::vector<VALUE_TYPE> h_send_first, h_send_last, h_recv_prev, h_recv_next;
    VALUE_TYPE *h_send_first_gpu = nullptr;
    VALUE_TYPE *h_send_last_gpu  = nullptr;
    VALUE_TYPE *h_recv_prev_gpu  = nullptr;
    VALUE_TYPE *h_recv_next_gpu  = nullptr;

    if (isGPUWorker) {
        CUDA_SAFE_CALL(cudaStreamCreate(&haloStream));
        CUDA_SAFE_CALL(cudaStreamCreate(&computeStream));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_send_first, haloElems * sizeof(VALUE_TYPE)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_send_last,  haloElems * sizeof(VALUE_TYPE)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_recv_prev,  haloElems * sizeof(VALUE_TYPE)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_recv_next,  haloElems * sizeof(VALUE_TYPE)));

        CUDA_SAFE_CALL(cudaHostAlloc((void**)&h_send_first_gpu, haloElems * sizeof(VALUE_TYPE), cudaHostAllocDefault));
        CUDA_SAFE_CALL(cudaHostAlloc((void**)&h_send_last_gpu,  haloElems * sizeof(VALUE_TYPE), cudaHostAllocDefault));
        CUDA_SAFE_CALL(cudaHostAlloc((void**)&h_recv_prev_gpu,  haloElems * sizeof(VALUE_TYPE), cudaHostAllocDefault));
        CUDA_SAFE_CALL(cudaHostAlloc((void**)&h_recv_next_gpu,  haloElems * sizeof(VALUE_TYPE), cudaHostAllocDefault));
    } else {
        h_send_first.resize(haloElems);
        h_send_last.resize(haloElems);
        h_recv_prev.resize(haloElems);
        h_recv_next.resize(haloElems);
    }

    dim3 dimBlock(8, 8, 8);
    dim3 dimGrid((Nnx + dimBlock.x - 1) / dimBlock.x,
                 (Nny + dimBlock.y - 1) / dimBlock.y,
                 (Nnz + dimBlock.z - 1) / dimBlock.z);

    dim3 block1D(256);
    dim3 grid1D((unsigned int)((planeSize + 255) / 256));

    std::ostringstream gss;
    if (isGPUWorker) {
        cudaDeviceProp prop{};
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, gpuid));
        const int coresPerSM = cuda_cores_per_sm_from_device_prop(prop);
        const int totalCudaCores = prop.multiProcessorCount * coresPerSM;
        const int threadsPerBlock3D = dimBlock.x * dimBlock.y * dimBlock.z;
        const long long gridBlocks3D =
            (long long)dimGrid.x * (long long)dimGrid.y * (long long)dimGrid.z;
        const long long launchedThreads3D = gridBlocks3D * (long long)threadsPerBlock3D;
        const long long logicalCells =
            (long long)Nnx * (long long)Nny * (long long)Nnz;

        gss << "[Rank " << rank << "] GPU device report:\n"
            << "  gpu id                         : " << gpuid << "\n"
            << "  name                           : " << prop.name << "\n"
            << "  compute capability             : sm_" << prop.major << prop.minor << "\n"
            << "  SM count                       : " << prop.multiProcessorCount << "\n"
            << "  estimated CUDA cores / SM      : " << coresPerSM << "\n"
            << "  estimated total CUDA cores     : " << totalCudaCores << "\n"
            << "  global memory                  : " << rrc::human_bytes((long long)prop.totalGlobalMem) << "\n"
            << "  max threads per block          : " << prop.maxThreadsPerBlock << "\n"
            << "  max threads per SM             : " << prop.maxThreadsPerMultiProcessor << "\n"
            << "  main 3D block                  : "
            << dimBlock.x << " x " << dimBlock.y << " x " << dimBlock.z
            << " = " << threadsPerBlock3D << " threads/block\n"
            << "  main 3D grid                   : "
            << dimGrid.x << " x " << dimGrid.y << " x " << dimGrid.z
            << " = " << gridBlocks3D << " blocks\n"
            << "  logical cells in local domain  : " << logicalCells << "\n"
            << "  logical CUDA threads launched  : " << launchedThreads3D << "\n"
            << "  1D halo block                  : " << block1D.x << " threads/block\n"
            << "  1D halo grid                   : " << grid1D.x << " blocks\n";
    }
    ordered_rank_print(rank, nranks, gss.str());

    if (isGPUWorker) {
        initialiseDenVelBasic<<<dimGrid, dimBlock>>>(Nnx, Nny, Nnz, fld.rho, rhoinit, fld.vx, vxinit, fld.vy, vyinit, fld.vz, vzinit);
        CUDA_CHECK_ERROR();

        initialiseEDF<<<dimGrid, dimBlock>>>(fld.f, fld.f_post, Nnx, Nny, Nnz, fld.rho, rhoinit, simtype, tau, fld.vx, fld.vy, fld.vz);
        CUDA_CHECK_ERROR();

        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    } else {
        hetero_host::initialiseDenVelBasic_host<VALUE_TYPE, INDEX_TYPE>(Nnx, Nny, Nnz, fld.rho, rhoinit, fld.vx, vxinit, fld.vy, vyinit, fld.vz, vzinit);
        hetero_host::initialiseEDF_host<VALUE_TYPE, INDEX_TYPE>(fld.f, fld.f_post, Nnx, Nny, Nnz, fld.rho, rhoinit, simtype, tau, fld.vx, fld.vy, fld.vz);
    }

    initialise_error_history(Ntotal, fld.vx, fld.vy, fld.vz, fld.vxerr, fld.vyerr, fld.vzerr, fld.e1, fld.e2);

    gather_and_write_outputs(
        rank, nranks, zdec,
        Ncx, Ncy, Ncz,
        NnxG, NnyG, NnzG,
        periodicinx, periodiciny, periodicinz,
        physlengthx, physlengthy, physlengthz,
        convfactvelcv, convfactpresscp,
        physvxtop, 0,
        Nnx, Nny, Nnz,
        isGPUWorker,
        fld.rho, fld.vx, fld.vy, fld.vz,
        filename3dvelocities, filename3dpressure, filename3dxvelvsanalytical,
        f3dv, f3dp, f3dxvelvsa
    );

    INDEX_TYPE startX = (periodicinx == 1) ? 1 : 0;
    INDEX_TYPE endX   = (periodicinx == 1) ? (Nnx - 1) : Nnx;
    INDEX_TYPE startY = (periodiciny == 1) ? 1 : 0;
    INDEX_TYPE endY   = (periodiciny == 1) ? (Nny - 1) : Nny;
    INDEX_TYPE startZ = 1;
    INDEX_TYPE endZ   = zdec.localNzInt + 1;

    INDEX_TYPE nxI = endX - startX;
    INDEX_TYPE nyI = endY - startY;
    INDEX_TYPE nzI = endZ - startZ;

    INDEX_TYPE M2 = (NnxG - 1) / 2;
    INDEX_TYPE N2 = (NnyG - 1) / 2;
    INDEX_TYPE O2 = (NnzG - 1) / 2;

    double t_compute = 0.0;
    double t_comm    = 0.0;
    double t_io      = 0.0;

    // Classify each rank's neighbours as GPU or CPU once before the loop.
    // Phase 1 of halo exchange handles GPU neighbours (fast).
    // Phase 2 handles CPU neighbours (slow) separately so GPU-GPU exchange
    // is never blocked by the slow CPU in the periodic-Z ring topology.
    const bool prevRankIsCPU = (zdec.prevRank != MPI_PROC_NULL) &&
                                (zdec.prevRank >= 0 && zdec.prevRank < nranks) &&
                                (workerTypes[zdec.prevRank] == WORKER_CPU);
    const bool nextRankIsCPU = (zdec.nextRank != MPI_PROC_NULL) &&
                                (zdec.nextRank >= 0 && zdec.nextRank < nranks) &&
                                (workerTypes[zdec.nextRank] == WORKER_CPU);
    const int gpuPrevRank    = (!prevRankIsCPU) ? zdec.prevRank : MPI_PROC_NULL;
    const int gpuNextRank    = (!nextRankIsCPU) ? zdec.nextRank : MPI_PROC_NULL;
    const int cpuPrevRank    = prevRankIsCPU    ? zdec.prevRank : MPI_PROC_NULL;
    const int cpuNextRank    = nextRankIsCPU    ? zdec.nextRank : MPI_PROC_NULL;
    const bool hasCPUNeighbor = (cpuPrevRank != MPI_PROC_NULL) || (cpuNextRank != MPI_PROC_NULL);
    const bool useMPIHalo = (zdec.prevRank != MPI_PROC_NULL) || (zdec.nextRank != MPI_PROC_NULL);
    const bool localPeriodicZ = (periodicinz == 1) && !useMPIHalo;

    INDEX_TYPE step = 0;
    VALUE_TYPE err = 1.0;
    double wallStart = MPI_Wtime();

    while (err > 1.0e-6) {

        step++;

        double t0_compute = MPI_Wtime();

        if (isGPUWorker) {
            if (relaxtype == 0) {
                collBGK<<<dimGrid, dimBlock>>>(fld.f, fld.f_post, Nnx, Nny, Nnz, fld.rho, rhoinit, simtype, tau, fld.vx, fld.vy, fld.vz);
                CUDA_CHECK_ERROR();
            } else {
                collMRT<<<dimGrid, dimBlock>>>(fld.f, fld.f_post, Nnx, Nny, Nnz, fld.rho, rhoinit, simtype, fld.vx, fld.vy, fld.vz);
                CUDA_CHECK_ERROR();
            }

            if (periodicinx == 1) {
                periodicLr<<<dimGrid, dimBlock>>>(fld.f_post, Nnx, Nny, Nnz);
                CUDA_CHECK_ERROR();
            }
            if (periodiciny == 1) {
                periodicTb<<<dimGrid, dimBlock>>>(fld.f_post, Nnx, Nny, Nnz);
                CUDA_CHECK_ERROR();
            }
            if (localPeriodicZ) {
                periodicFb<<<dimGrid, dimBlock>>>(fld.f_post, Nnx, Nny, Nnz);
                CUDA_CHECK_ERROR();
            }

        } else {
            if (relaxtype == 0) {
                hetero_host::collBGK_host<VALUE_TYPE, INDEX_TYPE>(
                    fld.f, fld.f_post, Nnx, Nny, Nnz, fld.rho, rhoinit, simtype, tau, fld.vx, fld.vy, fld.vz
                );
            } else {
                hetero_host::collMRT_host<VALUE_TYPE, INDEX_TYPE>(
                    fld.f, fld.f_post, Nnx, Nny, Nnz, fld.rho, rhoinit, simtype, tau, fld.vx, fld.vy, fld.vz
                );
            }

            if (periodicinx == 1) hetero_host::periodicLr_host<VALUE_TYPE, INDEX_TYPE>(fld.f_post, Nnx, Nny, Nnz);
            if (periodiciny == 1) hetero_host::periodicTb_host<VALUE_TYPE, INDEX_TYPE>(fld.f_post, Nnx, Nny, Nnz);
            if (localPeriodicZ) hetero_host::periodicFb_host<VALUE_TYPE, INDEX_TYPE>(fld.f_post, Nnx, Nny, Nnz);
        }

        double t1_compute = MPI_Wtime();
        t_compute += (t1_compute - t0_compute);

        double t0_comm = MPI_Wtime();

        const INDEX_TYPE kFirst = 1;
        const INDEX_TYPE kLast  = zdec.localNzInt;

        if (!useMPIHalo) {
            // Single-rank full-domain runs use local periodic/boundary kernels,
            // so no MPI halo pack/copy/exchange is required.
        } else if (isGPUWorker) {

            // Pack both boundary planes.
            // In hetero mode, send only the directions required by each neighbour.
            if (useHalfZHalo) {
                packZPlaneHalfDirs<VALUE_TYPE, INDEX_TYPE>
                    <<<grid1D, block1D, 0, haloStream>>>(fld.f_post, d_send_first, Nnx, Nny, Nnz, kFirst, ZDIR_SET_MINUS);
                packZPlaneHalfDirs<VALUE_TYPE, INDEX_TYPE>
                    <<<grid1D, block1D, 0, haloStream>>>(fld.f_post, d_send_last,  Nnx, Nny, Nnz, kLast,  ZDIR_SET_PLUS);
            } else {
                packZPlaneZDirs<VALUE_TYPE, INDEX_TYPE>
                    <<<grid1D, block1D, 0, haloStream>>>(fld.f_post, d_send_first, Nnx, Nny, Nnz, kFirst);
                packZPlaneZDirs<VALUE_TYPE, INDEX_TYPE>
                    <<<grid1D, block1D, 0, haloStream>>>(fld.f_post, d_send_last,  Nnx, Nny, Nnz, kLast);
            }
            CUDA_CHECK_ERROR();

            // Host-staged MPI is the default. With --gpu-aware-mpi, GPU-GPU
            // neighbours exchange device buffers directly, while CPU neighbours
            // still use host-staged buffers because the CPU rank owns host memory.
            if (cpuPrevRank != MPI_PROC_NULL ||
                (!cli_gpu_aware_mpi && gpuPrevRank != MPI_PROC_NULL)) {
                CUDA_SAFE_CALL(cudaMemcpyAsync(h_send_first_gpu, d_send_first,
                    haloElems * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost, haloStream));
            }
            if (cpuNextRank != MPI_PROC_NULL ||
                (!cli_gpu_aware_mpi && gpuNextRank != MPI_PROC_NULL)) {
                CUDA_SAFE_CALL(cudaMemcpyAsync(h_send_last_gpu, d_send_last,
                    haloElems * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost, haloStream));
            }

            // Overlap interior streaming with halo preparation on separate stream.
            if (zdec.localNzInt > 2) {
                size_t interiorCells = (size_t)Nnx * (size_t)Nny * (size_t)(zdec.localNzInt - 2);
                dim3 sBlock(256), sGrid((unsigned int)((interiorCells + 255) / 256));
                streaming_zrange_kernel<VALUE_TYPE, INDEX_TYPE>
                    <<<sGrid, sBlock, 0, computeStream>>>(
                        fld.f, fld.f_post, Nnx, Nny, Nnz, 2, zdec.localNzInt);
                CUDA_CHECK_ERROR();
            }

            CUDA_SAFE_CALL(cudaStreamSynchronize(haloStream));

            // ---- PHASE 1: exchange with GPU neighbours (fast) ----
            {
                MPI_Request p1[4];
                int n1 = 0;
                if (cli_gpu_aware_mpi) {
                    if (gpuNextRank != MPI_PROC_NULL) {
                        MPI_Irecv(d_recv_next, (int)haloElems, mpi_value_type(), gpuNextRank, 100, MPI_COMM_WORLD, &p1[n1++]);
                        MPI_Isend(d_send_last, (int)haloElems, mpi_value_type(), gpuNextRank, 200, MPI_COMM_WORLD, &p1[n1++]);
                    }
                    if (gpuPrevRank != MPI_PROC_NULL) {
                        MPI_Irecv(d_recv_prev,  (int)haloElems, mpi_value_type(), gpuPrevRank, 200, MPI_COMM_WORLD, &p1[n1++]);
                        MPI_Isend(d_send_first, (int)haloElems, mpi_value_type(), gpuPrevRank, 100, MPI_COMM_WORLD, &p1[n1++]);
                    }
                    if (n1 > 0) MPI_Waitall(n1, p1, MPI_STATUSES_IGNORE);

                    if (gpuPrevRank != MPI_PROC_NULL) {
                        if (useHalfZHalo) {
                            unpackZPlaneHalfDirs<VALUE_TYPE, INDEX_TYPE>
                                <<<grid1D, block1D, 0, haloStream>>>(d_recv_prev, fld.f_post, Nnx, Nny, Nnz, 0, ZDIR_SET_PLUS);
                        } else {
                            unpackZPlaneZDirs<VALUE_TYPE, INDEX_TYPE>
                                <<<grid1D, block1D, 0, haloStream>>>(d_recv_prev, fld.f_post, Nnx, Nny, Nnz, 0);
                        }
                        CUDA_CHECK_ERROR();
                    }
                    if (gpuNextRank != MPI_PROC_NULL) {
                        if (useHalfZHalo) {
                            unpackZPlaneHalfDirs<VALUE_TYPE, INDEX_TYPE>
                                <<<grid1D, block1D, 0, haloStream>>>(d_recv_next, fld.f_post, Nnx, Nny, Nnz, zdec.localNzInt + 1, ZDIR_SET_MINUS);
                        } else {
                            unpackZPlaneZDirs<VALUE_TYPE, INDEX_TYPE>
                                <<<grid1D, block1D, 0, haloStream>>>(d_recv_next, fld.f_post, Nnx, Nny, Nnz, zdec.localNzInt + 1);
                        }
                        CUDA_CHECK_ERROR();
                    }
                    if (n1 > 0) {
                        CUDA_SAFE_CALL(cudaStreamSynchronize(haloStream));
                        CUDA_SAFE_CALL(cudaStreamSynchronize(computeStream));
                    }
                } else {
                    if (gpuNextRank != MPI_PROC_NULL) {
                        MPI_Irecv(h_recv_next_gpu, (int)haloElems, mpi_value_type(), gpuNextRank, 100, MPI_COMM_WORLD, &p1[n1++]);
                        MPI_Isend(h_send_last_gpu, (int)haloElems, mpi_value_type(), gpuNextRank, 200, MPI_COMM_WORLD, &p1[n1++]);
                    }
                    if (gpuPrevRank != MPI_PROC_NULL) {
                        MPI_Irecv(h_recv_prev_gpu,  (int)haloElems, mpi_value_type(), gpuPrevRank, 200, MPI_COMM_WORLD, &p1[n1++]);
                        MPI_Isend(h_send_first_gpu, (int)haloElems, mpi_value_type(), gpuPrevRank, 100, MPI_COMM_WORLD, &p1[n1++]);
                    }
                    if (n1 > 0) MPI_Waitall(n1, p1, MPI_STATUSES_IGNORE);

                    if (gpuPrevRank != MPI_PROC_NULL) {
                        CUDA_SAFE_CALL(cudaMemcpyAsync(d_recv_prev, h_recv_prev_gpu,
                            haloElems * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice, haloStream));
                        if (useHalfZHalo) {
                            unpackZPlaneHalfDirs<VALUE_TYPE, INDEX_TYPE>
                                <<<grid1D, block1D, 0, haloStream>>>(d_recv_prev, fld.f_post, Nnx, Nny, Nnz, 0, ZDIR_SET_PLUS);
                        } else {
                            unpackZPlaneZDirs<VALUE_TYPE, INDEX_TYPE>
                                <<<grid1D, block1D, 0, haloStream>>>(d_recv_prev, fld.f_post, Nnx, Nny, Nnz, 0);
                        }
                        CUDA_CHECK_ERROR();
                    }
                    if (gpuNextRank != MPI_PROC_NULL) {
                        CUDA_SAFE_CALL(cudaMemcpyAsync(d_recv_next, h_recv_next_gpu,
                            haloElems * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice, haloStream));
                        if (useHalfZHalo) {
                            unpackZPlaneHalfDirs<VALUE_TYPE, INDEX_TYPE>
                                <<<grid1D, block1D, 0, haloStream>>>(d_recv_next, fld.f_post, Nnx, Nny, Nnz, zdec.localNzInt + 1, ZDIR_SET_MINUS);
                        } else {
                            unpackZPlaneZDirs<VALUE_TYPE, INDEX_TYPE>
                                <<<grid1D, block1D, 0, haloStream>>>(d_recv_next, fld.f_post, Nnx, Nny, Nnz, zdec.localNzInt + 1);
                        }
                        CUDA_CHECK_ERROR();
                    }
                    if (n1 > 0) {
                        CUDA_SAFE_CALL(cudaStreamSynchronize(haloStream));
                        CUDA_SAFE_CALL(cudaStreamSynchronize(computeStream));
                    }
                }
            }

            // ---- PHASE 2: exchange with CPU neighbour (slow, decoupled) ----
            if (hasCPUNeighbor) {
                MPI_Request p2[4];
                int n2 = 0;
                if (cpuNextRank != MPI_PROC_NULL) {
                    MPI_Irecv(h_recv_next_gpu,  (int)haloElems, mpi_value_type(), cpuNextRank, 100, MPI_COMM_WORLD, &p2[n2++]);
                    MPI_Isend(h_send_last_gpu,  (int)haloElems, mpi_value_type(), cpuNextRank, 200, MPI_COMM_WORLD, &p2[n2++]);
                }
                if (cpuPrevRank != MPI_PROC_NULL) {
                    MPI_Irecv(h_recv_prev_gpu,  (int)haloElems, mpi_value_type(), cpuPrevRank, 200, MPI_COMM_WORLD, &p2[n2++]);
                    MPI_Isend(h_send_first_gpu, (int)haloElems, mpi_value_type(), cpuPrevRank, 100, MPI_COMM_WORLD, &p2[n2++]);
                }
                if (n2 > 0) MPI_Waitall(n2, p2, MPI_STATUSES_IGNORE);

                // H2D and unpack CPU-neighbour halos.
                if (cpuPrevRank != MPI_PROC_NULL) {
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_recv_prev, h_recv_prev_gpu,
                        haloElems * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice, haloStream));
                    unpackZPlaneHalfDirs<VALUE_TYPE, INDEX_TYPE>
                        <<<grid1D, block1D, 0, haloStream>>>(d_recv_prev, fld.f_post, Nnx, Nny, Nnz, 0, ZDIR_SET_PLUS);
                    CUDA_CHECK_ERROR();
                }
                if (cpuNextRank != MPI_PROC_NULL) {
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_recv_next, h_recv_next_gpu,
                        haloElems * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice, haloStream));
                    unpackZPlaneHalfDirs<VALUE_TYPE, INDEX_TYPE>
                        <<<grid1D, block1D, 0, haloStream>>>(d_recv_next, fld.f_post, Nnx, Nny, Nnz, zdec.localNzInt + 1, ZDIR_SET_MINUS);
                    CUDA_CHECK_ERROR();
                }
                CUDA_SAFE_CALL(cudaStreamSynchronize(haloStream));
            }

            CUDA_SAFE_CALL(cudaStreamSynchronize(haloStream));
            CUDA_SAFE_CALL(cudaStreamSynchronize(computeStream));

        } else {
            // CPU WORKER: pack and post ALL MPI ops immediately so messages
            // are already in flight while GPU ranks are running Phase 1.
            if (useHalfZHalo) {
                packZPlaneHalfDirs_host<VALUE_TYPE, INDEX_TYPE>(
                    fld.f_post, h_send_first.data(), Nnx, Nny, Nnz, kFirst, ZDIR_SET_MINUS);
                packZPlaneHalfDirs_host<VALUE_TYPE, INDEX_TYPE>(
                    fld.f_post, h_send_last.data(),  Nnx, Nny, Nnz, kLast,  ZDIR_SET_PLUS);
            } else {
                packZPlaneZDirs_host<VALUE_TYPE, INDEX_TYPE>(fld.f_post, h_send_first.data(), Nnx, Nny, Nnz, kFirst);
                packZPlaneZDirs_host<VALUE_TYPE, INDEX_TYPE>(fld.f_post, h_send_last.data(),  Nnx, Nny, Nnz, kLast);
            }

            MPI_Request reqs[4];
            int nreqs = 0;
            if (zdec.nextRank != MPI_PROC_NULL)
                MPI_Irecv(h_recv_next.data(),  (int)haloElems, mpi_value_type(), zdec.nextRank, 100, MPI_COMM_WORLD, &reqs[nreqs++]);
            if (zdec.prevRank != MPI_PROC_NULL)
                MPI_Irecv(h_recv_prev.data(),  (int)haloElems, mpi_value_type(), zdec.prevRank, 200, MPI_COMM_WORLD, &reqs[nreqs++]);
            if (zdec.prevRank != MPI_PROC_NULL)
                MPI_Isend(h_send_first.data(), (int)haloElems, mpi_value_type(), zdec.prevRank, 100, MPI_COMM_WORLD, &reqs[nreqs++]);
            if (zdec.nextRank != MPI_PROC_NULL)
                MPI_Isend(h_send_last.data(),  (int)haloElems, mpi_value_type(), zdec.nextRank, 200, MPI_COMM_WORLD, &reqs[nreqs++]);

            // Stream interior planes while GPU Phase 1 runs and MPI is in flight.
            if (zdec.localNzInt >= 2)
                streaming_host_zrange(fld.f, fld.f_post, Nnx, Nny, Nnz, 2, zdec.localNzInt);

            if (nreqs > 0) MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

            if (zdec.prevRank != MPI_PROC_NULL) {
                if (useHalfZHalo) {
                    unpackZPlaneHalfDirs_host<VALUE_TYPE, INDEX_TYPE>(
                        h_recv_prev.data(), fld.f_post, Nnx, Nny, Nnz, 0, ZDIR_SET_PLUS);
                } else {
                    unpackZPlaneZDirs_host<VALUE_TYPE, INDEX_TYPE>(h_recv_prev.data(), fld.f_post, Nnx, Nny, Nnz, 0);
                }
            }
            if (zdec.nextRank != MPI_PROC_NULL) {
                if (useHalfZHalo) {
                    unpackZPlaneHalfDirs_host<VALUE_TYPE, INDEX_TYPE>(
                        h_recv_next.data(), fld.f_post, Nnx, Nny, Nnz, zdec.localNzInt + 1, ZDIR_SET_MINUS);
                } else {
                    unpackZPlaneZDirs_host<VALUE_TYPE, INDEX_TYPE>(h_recv_next.data(), fld.f_post, Nnx, Nny, Nnz, zdec.localNzInt + 1);
                }
            }
        }

        double t1_comm = MPI_Wtime();
        t_comm += (t1_comm - t0_comm);

        double t2_compute = MPI_Wtime();

        if (isGPUWorker) {
            if (!useMPIHalo) {
                streaming<<<dimGrid, dimBlock>>>(fld.f, fld.f_post, Nnx, Nny, Nnz);
                CUDA_CHECK_ERROR();
            } else if (zdec.localNzInt > 2) {
                size_t firstBoundaryCells = (size_t)Nnx * (size_t)Nny;
                dim3 streamBlock(256);
                dim3 streamGrid((unsigned int)((firstBoundaryCells + 255) / 256));

                streaming_zrange_kernel<VALUE_TYPE, INDEX_TYPE>
                    <<<streamGrid, streamBlock>>>(
                        fld.f,
                        fld.f_post,
                        Nnx,
                        Nny,
                        Nnz,
                        1,
                        2
                    );

                streaming_zrange_kernel<VALUE_TYPE, INDEX_TYPE>
                    <<<streamGrid, streamBlock>>>(
                        fld.f,
                        fld.f_post,
                        Nnx,
                        Nny,
                        Nnz,
                        zdec.localNzInt,
                        zdec.localNzInt + 1
                    );

                CUDA_CHECK_ERROR();
            } else {
                streaming<<<dimGrid, dimBlock>>>(fld.f, fld.f_post, Nnx, Nny, Nnz);
                CUDA_CHECK_ERROR();
            }

            if (periodicinx == 0) {
                halfwayBouncebackLeft<<<dimGrid, dimBlock>>>(fld.f, fld.f_post, latvxleft, latvyleft, latvzleft, Nnx, Nny, Nnz, fld.rho, fld.vx, fld.vy, fld.vz);
                CUDA_CHECK_ERROR();
                halfwayBouncebackRight<<<dimGrid, dimBlock>>>(fld.f, fld.f_post, latvxright, latvyright, latvzright, Nnx, Nny, Nnz, fld.rho, fld.vx, fld.vy, fld.vz);
                CUDA_CHECK_ERROR();
            }

            if (periodiciny == 0) {
                halfwayBouncebackBottom<<<dimGrid, dimBlock>>>(fld.f, fld.f_post, latvxbottom, latvybottom, latvzbottom, Nnx, Nny, Nnz, fld.rho, fld.vx, fld.vy, fld.vz);
                CUDA_CHECK_ERROR();
                halfwayBouncebackTop<<<dimGrid, dimBlock>>>(fld.f, fld.f_post, latvxtop, latvytop, latvztop, Nnx, Nny, Nnz, fld.rho, fld.vx, fld.vy, fld.vz);
                CUDA_CHECK_ERROR();
            }

            if (periodicinz == 0) {
                bool ownsBack  = (zdec.localStartK == 0);
                bool ownsFront = ((zdec.localStartK + zdec.localNzInt - 1) == (NnzG - 1));

                if (ownsBack) {
                    halfwayBouncebackBack<<<dimGrid, dimBlock>>>(fld.f, fld.f_post, latvxback, latvyback, latvzback, Nnx, Nny, Nnz, fld.rho, fld.vx, fld.vy, fld.vz);
                    CUDA_CHECK_ERROR();
                }
                if (ownsFront) {
                    halfwayBouncebackFront<<<dimGrid, dimBlock>>>(fld.f, fld.f_post, latvxfront, latvyfront, latvzfront, Nnx, Nny, Nnz, fld.rho, fld.vx, fld.vy, fld.vz);
                    CUDA_CHECK_ERROR();
                }
            }

            denVel<<<dimGrid, dimBlock>>>(fld.f, Nnx, Nny, Nnz, fld.rho, rhoinit, simtype, fld.vx, fld.vy, fld.vz);
            CUDA_CHECK_ERROR();
            CUDA_SAFE_CALL(cudaDeviceSynchronize());

        } else {
            if (!useMPIHalo) {
                hetero_host::streaming_host<VALUE_TYPE, INDEX_TYPE>(fld.f, fld.f_post, Nnx, Nny, Nnz);
            } else if (zdec.localNzInt > 2) {
                streaming_host_zrange(fld.f, fld.f_post, Nnx, Nny, Nnz, 1, 2);
                streaming_host_zrange(fld.f, fld.f_post, Nnx, Nny, Nnz, zdec.localNzInt, zdec.localNzInt + 1);
            } else {
                hetero_host::streaming_host<VALUE_TYPE, INDEX_TYPE>(fld.f, fld.f_post, Nnx, Nny, Nnz);
            }

            if (periodicinx == 0) {
                hetero_host::halfwayBouncebackLeft_host<VALUE_TYPE, INDEX_TYPE>(fld.f, fld.f_post, latvxleft, latvyleft, latvzleft, Nnx, Nny, Nnz, fld.rho);
                hetero_host::halfwayBouncebackRight_host<VALUE_TYPE, INDEX_TYPE>(fld.f, fld.f_post, latvxright, latvyright, latvzright, Nnx, Nny, Nnz, fld.rho);
            }

            if (periodiciny == 0) {
                hetero_host::halfwayBouncebackBottom_host<VALUE_TYPE, INDEX_TYPE>(fld.f, fld.f_post, latvxbottom, latvybottom, latvzbottom, Nnx, Nny, Nnz, fld.rho);
                hetero_host::halfwayBouncebackTop_host<VALUE_TYPE, INDEX_TYPE>(fld.f, fld.f_post, latvxtop, latvytop, latvztop, Nnx, Nny, Nnz, fld.rho);
            }

            if (periodicinz == 0) {
                bool ownsBack  = (zdec.localStartK == 0);
                bool ownsFront = ((zdec.localStartK + zdec.localNzInt - 1) == (NnzG - 1));
                if (ownsBack) {
                    hetero_host::halfwayBouncebackBack_host<VALUE_TYPE, INDEX_TYPE>(fld.f, fld.f_post, latvxback, latvyback, latvzback, Nnx, Nny, Nnz, fld.rho);
                }
                if (ownsFront) {
                    hetero_host::halfwayBouncebackFront_host<VALUE_TYPE, INDEX_TYPE>(fld.f, fld.f_post, latvxfront, latvyfront, latvzfront, Nnx, Nny, Nnz, fld.rho);
                }
            }

            hetero_host::denVel_host<VALUE_TYPE, INDEX_TYPE>(fld.f, Nnx, Nny, Nnz, fld.rho, rhoinit, simtype, fld.vx, fld.vy, fld.vz);
        }

        if (step % 1000 == 0) {

            if (isGPUWorker) {
                error<<<dimGrid, dimBlock>>>(fld.e1, fld.e2, Nnx, Nny, Nnz, fld.vxerr, fld.vyerr, fld.vzerr, fld.vx, fld.vy, fld.vz);
                CUDA_CHECK_ERROR();
                CUDA_SAFE_CALL(cudaDeviceSynchronize());
            } else {
                hetero_host::error_host<VALUE_TYPE, INDEX_TYPE>(fld.e1, fld.e2, Nnx, Nny, Nnz, fld.vxerr, fld.vyerr, fld.vzerr, fld.vx, fld.vy, fld.vz);
            }

            VALUE_TYPE local1 = 0.0, local2 = 0.0;
            hetero_host::interior_error_sums_host<VALUE_TYPE, INDEX_TYPE>(
                local1, local2,
                fld.e1, fld.e2,
                Nnx, Nny,
                startX, startY, startZ,
                nxI, nyI, nzI
            );

            VALUE_TYPE global1 = 0.0, global2 = 0.0;
            MPI_Allreduce(&local1, &global1, 1, mpi_value_type(), MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local2, &global2, 1, mpi_value_type(), MPI_SUM, MPI_COMM_WORLD);

            err = global1 / global2;

            int localStartInt = static_cast<int>(zdec.localStartK);
            int localCountInt = static_cast<int>(zdec.localNzInt);
            std::vector<int> allStartInt(nranks, 0);
            std::vector<int> allCountInt(nranks, 0);

            MPI_Allgather(&localStartInt, 1, MPI_INT,
                          allStartInt.data(), 1, MPI_INT,
                          MPI_COMM_WORLD);
            MPI_Allgather(&localCountInt, 1, MPI_INT,
                          allCountInt.data(), 1, MPI_INT,
                          MPI_COMM_WORLD);

            int ownerRank = -1;
            for (int r = 0; r < nranks; r++) {
                INDEX_TYPE gk0 = static_cast<INDEX_TYPE>(allStartInt[r]);
                INDEX_TYPE gk1 = static_cast<INDEX_TYPE>(allStartInt[r] + allCountInt[r] - 1);
                if (O2 >= gk0 && O2 <= gk1) {
                    ownerRank = r;
                    break;
                }
            }

            if (ownerRank < 0) {
                if (rank == 0) {
                    std::cerr << "ERROR: center velocity owner rank not found." << std::endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 120);
            }

            VALUE_TYPE vxc = 0.0, vyc = 0.0, vzc = 0.0;
            if (rank == ownerRank) {
                INDEX_TYPE localK = (O2 - zdec.localStartK) + 1;
                size_t id = (size_t)M2 + (size_t)N2 * (size_t)Nnx + (size_t)localK * (size_t)Nnx * (size_t)Nny;
                vxc = fld.vx[id];
                vyc = fld.vy[id];
                vzc = fld.vz[id];
            }

            MPI_Bcast(&vxc, 1, mpi_value_type(), ownerRank, MPI_COMM_WORLD);
            MPI_Bcast(&vyc, 1, mpi_value_type(), ownerRank, MPI_COMM_WORLD);
            MPI_Bcast(&vzc, 1, mpi_value_type(), ownerRank, MPI_COMM_WORLD);

            if (rank == 0) {
                printf("Err = %e\nvx_c = %e vy_c = %e vz_c = %e ts = %d\n",
                       err, vxc, vyc, vzc, step);
                fflush(stdout);
            }

            if (multitimestepoutput == 1) {
                double io0 = MPI_Wtime();
                gather_and_write_outputs(
                    rank, nranks, zdec,
                    Ncx, Ncy, Ncz,
                    NnxG, NnyG, NnzG,
                    periodicinx, periodiciny, periodicinz,
                    physlengthx, physlengthy, physlengthz,
                    convfactvelcv, convfactpresscp,
                    physvxtop, step,
                    Nnx, Nny, Nnz,
                    isGPUWorker,
                    fld.rho, fld.vx, fld.vy, fld.vz,
                    filename3dvelocities, filename3dpressure, filename3dxvelvsanalytical,
                    f3dv, f3dp, f3dxvelvsa
                );
                double io1 = MPI_Wtime();
                t_io += (io1 - io0);
            }
        }

        if (rrc_opt.print_monitor && (step % std::max<INDEX_TYPE>(1, (INDEX_TYPE)rrc_opt.monitor_every) == 0)) {
            std::ostringstream mss;
            rrc::print_monitor_snapshot(mss, rank, isGPUWorker ? "GPU" : "CPU",
                                        (int)step, isGPUWorker, t_compute, t_comm);
            ordered_rank_print(rank, nranks, mss.str());
        }

        double t3_compute = MPI_Wtime();
        t_compute += (t3_compute - t2_compute);
    }

    double io2 = MPI_Wtime();
    gather_and_write_outputs(
        rank, nranks, zdec,
        Ncx, Ncy, Ncz,
        NnxG, NnyG, NnzG,
        periodicinx, periodiciny, periodicinz,
        physlengthx, physlengthy, physlengthz,
        convfactvelcv, convfactpresscp,
        physvxtop, step,
        Nnx, Nny, Nnz,
        isGPUWorker,
        fld.rho, fld.vx, fld.vy, fld.vz,
        filename3dvelocities, filename3dpressure, filename3dxvelvsanalytical,
        f3dv, f3dp, f3dxvelvsa
    );
    double io3 = MPI_Wtime();
    t_io += (io3 - io2);

    double wallEnd = MPI_Wtime();
    double wallLocal = wallEnd - wallStart;

    double total_local_measured = t_compute + t_comm + t_io;
    if (total_local_measured <= 0.0) total_local_measured = 1e-12;

    double comm_percent_local = 100.0 * t_comm / total_local_measured;
    double avg_step_local = (step > 0) ? (wallLocal / (double)step) : 0.0;

    size_t local_slab_cells = (size_t)Nnx * (size_t)Nny * (size_t)zdec.localNzInt;

    // print per-rank timing summary on screen in rank order
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < nranks; r++) {
        if (rank == r) {
            std::cout << "[Rank " << rank << "] "
                      << (isGPUWorker ? "GPU" : "CPU");
            if (isGPUWorker) std::cout << " gpu=" << gpuid;
            std::cout << " summary:\n";
            std::cout << "  local slab size (Z planes): " << zdec.localNzInt << "\n";
            std::cout << "  local slab cells          : " << local_slab_cells << "\n";
            std::cout << "  compute time             : " << t_compute << " s\n";
            std::cout << "  communication time       : " << t_comm << " s\n";
            std::cout << "  compute+comm time        : " << (t_compute + t_comm) << " s\n";
            std::cout << "  communication percentage : " << comm_percent_local << " %\n";
            std::cout << "  average time per step    : " << avg_step_local << " s\n";
            std::cout << "  wall time (local rank)   : " << wallLocal << " s\n";
            std::cout << std::flush;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // save per-rank timing summary to file in rank order
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < nranks; r++) {
        if (rank == r) {
            append_rank_timing_summary_to_file(
                rank,
                isGPUWorker,
                gpuid,
                zdec,
                Nnx,
                Nny,
                t_compute,
                t_comm,
                t_io,
                wallLocal,
                step
            );
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double t_compute_max = 0.0, t_comm_max = 0.0, t_io_max = 0.0, wall_max = 0.0;
    MPI_Reduce(&t_compute, &t_compute_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm,    &t_comm_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_io,      &t_io_max,      1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&wallLocal, &wall_max,      1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Wall Time = " << wall_max << endl;
        cout << "Final time step: " << step << endl;

        double total = t_compute_max + t_comm_max + t_io_max;
        if (total <= 0.0) total = 1e-12;

        cout << "Timing breakdown (max across ranks):" << endl;
        cout << "  compute: " << t_compute_max << " s (" << (100.0 * t_compute_max / total) << "%)" << endl;
        cout << "  comm:    " << t_comm_max    << " s (" << (100.0 * t_comm_max / total)    << "%)" << endl;
        cout << "  io:      " << t_io_max      << " s (" << (100.0 * t_io_max / total)      << "%)" << endl;
        cout << "  total (compute+comm+io): " << total << " s" << endl;
    }

    if (isGPUWorker) {
        if (computeStream) CUDA_SAFE_CALL(cudaStreamDestroy(computeStream));
        if (haloStream) CUDA_SAFE_CALL(cudaStreamDestroy(haloStream));
        if (h_send_first_gpu) CUDA_SAFE_CALL(cudaFreeHost(h_send_first_gpu));
        if (h_send_last_gpu)  CUDA_SAFE_CALL(cudaFreeHost(h_send_last_gpu));
        if (h_recv_prev_gpu)  CUDA_SAFE_CALL(cudaFreeHost(h_recv_prev_gpu));
        if (h_recv_next_gpu)  CUDA_SAFE_CALL(cudaFreeHost(h_recv_next_gpu));
        if (d_send_first) CUDA_SAFE_CALL(cudaFree(d_send_first));
        if (d_send_last)  CUDA_SAFE_CALL(cudaFree(d_send_last));
        if (d_recv_prev)  CUDA_SAFE_CALL(cudaFree(d_recv_prev));
        if (d_recv_next)  CUDA_SAFE_CALL(cudaFree(d_recv_next));
        free_gpu_fields(fld);
    }

    MPI_Finalize();
    return 0;
}
