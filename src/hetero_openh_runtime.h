#ifndef HETERO_OPENH_RUNTIME_H
#define HETERO_OPENH_RUNTIME_H

#include <vector>
#include <string>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <cstring>
#include <iostream>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <openh.h>

struct HeteroCliOptions {
    bool hetero_on = false;
    double cpu_weight = 0.10;
    int cpu_threads = -1;
    bool bind_openh = true;
};

struct LocalGeom {
    int Nnx = 0;
    int Nny = 0;
    int Nnz = 0;
    size_t Ntotal = 0;
};

struct LocalPartition {
    int localNzInt = 0;
    int localStartK = 0;
    int globalStartK = 0;
    int globalNzInt = 0;
};

struct OpenHBindingPlan {
    int main_lcore = -1;
    int gpu0_lcore = -1;
    int gpu1_lcore = -1;
    std::vector<int> cpu_lcores;
};

enum WorkerKind {
    WORKER_GPU = 0,
    WORKER_CPU = 1
};

struct WorkerState {
    WorkerKind kind = WORKER_GPU;
    int worker_id = -1;
    int gpu_id = -1;
    int cpu_component_id = 0;
    int cpu_threads = 1;

    LocalGeom geom;
    LocalPartition part;

    VALUE_TYPE* f = nullptr;
    VALUE_TYPE* f_post = nullptr;
    VALUE_TYPE* vx = nullptr;
    VALUE_TYPE* vy = nullptr;
    VALUE_TYPE* vz = nullptr;
    VALUE_TYPE* rho = nullptr;
    VALUE_TYPE* vxerr = nullptr;
    VALUE_TYPE* vyerr = nullptr;
    VALUE_TYPE* vzerr = nullptr;
    VALUE_TYPE* e1 = nullptr;
    VALUE_TYPE* e2 = nullptr;

    VALUE_TYPE* d_send_first = nullptr;
    VALUE_TYPE* d_send_last  = nullptr;
    VALUE_TYPE* d_recv_prev  = nullptr;
    VALUE_TYPE* d_recv_next  = nullptr;

    size_t planeSize = 0;
    size_t haloElems = 0;

    double local_err_num = 0.0;
    double local_err_den = 0.0;

    double center_vx = 0.0;
    double center_vy = 0.0;
    double center_vz = 0.0;

    double t_compute = 0.0;
    double t_comm = 0.0;
};

static inline void pin_current_thread_to_lcores(const std::vector<int>& lcores) {
    if (lcores.empty()) return;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (int c : lcores) {
        if (c >= 0) CPU_SET(c, &cpuset);
    }

    int rc = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "WARNING: sched_setaffinity failed for current thread\n";
    }
}

static inline OpenHBindingPlan build_openh_binding_plan_for_2gpus_plus_cpu(
    int cpu_threads_requested
) {
    OpenHBindingPlan plan;

    openh_init();

    plan.main_lcore = 0;

    int gpu0 = openh_get_unique_lcore(0);
    openh_assign_acc_lcpuids(0, &gpu0, 1);
    plan.gpu0_lcore = gpu0;

    int gpu1 = openh_get_unique_lcore(1);
    openh_assign_acc_lcpuids(1, &gpu1, 1);
    plan.gpu1_lcore = gpu1;

    int total_lcores = openh_get_num_lcores();

    int want_threads = cpu_threads_requested;
    if (want_threads <= 0) {
        want_threads = std::max(1, total_lcores - 3);
    }

    for (int c = 0; c < total_lcores && (int)plan.cpu_lcores.size() < want_threads; c++) {
        if (c == plan.main_lcore) continue;
        if (c == plan.gpu0_lcore) continue;
        if (c == plan.gpu1_lcore) continue;
        plan.cpu_lcores.push_back(c);
    }

    if (!plan.cpu_lcores.empty()) {
        openh_assign_cpu_lcpuids(0, plan.cpu_lcores.data(), (int)plan.cpu_lcores.size());
    }

    return plan;
}

static inline void openh_setup_bindings_for_2gpu_plus_cpu(
    const HeteroCliOptions& cli,
    OpenHBindingPlan& plan,
    bool rank0_print
) {
    if (!cli.bind_openh) return;

    openh_assign_main_lcpuid(plan.main_lcore);
    openh_bind_main_self();

    if (rank0_print) {
        std::cout << "OpenH bindings:\n";
        std::cout << "  main lcore : " << plan.main_lcore << "\n";
        std::cout << "  gpu0 lcore : " << plan.gpu0_lcore << "\n";
        std::cout << "  gpu1 lcore : " << plan.gpu1_lcore << "\n";
        std::cout << "  cpu lcores : ";
        for (size_t i = 0; i < plan.cpu_lcores.size(); i++) {
            std::cout << plan.cpu_lcores[i] << " ";
        }
        std::cout << "\n";
        openh_print();
    }
}

static inline void openh_finalize_if_needed(bool enabled) {
    if (enabled) {
        openh_finalize();
    }
}

#endif
