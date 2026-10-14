#ifndef HETERO_OPENH_PARTITION_H
#define HETERO_OPENH_PARTITION_H

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

struct HeteroCliOptions {
    bool hetero_on = false;
    double cpu_weight = 0.05;
    int cpu_threads = 16;
    bool verbose = true;
};

struct WorkerPartition {
    int worker_id = -1;          // 0=gpu0, 1=gpu1, 2=cpu
    bool is_gpu = false;
    int gpu_id = -1;

    long long globalStartK = 0;
    long long globalNzInt = 0;

    long long localStartK = 0;
    long long localNzInt = 0;
};

struct OpenHBindingPlan {
    int main_lcore = 0;
    int gpu0_lcore = 2;
    int gpu1_lcore = 3;
    std::vector<int> cpu_lcores;
};

static inline HeteroCliOptions parse_hetero_cli_options(int argc, char** argv) {
    HeteroCliOptions opt;

    for (int a = 1; a < argc; a++) {
        std::string arg = argv[a];

        if (arg == "--hetero" && a + 1 < argc) {
            std::string v = argv[++a];
            std::transform(v.begin(), v.end(), v.begin(), ::tolower);
            opt.hetero_on = (v == "on" || v == "1" || v == "true" || v == "yes");
        } else if (arg == "--cpu-weight" && a + 1 < argc) {
            opt.cpu_weight = std::stod(argv[++a]);
        } else if (arg == "--cpu-threads" && a + 1 < argc) {
            opt.cpu_threads = std::stoi(argv[++a]);
        } else if (arg == "--hetero-verbose" && a + 1 < argc) {
            std::string v = argv[++a];
            std::transform(v.begin(), v.end(), v.begin(), ::tolower);
            opt.verbose = (v == "on" || v == "1" || v == "true" || v == "yes");
        }
    }

    if (opt.cpu_weight < 0.0) opt.cpu_weight = 0.0;
    if (opt.cpu_weight > 0.40) opt.cpu_weight = 0.40;
    if (opt.cpu_threads <= 0) opt.cpu_threads = 1;

    return opt;
}

static inline std::vector<long long> partition_three_way_planes(
    long long globalNzInt,
    double cpu_weight
) {
    if (globalNzInt < 3) {
        throw std::runtime_error("Need at least 3 interior Z planes for 2GPU+CPU heterogeneous run.");
    }

    const double gpu_weight_each = (1.0 - cpu_weight) * 0.5;

    long long cpu_planes  = (long long)std::llround((double)globalNzInt * cpu_weight);
    long long gpu0_planes = (long long)std::llround((double)globalNzInt * gpu_weight_each);
    long long gpu1_planes = (long long)std::llround((double)globalNzInt * gpu_weight_each);

    cpu_planes  = std::max<long long>(1, cpu_planes);
    gpu0_planes = std::max<long long>(1, gpu0_planes);
    gpu1_planes = std::max<long long>(1, gpu1_planes);

    long long sum = cpu_planes + gpu0_planes + gpu1_planes;

    while (sum > globalNzInt) {
        if (gpu0_planes >= gpu1_planes && gpu0_planes >= cpu_planes && gpu0_planes > 1) gpu0_planes--;
        else if (gpu1_planes >= gpu0_planes && gpu1_planes >= cpu_planes && gpu1_planes > 1) gpu1_planes--;
        else if (cpu_planes > 1) cpu_planes--;
        sum = cpu_planes + gpu0_planes + gpu1_planes;
    }

    while (sum < globalNzInt) {
        if (gpu1_planes <= gpu0_planes) gpu1_planes++;
        else gpu0_planes++;
        sum = cpu_planes + gpu0_planes + gpu1_planes;
    }

    return {gpu0_planes, gpu1_planes, cpu_planes};
}

static inline std::vector<WorkerPartition> build_hetero_partitions(
    long long NnzG,
    int periodicinz,
    double cpu_weight
) {
    std::vector<WorkerPartition> parts(3);

    const long long globalStartK = (periodicinz == 1) ? 1 : 0;
    const long long globalNzInt  = (periodicinz == 1) ? (NnzG - 2) : NnzG;

    auto sizes = partition_three_way_planes(globalNzInt, cpu_weight);

    long long start = globalStartK;

    parts[0].worker_id = 0;
    parts[0].is_gpu = true;
    parts[0].gpu_id = 0;
    parts[0].globalStartK = globalStartK;
    parts[0].globalNzInt = globalNzInt;
    parts[0].localStartK = start;
    parts[0].localNzInt = sizes[0];
    start += sizes[0];

    parts[1].worker_id = 1;
    parts[1].is_gpu = true;
    parts[1].gpu_id = 1;
    parts[1].globalStartK = globalStartK;
    parts[1].globalNzInt = globalNzInt;
    parts[1].localStartK = start;
    parts[1].localNzInt = sizes[1];
    start += sizes[1];

    parts[2].worker_id = 2;
    parts[2].is_gpu = false;
    parts[2].gpu_id = -1;
    parts[2].globalStartK = globalStartK;
    parts[2].globalNzInt = globalNzInt;
    parts[2].localStartK = start;
    parts[2].localNzInt = sizes[2];

    return parts;
}

static inline OpenHBindingPlan build_default_openh_binding_plan(
    int cpu_threads
) {
    OpenHBindingPlan plan;
    plan.main_lcore = 0;
    plan.gpu0_lcore = 2;
    plan.gpu1_lcore = 3;

    plan.cpu_lcores.clear();
    int l = 4;
    while ((int)plan.cpu_lcores.size() < cpu_threads) {
        plan.cpu_lcores.push_back(l);
        l += 2;
    }
    return plan;
}

#endif
