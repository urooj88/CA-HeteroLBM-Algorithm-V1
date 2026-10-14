#ifndef HETERO_OPENH_PARTITION_H
#define HETERO_OPENH_PARTITION_H

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <cmath>

#include "hetero_openh_runtime.h"

struct HeteroSplitResult {
    int gpu0_planes = 0;
    int gpu1_planes = 0;
    int cpu_planes  = 0;
    int total_planes = 0;
};

static inline int clamp_int_value(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline HeteroSplitResult build_2gpu_cpu_split(
    int totalInteriorPlanes,
    double cpuWeight,
    bool heteroOn
) {
    HeteroSplitResult s{};
    s.total_planes = totalInteriorPlanes;

    if (totalInteriorPlanes <= 0) {
        return s;
    }

    if (!heteroOn) {
        int a = totalInteriorPlanes / 2;
        int b = totalInteriorPlanes - a;
        s.gpu0_planes = a;
        s.gpu1_planes = b;
        s.cpu_planes  = 0;
        return s;
    }

    double cw = cpuWeight;
    if (cw < 0.0) cw = 0.0;
    if (cw > 0.40) cw = 0.40;

    int cpuPlanes = (int)std::llround(cw * (double)totalInteriorPlanes);

    if (cpuPlanes < 1 && cw > 0.0 && totalInteriorPlanes >= 3) {
        cpuPlanes = 1;
    }

    cpuPlanes = clamp_int_value(cpuPlanes, 0, totalInteriorPlanes);

    int gpuPlanes = totalInteriorPlanes - cpuPlanes;
    int gpu0 = gpuPlanes / 2;
    int gpu1 = gpuPlanes - gpu0;

    s.gpu0_planes = gpu0;
    s.gpu1_planes = gpu1;
    s.cpu_planes  = cpuPlanes;

    return s;
}

static inline void print_hetero_split_summary(
    const HeteroSplitResult& s,
    bool rank0_print
) {
    if (!rank0_print) return;

    std::cout << "Heterogeneous split summary:\n";
    std::cout << "  total interior planes : " << s.total_planes << "\n";
    std::cout << "  gpu0 planes           : " << s.gpu0_planes << "\n";
    std::cout << "  gpu1 planes           : " << s.gpu1_planes << "\n";
    std::cout << "  cpu planes            : " << s.cpu_planes << "\n";
}

#endif
