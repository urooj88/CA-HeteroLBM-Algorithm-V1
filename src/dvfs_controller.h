// dvfs_controller.h — Adaptive DVFS controller for L40S GPU
//
// Automatically selects one of three runtime modes:
//
//   ADAPTIVE_NVML  — nvmlDeviceSetGpuLockedClocks() works (admin NVML perms).
//                    Per-step SM clock switching: high during compute, low during MPI.
//
//   STATIC_SUDO    — NVML clock lock fails but sudo nvidia-smi -lgc works.
//                    Locks SM clock ONCE at compute-optimal level for whole run.
//                    enter_barrier / enter_compute only track timing (no clock switch).
//                    Requires: sudo credential cache (run any sudo cmd <15 min before sim).
//
//   DISABLED       — Neither works. All functions are no-ops; stats still printed.
//
// Clock targets for L40S (Ada Lovelace, sm_89):
//   Compute: DVFS_SM_COMPUTE_MHZ (default 1305) — memory-bandwidth-optimal for LBM
//   Barrier: DVFS_SM_BARRIER_MHZ (default  405) — minimum, GPU idles during MPI wait
//   Memory:  9001 MHz — GDDR6 max, left to driver (nvmlDeviceSetGpuLockedClocks
//            only controls SM clock; memory auto-stays at P0 during compute)
//
// Build:  make USE_NVML=1    (links -lnvidia-ml, enables NVML path)
//         make USE_NVML=0    (no NVML; STATIC_SUDO or DISABLED at runtime)

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <time.h>

#ifdef USE_NVML
#include <nvml.h>
#endif

// ─── Tuneable clock targets ────────────────────────────────────────────────
// L40S SM range: ~300 – 2520 MHz.
// 1305 MHz gives ~95% of peak memory bandwidth with ~48% less SM dynamic power
// than 2520 MHz — ideal for memory-bound LBM collision+streaming kernels.
#ifndef DVFS_SM_COMPUTE_MHZ
#define DVFS_SM_COMPUTE_MHZ  1305u
#endif

// Minimum supported SM clock — GPU is fully idle during MPI Waitall.
#ifndef DVFS_SM_BARRIER_MHZ
#define DVFS_SM_BARRIER_MHZ  405u
#endif
// ──────────────────────────────────────────────────────────────────────────

namespace dvfs {

enum class Mode { DISABLED, STATIC_SUDO, ADAPTIVE_NVML };

struct Stats {
    long long n_compute_switches = 0;
    long long n_barrier_switches = 0;
    double    t_in_compute_mode  = 0.0;
    double    t_in_barrier_mode  = 0.0;
    double    t_switch_overhead  = 0.0;
};

struct Controller {
    Mode mode            = Mode::DISABLED;
    int  gpu_id          = -1;
    bool in_barrier_mode = false;
    Stats stats;
    double _phase_start  = 0.0;

#ifdef USE_NVML
    nvmlDevice_t device      = nullptr;
    unsigned int sm_compute  = DVFS_SM_COMPUTE_MHZ;
    unsigned int sm_barrier  = DVFS_SM_BARRIER_MHZ;
#endif

    static double now_sec() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec + ts.tv_nsec * 1e-9;
    }
};

static Controller g_ctrl;

// ─── Internal: sudo nvidia-smi clock lock/unlock ──────────────────────────
// Uses "sudo -n" (non-interactive) so it never hangs waiting for a password.
// Works only if sudo credentials are cached (user ran any sudo cmd recently).

static bool _sudo_lgc(int gpuid, unsigned int sm_mhz) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "sudo -n nvidia-smi -lgc %u,%u -i %d > /dev/null 2>&1",
             sm_mhz, sm_mhz, gpuid);
    return system(cmd) == 0;
}

static bool _sudo_rgc(int gpuid) {
    char cmd[128];
    snprintf(cmd, sizeof(cmd),
             "sudo -n nvidia-smi -rgc -i %d > /dev/null 2>&1", gpuid);
    return system(cmd) == 0;
}

// ─── init ─────────────────────────────────────────────────────────────────
// Call once after cudaSetDevice(), before the main simulation loop.
// gpuid: the CUDA device index owned by this MPI rank.
inline void init(int gpuid) {
    g_ctrl.gpu_id        = gpuid;
    g_ctrl.in_barrier_mode = false;
    g_ctrl._phase_start  = Controller::now_sec();
    g_ctrl.mode          = Mode::DISABLED;

#ifdef USE_NVML
    // ── Try NVML path (requires admin clock permissions) ──────────────────
    nvmlReturn_t r = nvmlInit_v2();
    if (r == NVML_SUCCESS) {
        r = nvmlDeviceGetHandleByIndex((unsigned int)gpuid, &g_ctrl.device);
    }
    if (r == NVML_SUCCESS) {
        // nvmlDeviceSetGpuLockedClocks is the Ada Lovelace replacement for
        // the deprecated nvmlDeviceSetApplicationsClocks.
        r = nvmlDeviceSetGpuLockedClocks(g_ctrl.device,
                                         g_ctrl.sm_compute,
                                         g_ctrl.sm_compute);
    }
    if (r == NVML_SUCCESS) {
        g_ctrl.mode = Mode::ADAPTIVE_NVML;
        fprintf(stderr,
            "[DVFS gpu=%d] Mode: ADAPTIVE_NVML  "
            "compute=%u MHz  barrier=%u MHz\n",
            gpuid, g_ctrl.sm_compute, g_ctrl.sm_barrier);
        return;
    }

    // NVML failed — explain why and fall through to sudo path.
    fprintf(stderr,
        "[DVFS gpu=%d] NVML clock lock failed: %s\n"
        "             Trying sudo nvidia-smi -lgc fallback...\n",
        gpuid, nvmlErrorString(r));
    if (g_ctrl.device) {
        nvmlShutdown();
        g_ctrl.device = nullptr;
    }
#endif

    // ── Try sudo nvidia-smi -lgc (static lock for whole run) ─────────────
    // Locks SM clock at compute-optimal level. No per-step switching in this
    // mode — enter_barrier / enter_compute only track timing statistics.
    if (_sudo_lgc(gpuid, DVFS_SM_COMPUTE_MHZ)) {
        g_ctrl.mode = Mode::STATIC_SUDO;
        fprintf(stderr,
            "[DVFS gpu=%d] Mode: STATIC_SUDO  SM locked at %u MHz for whole run.\n"
            "             (Per-step switching unavailable without NVML perms.)\n"
            "             Timing stats still collected for barrier/compute phases.\n",
            gpuid, (unsigned)DVFS_SM_COMPUTE_MHZ);
        return;
    }

    // ── Both paths failed ─────────────────────────────────────────────────
    fprintf(stderr,
        "[DVFS gpu=%d] DISABLED — neither NVML nor sudo nvidia-smi -lgc worked.\n"
        "             To enable: ask sysadmin for NVML clock permissions, OR\n"
        "             run 'sudo nvidia-smi -pm 1' and any sudo cmd within 15 min\n"
        "             of starting the simulation (to cache sudo credentials).\n",
        gpuid);
}

// ─── enter_barrier ────────────────────────────────────────────────────────
// Call just before MPI Irecv/Isend/Waitall on exchange steps.
// ADAPTIVE: drops SM to minimum while GPU idles during MPI.
// STATIC / DISABLED: tracks timing only.
inline void enter_barrier() {
    double t0 = Controller::now_sec();

    if (!g_ctrl.in_barrier_mode) {
        g_ctrl.stats.t_in_compute_mode += (t0 - g_ctrl._phase_start);
        g_ctrl._phase_start = t0;
    }

#ifdef USE_NVML
    if (g_ctrl.mode == Mode::ADAPTIVE_NVML && !g_ctrl.in_barrier_mode) {
        nvmlDeviceSetGpuLockedClocks(g_ctrl.device,
                                     g_ctrl.sm_barrier,
                                     g_ctrl.sm_barrier);
        g_ctrl.stats.t_switch_overhead += (Controller::now_sec() - t0);
    }
#endif

    g_ctrl.in_barrier_mode = true;
    g_ctrl.stats.n_barrier_switches++;
}

// ─── enter_compute ────────────────────────────────────────────────────────
// Call after MPI Waitall returns (halo exchange complete).
// ADAPTIVE: restores SM to compute-optimal clock.
// STATIC / DISABLED: tracks timing only.
inline void enter_compute() {
    double t0 = Controller::now_sec();

    if (g_ctrl.in_barrier_mode) {
        g_ctrl.stats.t_in_barrier_mode += (t0 - g_ctrl._phase_start);
        g_ctrl._phase_start = t0;
    }

#ifdef USE_NVML
    if (g_ctrl.mode == Mode::ADAPTIVE_NVML && g_ctrl.in_barrier_mode) {
        nvmlDeviceSetGpuLockedClocks(g_ctrl.device,
                                     g_ctrl.sm_compute,
                                     g_ctrl.sm_compute);
        g_ctrl.stats.t_switch_overhead += (Controller::now_sec() - t0);
    }
#endif

    g_ctrl.in_barrier_mode = false;
    g_ctrl.stats.n_compute_switches++;
}

// ─── shutdown ─────────────────────────────────────────────────────────────
// Call after the main loop. Resets GPU clocks and prints phase statistics.
inline void shutdown(int mpi_rank) {
    double now = Controller::now_sec();

    if (g_ctrl.in_barrier_mode)
        g_ctrl.stats.t_in_barrier_mode += (now - g_ctrl._phase_start);
    else
        g_ctrl.stats.t_in_compute_mode += (now - g_ctrl._phase_start);

    // Reset clocks.
#ifdef USE_NVML
    if (g_ctrl.mode == Mode::ADAPTIVE_NVML && g_ctrl.device) {
        nvmlDeviceResetGpuLockedClocks(g_ctrl.device);
        nvmlShutdown();
        g_ctrl.device = nullptr;
    }
#endif
    if (g_ctrl.mode == Mode::STATIC_SUDO) {
        _sudo_rgc(g_ctrl.gpu_id);
    }

    // Print stats (one GPU rank is enough).
    if (mpi_rank == 0 || mpi_rank == 1) {
        double total = g_ctrl.stats.t_in_compute_mode + g_ctrl.stats.t_in_barrier_mode;
        double barrier_frac = (total > 0.0) ? g_ctrl.stats.t_in_barrier_mode / total : 0.0;

        const char* mode_str =
            (g_ctrl.mode == Mode::ADAPTIVE_NVML) ? "ADAPTIVE_NVML" :
            (g_ctrl.mode == Mode::STATIC_SUDO)   ? "STATIC_SUDO"   : "DISABLED";

        // Linear estimate: SM dynamic power ∝ SM frequency.
        double clk_ratio = (double)DVFS_SM_BARRIER_MHZ / (double)DVFS_SM_COMPUTE_MHZ;
        double sm_saved_pct =
            (g_ctrl.mode == Mode::ADAPTIVE_NVML)
                ? barrier_frac * (1.0 - clk_ratio) * 100.0
                : 0.0;   // static mode: no per-step switching, savings tracked separately

        fprintf(stderr,
            "\n[DVFS gpu=%d] ── Summary (mode: %s) ──────────────────────\n"
            "  Compute-mode time     : %7.2f s  (SM=%u MHz target)\n"
            "  Barrier-mode time     : %7.2f s  (SM=%u MHz target)\n"
            "  Barrier fraction      : %5.1f%%  (comm overhead of total runtime)\n"
            "  Clock switch overhead : %7.3f ms  (total NVML call time)\n"
            "  n_compute_switches    : %lld\n"
            "  n_barrier_switches    : %lld\n"
            "  Est. SM energy saved  : ~%.1f%% of total SM energy\n"
            "     (linear model: P_SM ∝ f_SM; based on barrier fraction × clock ratio)\n",
            g_ctrl.gpu_id, mode_str,
            g_ctrl.stats.t_in_compute_mode, (unsigned)DVFS_SM_COMPUTE_MHZ,
            g_ctrl.stats.t_in_barrier_mode, (unsigned)DVFS_SM_BARRIER_MHZ,
            barrier_frac * 100.0,
            g_ctrl.stats.t_switch_overhead * 1000.0,
            g_ctrl.stats.n_compute_switches,
            g_ctrl.stats.n_barrier_switches,
            sm_saved_pct
        );

        if (g_ctrl.mode == Mode::STATIC_SUDO) {
            fprintf(stderr,
                "  [STATIC mode: SM clock was locked at %u MHz for whole run.\n"
                "   Energy saving vs full boost (2520 MHz) ≈ %.1f%% of SM power.\n"
                "   For adaptive per-step switching, request NVML admin permissions.]\n",
                (unsigned)DVFS_SM_COMPUTE_MHZ,
                (1.0 - (double)DVFS_SM_COMPUTE_MHZ / 2520.0) * 100.0);
        }
    }
}

} // namespace dvfs
