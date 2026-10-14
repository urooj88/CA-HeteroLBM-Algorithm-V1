#!/usr/bin/env python3
"""
Unified baseline runner for the LBM Couette executable.

This runner is the first-stage workflow before DVFS/URJA:
  - single-gpu: choose the freest GPU automatically unless --gpus is provided
  - multi-gpu: run on the selected GPU list
  - hetero: run on 2 GPUs plus one CPU worker

It prints and saves a report with mode, problem size, selected resources,
application timing, basic GPU/CPU state, and a coarse bottleneck label.
"""

import argparse
import csv
import json
import os
import re
import resource
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EXE = PROJECT_DIR / "bin" / "main_plane_couette_3D_CUDA_paramcalc_virtual_nodes_mpi_omp_cuda_1dZsv3.x"
RESULTS_DIR = PROJECT_DIR / "baseline_results"
VALUE_BYTES = 8
Q_DIRS = 19


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def estimate_cuda_cores_from_name(name):
    # nvidia-smi does not expose SM count. Keep this small and explicit for
    # systems used in these experiments; the C++ executable prints the device
    # property based report as the authoritative run-time view.
    upper = name.upper()
    if "L40S" in upper:
        return {"sm_count": 142, "cuda_cores_per_sm": 128, "total_cuda_cores": 18176}
    if "L40" in upper:
        return {"sm_count": 142, "cuda_cores_per_sm": 128, "total_cuda_cores": 18176}
    return {"sm_count": None, "cuda_cores_per_sm": None, "total_cuda_cores": None}


def run_cmd(cmd, cwd=None, check=False):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result


def parse_gpu_list(value):
    if value is None or value.strip().lower() == "auto":
        return None
    gpu_ids = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not gpu_ids:
        raise SystemExit("Use --gpus auto, --gpus 0, or --gpus 0,1.")
    return gpu_ids


def query_gpus():
    fields = [
        "index",
        "name",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
        "power.draw",
        "temperature.gpu",
        "clocks.sm",
        "clocks.mem",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        try:
            gpu = {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_MiB": float(parts[2]),
                "memory_total_MiB": float(parts[3]),
                "gpu_util_percent": float(parts[4]),
                "memory_util_percent": float(parts[5]),
                "power_draw_W": float(parts[6]) if parts[6] != "N/A" else None,
                "temperature_C": float(parts[7]) if parts[7] != "N/A" else None,
                "sm_clock_MHz": float(parts[8]) if parts[8] != "N/A" else None,
                "mem_clock_MHz": float(parts[9]) if parts[9] != "N/A" else None,
            }
            gpu.update(estimate_cuda_cores_from_name(gpu["name"]))
            gpus.append(gpu)
        except ValueError:
            continue
    return gpus


def choose_freest_gpus(count):
    gpus = query_gpus()
    if len(gpus) < count:
        raise SystemExit(
            f"Requested {count} GPU(s), but nvidia-smi reported {len(gpus)}. "
            "Check nvidia-smi, or pass explicit GPUs with --gpus 0 or --gpus 0,1."
        )
    ranked = sorted(gpus, key=lambda g: (g["memory_used_MiB"], g["gpu_util_percent"], g["index"]))
    return [g["index"] for g in ranked[:count]]


def human_mib(value_mib):
    if value_mib >= 1024:
        return f"{value_mib / 1024:.2f} GiB"
    return f"{value_mib:.2f} MiB"


def estimate_rank_memory_mib(nx, ny, nz_local_int, periodic_x=1, periodic_y=0):
    nnx = nx + 2 if periodic_x == 1 else nx
    nny = ny + 2 if periodic_y == 1 else ny
    nnz = nz_local_int + 2
    cells = nnx * nny * nnz
    plane_size = nnx * nny
    halo_elems = plane_size * 5

    bytes_needed = 0
    bytes_needed += cells * Q_DIRS * VALUE_BYTES  # f
    bytes_needed += cells * Q_DIRS * VALUE_BYTES  # f_post
    bytes_needed += cells * VALUE_BYTES * 9       # vx,vy,vz,rho,vxerr,vyerr,vzerr,e1,e2
    bytes_needed += halo_elems * VALUE_BYTES * 8  # device send/recv + pinned host staging
    return bytes_needed / (1024.0 * 1024.0)


def split_z_planes_for_memory(mode, nz, gpu_count, hetero_cpu_planes=0):
    if mode == "single-gpu":
        return {"gpu": [nz], "cpu": 0}
    if mode == "multi-gpu":
        base = nz // gpu_count
        extra = nz % gpu_count
        return {"gpu": [base + (1 if i < extra else 0) for i in range(gpu_count)], "cpu": 0}

    if hetero_cpu_planes and hetero_cpu_planes > 0 and gpu_count >= 2:
        cpu_planes = max(1, min(int(hetero_cpu_planes), nz - 2))
        gpu_remaining = nz - cpu_planes
        gpu0 = gpu_remaining // 2
        gpu1 = gpu_remaining - gpu0
        return {"gpu": [gpu0, gpu1], "cpu": cpu_planes}

    # Tuned hetero profiles are fixed for final reproducible measurements.
    if nz == 100 and gpu_count >= 2:
        return {"gpu": [50, 48], "cpu": 2}
    if nz == 150 and gpu_count >= 2:
        return {"gpu": [75, 72], "cpu": 3}
    if nz == 200 and gpu_count >= 2:
        return {"gpu": [99, 97], "cpu": 4}
    if nz == 250 and gpu_count >= 2:
        return {"gpu": [119, 126], "cpu": 5}

    # Other hetero sizes still use runtime relative-speed calibration.
    cpu_fraction = 0.03 if nz > 250 else 0.02
    cpu_planes = max(1, int(round(cpu_fraction * nz))) if nz >= 3 else 0
    gpu_remaining = nz - cpu_planes
    gpu0 = gpu_remaining // 2
    gpu1 = gpu_remaining - gpu0
    return {"gpu": [gpu0, gpu1], "cpu": cpu_planes}


def build_memory_safety_report(args, gpu_ids, gpus, meminfo):
    selected_by_id = {gpu["index"]: gpu for gpu in gpus if gpu["index"] in set(gpu_ids)}
    split = split_z_planes_for_memory(args.mode, args.nz, len(gpu_ids), args.hetero_cpu_planes)
    safety = args.memory_safety_factor

    gpu_reports = []
    ok = True
    reasons = []

    for i, gpu_id in enumerate(gpu_ids):
        gpu = selected_by_id.get(gpu_id)
        if gpu is None:
            ok = False
            reasons.append(f"GPU {gpu_id} not found in nvidia-smi output")
            continue
        local_z = split["gpu"][i] if i < len(split["gpu"]) else 0
        required = estimate_rank_memory_mib(args.nx, args.ny, local_z)
        free = gpu["memory_total_MiB"] - gpu["memory_used_MiB"]
        usable = free * safety
        fits = required <= usable
        if not fits:
            ok = False
            reasons.append(
                f"GPU {gpu_id} needs {human_mib(required)} but only "
                f"{human_mib(usable)} usable free VRAM is allowed"
            )
        gpu_reports.append({
            "gpu_id": gpu_id,
            "local_z_planes_estimate": local_z,
            "estimated_required_MiB": required,
            "free_vram_MiB": free,
            "usable_free_vram_MiB": usable,
            "safety_factor": safety,
            "fits": fits,
        })

    cpu_report = None
    if args.mode == "hetero":
        cpu_required = estimate_rank_memory_mib(args.nx, args.ny, split["cpu"])
        available = meminfo.get("MemAvailable", 0.0)
        usable_ram = available * safety
        cpu_fits = cpu_required <= usable_ram
        if not cpu_fits:
            ok = False
            reasons.append(
                f"CPU worker needs {human_mib(cpu_required)} but only "
                f"{human_mib(usable_ram)} usable system RAM is allowed"
            )
        cpu_report = {
            "local_z_planes_estimate": split["cpu"],
            "estimated_required_MiB": cpu_required,
            "available_ram_MiB": available,
            "usable_available_ram_MiB": usable_ram,
            "safety_factor": safety,
            "fits": cpu_fits,
        }

    return {
        "ok": ok,
        "reasons": reasons,
        "mode": args.mode,
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "gpu_reports": gpu_reports,
        "cpu_report": cpu_report,
    }


def print_memory_safety_report(report):
    print("Memory safety check:")
    print(f"  status        : {'OK' if report['ok'] else 'FAILED'}")
    for gpu in report["gpu_reports"]:
        print(f"  GPU {gpu['gpu_id']}:")
        print(f"    estimated local Z planes : {gpu['local_z_planes_estimate']}")
        print(f"    estimated required VRAM  : {human_mib(gpu['estimated_required_MiB'])}")
        print(f"    free VRAM                : {human_mib(gpu['free_vram_MiB'])}")
        print(f"    usable free VRAM         : {human_mib(gpu['usable_free_vram_MiB'])}")
        print(f"    fits                     : {gpu['fits']}")
    if report["cpu_report"] is not None:
        cpu = report["cpu_report"]
        print("  CPU worker:")
        print(f"    estimated local Z planes : {cpu['local_z_planes_estimate']}")
        print(f"    estimated required RAM   : {human_mib(cpu['estimated_required_MiB'])}")
        print(f"    available RAM            : {human_mib(cpu['available_ram_MiB'])}")
        print(f"    usable available RAM     : {human_mib(cpu['usable_available_ram_MiB'])}")
        print(f"    fits                     : {cpu['fits']}")
    for reason in report["reasons"]:
        print(f"  reason        : {reason}")


def meminfo_snapshot():
    out = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                key, value = line.split(":", 1)
                amount_kib = int(value.strip().split()[0])
                out[key] = amount_kib / 1024.0
    except OSError:
        pass
    return out


def proc_stat_snapshot():
    try:
        with open("/proc/stat", "r") as f:
            first = f.readline().split()
        values = [int(x) for x in first[1:]]
        idle = values[3] + values[4]
        total = sum(values)
        return {"idle": idle, "total": total}
    except OSError:
        return {"idle": 0, "total": 0}


def cpu_util_between(before, after):
    total_delta = after["total"] - before["total"]
    idle_delta = after["idle"] - before["idle"]
    if total_delta <= 0:
        return None
    return 100.0 * (1.0 - idle_delta / total_delta)


def rapl_snapshot():
    readings = {}
    for path in Path("/sys/class/powercap").glob("intel-rapl:*/energy_uj"):
        try:
            name_path = path.parent / "name"
            name = name_path.read_text().strip() if name_path.exists() else path.parent.name
            readings[str(path.parent)] = {
                "name": name,
                "energy_uj": int(path.read_text().strip()),
            }
        except OSError:
            continue
    return readings


def rapl_delta_j(before, after):
    total = 0.0
    details = {}
    for key, b in before.items():
        if key not in after:
            continue
        delta_uj = after[key]["energy_uj"] - b["energy_uj"]
        if delta_uj < 0:
            continue
        delta_j = delta_uj / 1_000_000.0
        details[key] = {"name": b["name"], "energy_J": delta_j}
        total += delta_j
    return total, details


def parse_application_output(stdout):
    def find_float(pattern):
        match = re.search(pattern, stdout, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def find_int(pattern):
        match = re.search(pattern, stdout, re.IGNORECASE)
        return int(match.group(1)) if match else None

    timing = {}
    for name in ("compute", "comm", "io"):
        timing[name + "_s"] = find_float(rf"{name}\s*:\s*([0-9.eE+-]+)\s*s")

    return {
        "wall_time_s": find_float(r"Wall\s*Time\s*=\s*([0-9.eE+-]+)"),
        "final_step": find_int(r"Final\s*time\s*step:\s*([0-9]+)"),
        "final_error": find_float(r"Err\s*=\s*([0-9.eE+-]+)(?![\s\S]*Err\s*=)"),
        "timing": timing,
    }


def classify_bound(parsed, gpu_after):
    timing = parsed.get("timing", {})
    compute = timing.get("compute_s") or 0.0
    comm = timing.get("comm_s") or 0.0
    io = timing.get("io_s") or 0.0
    total = compute + comm + io

    if total > 0 and comm / total >= 0.25:
        return "communication-bound"

    selected = [g for g in gpu_after if g.get("selected")]
    if selected:
        avg_mem_util = sum(g["memory_util_percent"] for g in selected) / len(selected)
        avg_gpu_util = sum(g["gpu_util_percent"] for g in selected) / len(selected)
        if avg_mem_util >= 60 and avg_gpu_util < 90:
            return "memory-bound"
        if avg_gpu_util >= 85:
            return "compute-bound"

    if total > 0 and compute / total >= 0.75:
        return "compute-heavy"
    return "balanced-or-unknown"


def build_command(args, gpu_ids):
    common = [
        "mpirun",
        "--mca", "coll_hcoll_enable", "0",
        "--mca", "pml", "ob1",
        "--mca", "btl", "self,smcuda,vader,tcp",
    ]

    if args.mode == "single-gpu":
        np = 1
        mode_args = ["--mode", "single-gpu", "--gpu", str(gpu_ids[0]), "--bind", args.bind]
    elif args.mode == "multi-gpu":
        np = len(gpu_ids)
        mode_args = ["--mode", "multi-gpu", "--use-cpu", "0", "--gpu", ",".join(map(str, gpu_ids)), "--bind", args.bind]
    else:
        if len(gpu_ids) != 2:
            raise RuntimeError("Current hetero executable supports exactly 2 GPUs + 1 CPU worker.")
        np = 3
        mode_args = [
            "--mode", "hetero", "--use-cpu", "1",
            "--gpu", ",".join(map(str, gpu_ids)),
            "--cpu-threads", str(args.cpu_threads),
            "--hetero-layout", args.hetero_layout,
            "--bind", args.bind,
        ]
        if args.hetero_cpu_planes > 0:
            mode_args.extend(["--hetero-cpu-planes", str(args.hetero_cpu_planes)])

    if args.print_binding:
        mode_args.append("--print-binding")
    if args.print_topology:
        mode_args.append("--print-topology")
    if args.print_monitor:
        mode_args.extend(["--print-monitor", "--monitor-every", str(args.monitor_every)])
    if args.gpu_aware_mpi:
        mode_args.append("--gpu-aware-mpi")

    return common + ["-np", str(np), str(EXE), "--nx", str(args.nx), "--ny", str(args.ny), "--nz", str(args.nz)] + mode_args


def mark_selected(gpus, selected_ids):
    selected = set(selected_ids)
    out = []
    for gpu in gpus:
        item = dict(gpu)
        item["selected"] = item["index"] in selected
        out.append(item)
    return out


def append_csv(summary_path, row):
    exists = summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def stream_pipe_to_log(pipe, log_path, mirror, collected):
    with open(log_path, "w") as log:
        for line in iter(pipe.readline, ""):
            collected.append(line)
            log.write(line)
            log.flush()
            mirror.write(line)
            mirror.flush()
    pipe.close()


def stop_process_group(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait()


def run_process_live(cmd, cwd, env, stdout_log, stderr_log):
    stdout_lines = []
    stderr_lines = []
    interrupted = False

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
        start_new_session=True,
    )

    stdout_thread = threading.Thread(
        target=stream_pipe_to_log,
        args=(proc.stdout, stdout_log, sys.stdout, stdout_lines),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=stream_pipe_to_log,
        args=(proc.stderr, stderr_log, sys.stderr, stderr_lines),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        proc.wait()
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Stopping launched MPI job and preserving logs...")
        stop_process_group(proc)

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    return proc, "".join(stdout_lines), "".join(stderr_lines), interrupted


def main():
    parser = argparse.ArgumentParser(description="Run LBM baseline modes and save resource reports.")
    parser.add_argument("--mode", choices=["single-gpu", "multi-gpu", "hetero"], required=True)
    parser.add_argument("--size", type=int,
                        help="Shortcut for cubic nx=ny=nz. Examples: 100, 150, 200, 250.")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nz", type=int)
    parser.add_argument("--gpus", default="auto",
                        help="GPU list like 0 or 0,1. Default: auto choose freest GPU(s).")
    parser.add_argument("--cpu-threads", type=int, default=0,
                        help="CPU threads for hetero mode. Use 0 for automatic selection from problem size and CPU slab size.")
    parser.add_argument("--hetero-layout", choices=["gpu-gpu-cpu", "cpu-gpu-gpu"], default="gpu-gpu-cpu",
                        help="Z-slab rank order for hetero mode. Default keeps the existing working layout.")
    parser.add_argument("--hetero-cpu-planes", type=int, default=0,
                        help="Hetero only: override CPU Z-plane count for tuning. 0 keeps the fixed/default profile.")
    parser.add_argument("--gpu-aware-mpi", action="store_true",
                        help="Use direct GPU-buffer MPI for GPU-GPU halo exchange. Default uses host-staged MPI.")
    parser.add_argument("--bind", choices=["auto", "manual", "off"], default="auto")
    parser.add_argument("--print-binding", action="store_true")
    parser.add_argument("--print-topology", action="store_true")
    parser.add_argument("--print-monitor", action="store_true")
    parser.add_argument("--monitor-every", type=int, default=1000)
    parser.add_argument("--memory-safety-factor", type=float, default=0.85,
                        help="Fraction of currently free VRAM/RAM allowed for estimated allocation.")
    parser.add_argument("--allow-memory-oversubscribe", action="store_true",
                        help="Print memory warning but do not cancel if estimate exceeds safe memory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the selected command and report path without running the app.")
    args = parser.parse_args()

    if args.size:
        if args.size <= 0:
            raise SystemExit("--size must be a positive integer.")
        args.nx = args.nx or args.size
        args.ny = args.ny or args.size
        args.nz = args.nz or args.size
    if not args.nx or not args.ny or not args.nz:
        raise SystemExit("Use --size 100/150/200/250 or provide --nx --ny --nz.")
    if args.nx <= 0 or args.ny <= 0 or args.nz <= 0:
        raise SystemExit("--nx, --ny, and --nz must be positive integers.")

    requested = parse_gpu_list(args.gpus)
    if requested is None:
        gpu_count = 1 if args.mode == "single-gpu" else 2
        gpu_ids = choose_freest_gpus(gpu_count)
        gpu_reason = "auto-selected by lowest VRAM usage, then lowest GPU utilization"
    else:
        gpu_ids = requested
        gpu_reason = "user-specified"

    if args.mode == "single-gpu" and len(gpu_ids) != 1:
        raise SystemExit("single-gpu mode needs exactly one GPU.")
    if args.mode in ("multi-gpu", "hetero") and len(gpu_ids) < 2:
        raise SystemExit(f"{args.mode} mode needs at least two GPUs.")

    gpus_now = query_gpus()
    mem_now = meminfo_snapshot()
    memory_report = build_memory_safety_report(args, gpu_ids, gpus_now, mem_now)

    cmd = build_command(args, gpu_ids)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"{timestamp()}_{args.mode}_{args.nx}x{args.ny}x{args.nz}"
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("LBM baseline run")
    print("================")
    print(f"Mode          : {args.mode}")
    print(f"Problem size  : {args.nx} x {args.ny} x {args.nz}")
    if args.mode == "hetero":
        print(f"Hetero layout : {args.hetero_layout}")
    print(f"Selected GPUs : {','.join(map(str, gpu_ids))}")
    print(f"GPU reason    : {gpu_reason}")
    print(f"Results dir   : {out_dir}")
    print_memory_safety_report(memory_report)
    print("Command       :")
    print("  " + " ".join(cmd))

    if not memory_report["ok"] and not args.allow_memory_oversubscribe:
        with open(out_dir / "memory_rejected_report.json", "w") as f:
            json.dump({
                "run_id": run_id,
                "created_at": datetime.now().isoformat(),
                "mode": args.mode,
                "hetero_layout": args.hetero_layout if args.mode == "hetero" else None,
                "nx": args.nx,
                "ny": args.ny,
                "nz": args.nz,
                "gpu_ids": gpu_ids,
                "gpu_selection_reason": gpu_reason,
                "memory_safety": memory_report,
                "command": cmd,
                "status": "cancelled_before_launch",
            }, f, indent=4)
        print("\nRun cancelled before launch: memory safety check failed.")
        print(f"Rejected report: {out_dir / 'memory_rejected_report.json'}")
        raise SystemExit(2)

    if args.dry_run:
        print("\nDry run only. Application was not launched.")
        return

    gpu_before = mark_selected(query_gpus(), gpu_ids)
    mem_before = meminfo_snapshot()
    stat_before = proc_stat_snapshot()
    rapl_before = rapl_snapshot()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)

    t0 = time.time()
    env = os.environ.copy()
    env["OMPI_MCA_mpi_yield_when_idle"] = "1"
    env["OMP_WAIT_POLICY"] = "passive"
    env["GOMP_SPINCOUNT"] = "0"

    stdout_log = out_dir / "app_stdout.log"
    stderr_log = out_dir / "app_stderr.log"
    proc, stdout, stderr, interrupted = run_process_live(
        cmd, PROJECT_DIR, env, stdout_log, stderr_log
    )
    t1 = time.time()

    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    gpu_after = mark_selected(query_gpus(), gpu_ids)
    mem_after = meminfo_snapshot()
    stat_after = proc_stat_snapshot()
    rapl_after = rapl_snapshot()
    cpu_energy_j, cpu_energy_detail = rapl_delta_j(rapl_before, rapl_after)

    parsed = parse_application_output(stdout)
    bound = classify_bound(parsed, gpu_after)
    cpu_util = cpu_util_between(stat_before, stat_after)
    max_rss_delta_kib = usage_after.ru_maxrss - usage_before.ru_maxrss

    report = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "mode": args.mode,
        "hetero_layout": args.hetero_layout if args.mode == "hetero" else None,
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "gpu_ids": gpu_ids,
        "gpu_selection_reason": gpu_reason,
        "cpu_threads": args.cpu_threads if args.mode == "hetero" else 0,
        "command": cmd,
        "returncode": proc.returncode,
        "status": "interrupted" if interrupted else "finished",
        "python_wall_time_s": t1 - t0,
        "application": parsed,
        "bound_type": bound,
        "memory_safety": memory_report,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "meminfo_before_MiB": mem_before,
        "meminfo_after_MiB": mem_after,
        "cpu_util_percent_during_run": cpu_util,
        "child_max_rss_delta_kib": max_rss_delta_kib,
        "cpu_rapl_energy_J": cpu_energy_j,
        "cpu_rapl_domains": cpu_energy_detail,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=4)

    csv_row = {
        "run_id": run_id,
        "mode": args.mode,
        "hetero_layout": args.hetero_layout if args.mode == "hetero" else "",
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "gpu_ids": ",".join(map(str, gpu_ids)),
        "returncode": proc.returncode,
        "wall_time_s": parsed["wall_time_s"],
        "python_wall_time_s": round(t1 - t0, 6),
        "final_step": parsed["final_step"],
        "final_error": parsed["final_error"],
        "bound_type": bound,
        "cpu_util_percent": None if cpu_util is None else round(cpu_util, 2),
        "cpu_rapl_energy_J": round(cpu_energy_j, 6),
        "report_json": str(out_dir / "report.json"),
    }
    if not interrupted:
        append_csv(RESULTS_DIR / "baseline_summary.csv", csv_row)

    print("\nRun interrupted" if interrupted else "\nRun finished")
    print("===============" if interrupted else "============")
    print(f"Return code     : {proc.returncode}")
    print(f"App wall time   : {parsed['wall_time_s']}")
    print(f"Final step      : {parsed['final_step']}")
    print(f"Final error     : {parsed['final_error']}")
    print(f"Bound type      : {bound}")
    print(f"Report JSON     : {out_dir / 'report.json'}")
    if interrupted:
        print(f"Partial stdout  : {stdout_log}")
        print(f"Partial stderr  : {stderr_log}")
    else:
        print(f"Summary CSV     : {RESULTS_DIR / 'baseline_summary.csv'}")

    if interrupted:
        raise SystemExit(130)

    if proc.returncode != 0:
        print("\nApplication stderr tail:")
        print(stderr[-2000:])
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
