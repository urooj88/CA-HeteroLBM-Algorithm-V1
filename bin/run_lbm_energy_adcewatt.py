#!/usr/bin/env python3
"""
Measure LBM energy with the Adcewatt physical power meter.

This wrapper is for the clean baseline workflow:
  - single-gpu
  - multi-gpu
  - hetero

For each requested mode/size it starts the wattmeter, runs run_lbm_baseline.py,
stops the wattmeter, and saves stdout/stderr, raw power CSV, and JSON/CSV
energy summaries.
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BASELINE_RUNNER = SCRIPT_DIR / "run_lbm_baseline.py"
RESULTS_DIR = SCRIPT_DIR / "energy_results"
BASELINE_CACHE_FILE = RESULTS_DIR / "lbm_baseline_power_cache.json"
WATTMETER = Path("/home/urooj/Adcewatts-stool/wattmetre-readmv2new")
WATTMETER_DEVICE = "/dev/ttyUSB0"
BASELINE_MAX_AGE_HOURS = 24


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_sizes(value):
    sizes = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        size = int(item)
        if size <= 0:
            raise argparse.ArgumentTypeError("sizes must be positive integers")
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("provide at least one size")
    return sizes


def parse_modes(mode):
    if mode == "all":
        return ["single-gpu", "multi-gpu", "hetero"]
    return [mode]


def run_cmd(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def read_baseline_from_existing_file():
    baseline_file = SCRIPT_DIR / "all_powerresults_adcewatt_varstd.txt"
    if not baseline_file.exists():
        return None
    with open(baseline_file, "r") as f:
        for line in f:
            if "Average Power Consumption" in line:
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if numbers:
                    return float(numbers[-1])
    return None


def check_baseline_age(cached):
    try:
        created = datetime.fromisoformat(cached["created_at"])
        age = datetime.now() - created
        if age > timedelta(hours=BASELINE_MAX_AGE_HOURS):
            hours_old = age.total_seconds() / 3600.0
            print(
                f"WARNING: cached baseline is {hours_old:.1f} hours old. "
                "Use --new-baseline for a fresh idle-power measurement."
            )
    except Exception:
        pass


def measure_baseline(force_new):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if BASELINE_CACHE_FILE.exists() and not force_new:
        with open(BASELINE_CACHE_FILE, "r") as f:
            cached = json.load(f)
        print("\nUsing cached idle baseline power")
        print(f"  baseline power : {cached['baseline_power_watts']:.2f} W")
        print(f"  cached at      : {cached.get('created_at', 'unknown')}")
        check_baseline_age(cached)
        return cached

    print("\nMeasuring idle baseline power")
    print("=============================")
    cmd = [
        "python3",
        str(SCRIPT_DIR / "run_basepower_adcewatt_var_std.py"),
        "--repetitions", "12",
        "--measurement-time", "10",
        "--sleep", "2",
    ]
    result = run_cmd(cmd, cwd=SCRIPT_DIR)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("idle baseline power measurement failed")

    baseline_power = read_baseline_from_existing_file()
    if baseline_power is None:
        raise RuntimeError("could not read idle baseline power")

    baseline_time_s = 12 * 10
    data = {
        "baseline_power_watts": baseline_power,
        "baseline_measurement_time_seconds": baseline_time_s,
        "baseline_measurement_energy_joules": baseline_power * baseline_time_s,
        "baseline_measurement_energy_wh": (baseline_power * baseline_time_s) / 3600.0,
        "created_at": datetime.now().isoformat(),
    }
    with open(BASELINE_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=4)
    return data


def start_power_monitor(raw_csv_file):
    if not WATTMETER.exists():
        raise FileNotFoundError(f"wattmeter executable not found: {WATTMETER}")

    output_file = open(raw_csv_file, "w")
    process = subprocess.Popen(
        [str(WATTMETER), "-t", WATTMETER_DEVICE],
        stdout=output_file,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(1.0)
    if process.poll() is not None:
        output_file.close()
        raise RuntimeError("power monitor stopped unexpectedly")
    return process, output_file


def stop_power_monitor(process, output_file):
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    output_file.close()


def parse_float_from_output(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_int_from_output(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def read_power_csv(csv_file):
    with open(csv_file, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"power CSV has no samples: {csv_file}")
    return rows


def as_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def process_power_data(csv_file, fallback_duration_s):
    rows = read_power_csv(csv_file)
    required = ["#activepow8", "#activepow9"]
    for col in required:
        if col not in rows[0]:
            raise RuntimeError(f"power CSV missing column {col}; columns are {list(rows[0].keys())}")

    time_col = None
    for candidate in ("time", "timestamp", "#time", "#timestamp", "Time", "Timestamp"):
        if candidate in rows[0]:
            time_col = candidate
            break

    valid_samples = []
    for row in rows:
        p8 = as_float(row.get("#activepow8"))
        p9 = as_float(row.get("#activepow9"))
        if p8 is not None and p9 is not None:
            sample = {"power": p8 + p9}
            if time_col is not None:
                sample["time"] = as_float(row.get(time_col))
            valid_samples.append(sample)

    total_power = [sample["power"] for sample in valid_samples]

    if not total_power:
        raise RuntimeError("power CSV has no valid power samples")

    total_power_w = sum(total_power) / len(total_power)
    min_power = min(total_power)
    max_power = max(total_power)
    var_power = sum((x - total_power_w) ** 2 for x in total_power) / max(1, len(total_power) - 1)
    std_power = var_power ** 0.5

    integrated_energy_j = None
    integrated_duration_s = None
    integration_method = "mean power x application wall time"
    if time_col is not None:
        times = [sample.get("time") for sample in valid_samples]
        if all(t is not None for t in times) and len(times) >= 2:
            t0 = times[0]
            times = [t - t0 for t in times]
            energy = 0.0
            for i in range(1, len(times)):
                dt = times[i] - times[i - 1]
                if dt >= 0:
                    energy += 0.5 * (total_power[i] + total_power[i - 1]) * dt
            integrated_energy_j = energy
            integrated_duration_s = times[-1] - times[0]
            integration_method = "trapezoidal integration from wattmeter timestamps"

    if integrated_energy_j is None:
        integrated_energy_j = total_power_w * fallback_duration_s
        integrated_duration_s = fallback_duration_s

    return {
        "total_power_watts": total_power_w,
        "mean_power_watts": total_power_w,
        "min_power_watts": min_power,
        "max_power_watts": max_power,
        "std_power_watts": std_power,
        "var_power": var_power,
        "sample_count": len(total_power),
        "total_energy_joules": integrated_energy_j,
        "total_energy_wh": integrated_energy_j / 3600.0,
        "integrated_duration_seconds": integrated_duration_s,
        "integration_method": integration_method,
    }


def build_lbm_command(mode, size, gpus, cpu_threads, hetero_layout,
                      hetero_cpu_planes, gpu_aware_mpi, extra_args):
    cmd = [
        "python3",
        str(BASELINE_RUNNER),
        "--mode", mode,
        "--size", str(size),
    ]
    if gpus:
        cmd.extend(["--gpus", gpus])
    if mode == "hetero":
        cmd.extend(["--cpu-threads", str(cpu_threads)])
        cmd.extend(["--hetero-layout", hetero_layout])
        if hetero_cpu_planes > 0:
            cmd.extend(["--hetero-cpu-planes", str(hetero_cpu_planes)])
        if gpu_aware_mpi:
            cmd.append("--gpu-aware-mpi")
    cmd.extend(extra_args)
    return cmd


def run_one(mode, size, args, baseline_data, batch_dir):
    label = f"{mode}_{size}x{size}x{size}"
    run_dir = batch_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_power_file = run_dir / "raw_power.csv"
    stdout_file = run_dir / "app_stdout.log"
    stderr_file = run_dir / "app_stderr.log"
    summary_file = run_dir / "energy_summary.json"

    cmd = build_lbm_command(
        mode, size, args.gpus, args.cpu_threads,
        args.hetero_layout, args.hetero_cpu_planes,
        args.gpu_aware_mpi, args.extra_lbm_arg)

    print("\nEnergy measurement")
    print("==================")
    print(f"Mode       : {mode}")
    print(f"Size       : {size} x {size} x {size}")
    print(f"Run dir    : {run_dir}")
    print("Command    :")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return {
            "mode": mode,
            "size": size,
            "status": "dry-run",
            "command": cmd,
            "run_dir": str(run_dir),
        }

    monitor_process = None
    monitor_file = None
    try:
        monitor_process, monitor_file = start_power_monitor(raw_power_file)
        time.sleep(args.pre_sleep)

        t0 = time.time()
        result = run_cmd(cmd, cwd=PROJECT_DIR)
        t1 = time.time()
    finally:
        if monitor_process is not None and monitor_file is not None:
            stop_power_monitor(monitor_process, monitor_file)

    stdout_file.write_text(result.stdout)
    stderr_file.write_text(result.stderr)

    app_wall_time = parse_float_from_output(r"App\s+wall\s+time\s*:\s*([0-9.eE+-]+)", result.stdout)
    if app_wall_time is None:
        app_wall_time = parse_float_from_output(r"Wall\s*Time\s*=\s*([0-9.eE+-]+)", result.stdout)
    python_wall_time = t1 - t0
    execution_time = app_wall_time if app_wall_time is not None else python_wall_time

    power = process_power_data(raw_power_file, execution_time)
    baseline_power = baseline_data["baseline_power_watts"]
    baseline_energy_j = baseline_power * execution_time
    total_energy_j = power["total_energy_joules"]
    dynamic_energy_j = total_energy_j - baseline_energy_j
    total_power_w = power["total_power_watts"]
    dynamic_power_w = total_power_w - baseline_power

    report_json = None
    report_match = re.search(r"Report\s+JSON\s*:\s*(\S+)", result.stdout)
    if report_match:
        report_json = report_match.group(1)

    summary = {
        "created_at": datetime.now().isoformat(),
        "mode": mode,
        "size": size,
        "nx": size,
        "ny": size,
        "nz": size,
        "command": cmd,
        "returncode": result.returncode,
        "application_wall_time_seconds": app_wall_time,
        "python_wall_time_seconds": python_wall_time,
        "execution_time_used_seconds": execution_time,
        "final_step": parse_int_from_output(r"Final\s+step\s*:\s*([0-9]+)", result.stdout),
        "final_error": parse_float_from_output(r"Final\s+error\s*:\s*([0-9.eE+-]+)", result.stdout),
        "bound_type": (re.search(r"Bound\s+type\s*:\s*(.+)", result.stdout) or [None, None])[1],
        "baseline_power_watts": baseline_power,
        "total_power_watts": total_power_w,
        "total_mean_power_watts": total_power_w,
        "dynamic_power_watts": dynamic_power_w,
        "baseline_energy_joules": baseline_energy_j,
        "baseline_energy_wh": baseline_energy_j / 3600.0,
        "total_energy_joules": total_energy_j,
        "total_energy_wh": total_energy_j / 3600.0,
        "dynamic_energy_joules": dynamic_energy_j,
        "dynamic_energy_wh": dynamic_energy_j / 3600.0,
        "power": power,
        "lbm_report_json": report_json,
        "files": {
            "stdout": str(stdout_file),
            "stderr": str(stderr_file),
            "raw_power_csv": str(raw_power_file),
            "summary_json": str(summary_file),
        },
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    print("\nRun energy result")
    print("=================")
    print(f"Return code    : {result.returncode}")
    print(f"Wall time      : {execution_time:.3f} s")
    print(f"Total power    : {total_power_w:.2f} W")
    print(f"Dynamic power  : {dynamic_power_w:.2f} W")
    print(f"Total energy   : {total_energy_j / 3600.0:.6f} Wh ({total_energy_j:.2f} J)")
    print(f"Dynamic energy : {dynamic_energy_j / 3600.0:.6f} Wh ({dynamic_energy_j:.2f} J)")
    print(f"Summary JSON   : {summary_file}")

    return summary


def append_batch_csv(batch_dir, rows):
    csv_file = batch_dir / "lbm_energy_summary.csv"
    fields = [
        "mode", "size", "returncode", "application_wall_time_seconds",
        "execution_time_used_seconds", "final_step", "final_error", "bound_type",
        "baseline_power_watts", "total_power_watts", "total_mean_power_watts", "dynamic_power_watts",
        "total_energy_joules", "total_energy_wh", "dynamic_energy_joules",
        "dynamic_energy_wh", "summary_json", "lbm_report_json",
    ]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "mode": row.get("mode"),
                "size": row.get("size"),
                "returncode": row.get("returncode"),
                "application_wall_time_seconds": row.get("application_wall_time_seconds"),
                "execution_time_used_seconds": row.get("execution_time_used_seconds"),
                "final_step": row.get("final_step"),
                "final_error": row.get("final_error"),
                "bound_type": row.get("bound_type"),
                "baseline_power_watts": row.get("baseline_power_watts"),
                "total_power_watts": row.get("total_power_watts"),
                "total_mean_power_watts": row.get("total_mean_power_watts"),
                "dynamic_power_watts": row.get("dynamic_power_watts"),
                "total_energy_joules": row.get("total_energy_joules"),
                "total_energy_wh": row.get("total_energy_wh"),
                "dynamic_energy_joules": row.get("dynamic_energy_joules"),
                "dynamic_energy_wh": row.get("dynamic_energy_wh"),
                "summary_json": row.get("files", {}).get("summary_json"),
                "lbm_report_json": row.get("lbm_report_json"),
            })
    return csv_file


def main():
    parser = argparse.ArgumentParser(description="Measure LBM energy with Adcewatt for single/multi/hetero modes.")
    parser.add_argument("--mode", choices=["single-gpu", "multi-gpu", "hetero", "all"], required=True)
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("100,150,200,250"),
                        help="Comma-separated cubic sizes, for example 100,150,200,250.")
    parser.add_argument("--gpus", default=None,
                        help="Optional GPU list passed to run_lbm_baseline.py, for example 0 or 0,1.")
    parser.add_argument("--cpu-threads", type=int, default=0,
                        help="CPU threads for hetero mode. 0 uses the application's automatic size-based rule.")
    parser.add_argument("--hetero-layout", choices=["gpu-gpu-cpu", "cpu-gpu-gpu"], default="gpu-gpu-cpu",
                        help="Z-slab rank order for hetero mode.")
    parser.add_argument("--hetero-cpu-planes", type=int, default=0,
                        help="Hetero only: override CPU Z-plane count for tuning. 0 keeps the fixed/default profile.")
    parser.add_argument("--gpu-aware-mpi", action="store_true",
                        help="Use direct GPU-buffer MPI for GPU-GPU halo exchange. Default uses host-staged MPI.")
    parser.add_argument("--new-baseline", action="store_true",
                        help="Measure fresh idle power before the application runs.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned measurements without starting the wattmeter or application.")
    parser.add_argument("--pre-sleep", type=float, default=0.5,
                        help="Seconds to wait after starting meter before launching the app.")
    parser.add_argument("--extra-lbm-arg", action="append", default=[],
                        help="Extra argument forwarded to run_lbm_baseline.py. Repeat for multiple args.")
    args = parser.parse_args()

    batch_dir = RESULTS_DIR / f"lbm_energy_{timestamp()}_{args.mode}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    baseline_data = {
        "baseline_power_watts": 0.0,
        "created_at": datetime.now().isoformat(),
    } if args.dry_run else measure_baseline(force_new=args.new_baseline)

    rows = []
    for mode in parse_modes(args.mode):
        for size in args.sizes:
            try:
                rows.append(run_one(mode, size, args, baseline_data, batch_dir))
            except Exception as exc:
                print(f"\nFAILED: mode={mode} size={size}: {exc}")
                if not args.continue_on_error:
                    raise

    csv_file = append_batch_csv(batch_dir, rows) if rows else None
    print("\nBatch complete")
    print("==============")
    print(f"Batch dir   : {batch_dir}")
    if csv_file:
        print(f"Summary CSV : {csv_file}")


if __name__ == "__main__":
    main()
