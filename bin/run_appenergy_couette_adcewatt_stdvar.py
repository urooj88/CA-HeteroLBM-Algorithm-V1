## run if use already saved power --> python3 measure_energy.py --app-type first
## if use fresh baseline power run -->> python3 measure_energy.py --app-type first --new-baseline
#!/usr/bin/env python3

import subprocess
import time
import pandas as pd
import numpy as np
import os
import argparse
import json
import re
import shutil
from pathlib import Path
from datetime import datetime, timedelta


RESULTS_DIR = Path("energy_results")
BASELINE_CACHE_FILE = RESULTS_DIR / "baseline_cache.json"

# How old a cached baseline can be before warning (hours)
BASELINE_MAX_AGE_HOURS = 24

SCRIPT_MAP = {
    "first":  "./run_cuda_main_couette1.sh",
    "second": "./run_cuda_main_couette2.sh",
    "third":  "./run_cuda_main_couette3.sh",
    "fourth": "./run_cuda_main_couette4.sh",
    "fifth":  "./run_cuda_main_couette5.sh",
    "sixth":  "./run_cuda_main_couette6.sh",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_wall_time(output_text):
    """
    Parse 'Wall Time = 3.5' from application stdout.
    Handles both '=' and ':' separators, and optional trailing units.
    """
    match = re.search(r"Wall\s*Time\s*[=:]\s*([0-9.]+)", output_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def check_baseline_age(cached):
    """Warn if cached baseline is older than BASELINE_MAX_AGE_HOURS."""
    try:
        created = datetime.fromisoformat(cached["created_at"])
        age = datetime.now() - created
        if age > timedelta(hours=BASELINE_MAX_AGE_HOURS):
            hours_old = age.total_seconds() / 3600
            print(f"\n⚠️  WARNING: Cached baseline is {hours_old:.1f} hours old "
                  f"(limit: {BASELINE_MAX_AGE_HOURS}h).")
            print("   Consider running with --new-baseline for accurate results.\n")
    except Exception:
        pass  # If date parsing fails, just skip the check


# ─────────────────────────────────────────────
# Baseline
# ─────────────────────────────────────────────

def measure_baseline(force_new=False):
    make_results_dir()

    if BASELINE_CACHE_FILE.exists() and not force_new:
        with open(BASELINE_CACHE_FILE, "r") as f:
            cached = json.load(f)

        print("\nUsing cached baseline power:")
        print(f"  Baseline power:            {cached['baseline_power_watts']:.2f} W")
        print(f"  Baseline measurement time: {cached['baseline_measurement_time_seconds']:.3f} s")
        print(f"  Cached at:                 {cached.get('created_at', 'unknown')}")

        check_baseline_age(cached)
        return cached

    print("\nMeasuring baseline power...")
    print("===========================")

    baseline_repetitions      = 12
    baseline_measurement_time = 10
    baseline_sleep            = 2

    cmd = [
        "python3",
        "run_basepower_adcewatt_var_std.py",
        "--repetitions",      str(baseline_repetitions),
        "--measurement-time", str(baseline_measurement_time),
        "--sleep",            str(baseline_sleep),
    ]

    subprocess.run(cmd, check=True)

    baseline_file = "all_powerresults_adcewatt_varstd.txt"

    if not os.path.exists(baseline_file):
        raise RuntimeError(f"Baseline file not found: {baseline_file}")

    baseline_power = None
    with open(baseline_file, "r") as f:
        for line in f:
            if "Average Power Consumption" in line:
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if numbers:
                    baseline_power = float(numbers[-1])
                break

    if baseline_power is None:
        raise RuntimeError("Could not read baseline power from baseline result file.")

    baseline_total_time    = baseline_repetitions * baseline_measurement_time
    baseline_energy_joules = baseline_power * baseline_total_time
    baseline_energy_wh     = baseline_energy_joules / 3600.0

    baseline_data = {
        "baseline_power_watts":              baseline_power,
        "baseline_repetitions":              baseline_repetitions,
        "baseline_measurement_time_seconds": baseline_total_time,
        "baseline_energy_joules":            baseline_energy_joules,
        "baseline_energy_wh":                baseline_energy_wh,
        "created_at":                        datetime.now().isoformat(),
    }

    with open(BASELINE_CACHE_FILE, "w") as f:
        json.dump(baseline_data, f, indent=4)

    print(f"Baseline power measured:   {baseline_power:.2f} W")
    print(f"Baseline measurement time: {baseline_total_time:.3f} s")

    return baseline_data


# ─────────────────────────────────────────────
# Application runner
# ─────────────────────────────────────────────

def run_application(app_type):
    if app_type not in SCRIPT_MAP:
        raise ValueError(f"Unknown app type: {app_type}")

    script = SCRIPT_MAP[app_type]

    if not os.path.exists(script):
        raise FileNotFoundError(f"Run script not found: {script}")

    print(f"\nRunning application script: {script}")
    print("===================================")

    app_start_time = time.time()

    process = subprocess.Popen(
        ["bash", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = process.communicate()

    app_end_time = time.time()
    python_execution_time = app_end_time - app_start_time

    if process.returncode != 0:
        raise RuntimeError(f"Application failed:\n{stderr}")

    # Always show stderr if it has content (warnings, etc.)
    if stderr.strip():
        print(f"\n⚠️  Application stderr output:\n{stderr}")

    app_wall_time = parse_wall_time(stdout)

    if app_wall_time is None:
        print("\n⚠️  WARNING: 'Wall Time' not found in application output!")
        print("   Falling back to Python-measured execution time.")
        print("   Check that your app prints:  Wall Time = <seconds>")
    else:
        print(f"\n✅  Application Wall Time found: {app_wall_time:.3f} s")

    return {
        "script":                         script,
        "stdout":                         stdout,
        "stderr":                         stderr,
        "python_execution_time_seconds":  python_execution_time,
        "application_wall_time_seconds":  app_wall_time,
        "app_start_time_epoch":           app_start_time,
        "app_end_time_epoch":             app_end_time,
    }


# ─────────────────────────────────────────────
# Power monitor
# ─────────────────────────────────────────────

def start_power_monitor(raw_csv_file):
    print("\nStarting power monitor...")
    print("=========================")

    output_file = open(raw_csv_file, "w")

    process = subprocess.Popen(
        [
            "/home/urooj/Adcewatts-stool/wattmetre-readmv2new",
            "-t",
            "/dev/ttyUSB0",
        ],
        stdout=output_file,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Give the monitor a moment to start writing data
    time.sleep(1)

    if process.poll() is not None:
        output_file.close()
        raise RuntimeError("Power monitor stopped unexpectedly.")

    return process, output_file


def stop_power_monitor(process, output_file):
    print("\nStopping power monitor...")
    print("========================")

    process.terminate()

    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

    output_file.close()


# ─────────────────────────────────────────────
# Power data processing
# ─────────────────────────────────────────────

def process_power_data(csv_file):
    df = pd.read_csv(csv_file)

    required_cols = ["#activepow8", "#activepow9"]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(
                f"Power CSV missing column '{col}'. "
                f"Available columns: {list(df.columns)}"
            )

    power8 = df["#activepow8"]
    power9 = df["#activepow9"]
    total_power = power8 + power9

    # ── Time-based energy integration ──────────────────────────────────────
    # Try to find a timestamp column for accurate ∫P dt (trapezoidal rule).
    # Common column names from wattmetre tools:
    time_col = None
    for candidate in ["time", "timestamp", "#time", "#timestamp", "Time", "Timestamp"]:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is not None:
        time_values = df[time_col].values.astype(float)
        # Normalise so first sample = 0
        time_values = time_values - time_values[0]
        integrated_energy_joules = float(np.trapz(total_power.values, time_values))
        integrated_duration_s    = float(time_values[-1] - time_values[0])
        integration_method       = "trapezoidal (timestamp column)"
    else:
        # No timestamp column — store None; save_results will use mean × wall_time
        integrated_energy_joules = None
        integrated_duration_s    = None
        integration_method       = "mean × wall_time (no timestamp column found)"

    stats = {
        "mean":                    float(total_power.mean()),
        "min":                     float(total_power.min()),
        "max":                     float(total_power.max()),
        "std":                     float(total_power.std()),
        "var":                     float(total_power.var()),
        "power8_mean":             float(power8.mean()),
        "power8_std":              float(power8.std()),
        "power8_var":              float(power8.var()),
        "power9_mean":             float(power9.mean()),
        "power9_std":              float(power9.std()),
        "power9_var":              float(power9.var()),
        "sample_count":            int(len(df)),
        "integrated_energy_joules": integrated_energy_joules,
        "integrated_duration_s":    integrated_duration_s,
        "integration_method":       integration_method,
    }

    return stats


# ─────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────

def save_results(app_type, run_id, baseline_data, app_data, power_stats, raw_power_file):
    prefix = RESULTS_DIR / f"{run_id}_{app_type}"

    stdout_file     = f"{prefix}_app_stdout.log"
    stderr_file     = f"{prefix}_app_stderr.log"
    report_file     = f"{prefix}_energy_report.txt"
    summary_file    = f"{prefix}_summary.json"
    saved_power_file = f"{prefix}_raw_power.csv"

    with open(stdout_file, "w") as f:
        f.write(app_data["stdout"])

    with open(stderr_file, "w") as f:
        f.write(app_data["stderr"])

    shutil.copyfile(raw_power_file, saved_power_file)

    # ── Decide which time to use for energy calculation ────────────────────
    baseline_power       = baseline_data["baseline_power_watts"]
    python_execution_time = app_data["python_execution_time_seconds"]
    app_wall_time        = app_data["application_wall_time_seconds"]

    if app_wall_time is not None:
        execution_time = app_wall_time
        time_source    = "Application printed Wall Time"
    else:
        execution_time = python_execution_time
        time_source    = "Python execution time (Wall Time not found — fallback)"

    # ── Energy calculation ─────────────────────────────────────────────────
    total_power   = power_stats["mean"]
    dynamic_power = total_power - baseline_power

    # baseline energy is always: baseline_power (W) × execution_time (s)
    baseline_energy_during_app_joules = baseline_power * execution_time

    # Prefer timestamp-based integration for total energy when available
    if power_stats["integrated_energy_joules"] is not None and not np.isnan(power_stats["integrated_energy_joules"]):
        total_energy_joules = power_stats["integrated_energy_joules"]
        energy_calc_note = (
            f"Trapezoidal integration over {power_stats['integrated_duration_s']:.3f} s "
            f"(from power meter timestamps)"
        )
    else:
        total_energy_joules = total_power * execution_time
        energy_calc_note = (
            f"Mean power × {time_source} ({execution_time:.3f} s)"
        )

    # Dynamic Energy = Total Application Energy - (Baseline Power × Execution Time)
    dynamic_energy_joules = total_energy_joules - (baseline_power * execution_time)

    baseline_energy_during_app_wh = baseline_energy_during_app_joules / 3600.0
    total_energy_wh               = total_energy_joules / 3600.0
    dynamic_energy_wh             = dynamic_energy_joules / 3600.0

    # ── Build summary dict ─────────────────────────────────────────────────
    summary = {
        "app_type":                                app_type,
        "run_script":                              app_data["script"],
        "time_used_for_energy_calculation":        time_source,
        "execution_time_used_seconds":             execution_time,
        "python_execution_time_seconds":           python_execution_time,
        "application_wall_time_seconds":           app_wall_time,
        "energy_calculation_note":                 energy_calc_note,
        "baseline_power_watts":                    baseline_power,
        "baseline_measurement_time_seconds":       baseline_data["baseline_measurement_time_seconds"],
        "baseline_measurement_energy_joules":      baseline_data["baseline_energy_joules"],
        "baseline_measurement_energy_wh":          baseline_data["baseline_energy_wh"],
        "total_power_during_application_watts":    total_power,
        "dynamic_power_increase_watts":            dynamic_power,
        "baseline_energy_during_application_joules": baseline_energy_during_app_joules,
        "baseline_energy_during_application_wh":    baseline_energy_during_app_wh,
        "total_application_energy_joules":         total_energy_joules,
        "total_application_energy_wh":             total_energy_wh,
        "dynamic_application_energy_joules":       dynamic_energy_joules,
        "dynamic_application_energy_wh":           dynamic_energy_wh,
        "peak_power_watts":                        power_stats["max"],
        "min_power_watts":                         power_stats["min"],
        "power_std_watts":                         power_stats["std"],
        "power_variance":                          power_stats["var"],
        "power_sample_count":                      power_stats["sample_count"],
        "integration_method":                      power_stats["integration_method"],
        "files": {
            "report":        report_file,
            "stdout":        stdout_file,
            "stderr":        stderr_file,
            "raw_power_csv": saved_power_file,
            "summary_json":  summary_file,
        },
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    # ── Write human-readable report ────────────────────────────────────────
    with open(report_file, "w") as f:
        f.write(f"APPLICATION ENERGY REPORT ({app_type})\n")
        f.write("=" * 50 + "\n\n")

        # 1. Baseline
        f.write("1. Baseline\n")
        f.write("-" * 30 + "\n")
        f.write(f"Baseline average power:      {baseline_power:.2f} W\n")
        f.write(f"Baseline measurement time:   {baseline_data['baseline_measurement_time_seconds']:.3f} s\n")
        f.write(f"Baseline measurement energy: {baseline_data['baseline_energy_wh']:.6f} Wh "
                f"({baseline_data['baseline_energy_joules']:.2f} J)\n\n")

        # 2. Timing
        f.write("2. Application Timing\n")
        f.write("-" * 30 + "\n")
        f.write(f"Python measured execution time:  {python_execution_time:.3f} s\n")

        if app_wall_time is not None:
            f.write(f"Application printed Wall Time:   {app_wall_time:.3f} s  ✅\n")
        else:
            f.write("Application printed Wall Time:   NOT FOUND  ⚠️\n")

        f.write(f"\n>>> Time used for energy calculation: {execution_time:.3f} s\n")
        f.write(f">>> Source: {time_source}\n\n")

        # 3. Power
        f.write("3. Power During Application\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total mean power:       {total_power:.2f} W\n")
        f.write(f"Dynamic power increase: {dynamic_power:.2f} W\n")
        f.write(f">>> Dynamic Power:      {dynamic_power:.2f} W  "
                f"(Total {total_power:.2f} W - Baseline {baseline_power:.2f} W)\n")
        f.write(f"Peak power:            {power_stats['max']:.2f} W\n")
        f.write(f"Min power:             {power_stats['min']:.2f} W\n")
        f.write(f"Power std deviation:   {power_stats['std']:.2f} W\n")
        f.write(f"Power variance:        {power_stats['var']:.2f}\n")
        f.write(f"Power samples:         {power_stats['sample_count']}\n\n")

        # 4. Energy
        f.write("4. Energy During Application\n")
        f.write("-" * 30 + "\n")
        f.write(f"Energy calculation method: {energy_calc_note}\n\n")
        f.write(f"Baseline energy during app:  {baseline_energy_during_app_wh:.6f} Wh "
                f"({baseline_energy_during_app_joules:.2f} J)\n")
        f.write(f"Total application energy:    {total_energy_wh:.6f} Wh "
                f"({total_energy_joules:.2f} J)\n")
        f.write(f"Dynamic application energy:  {dynamic_energy_wh:.6f} Wh "
                f"({dynamic_energy_joules:.2f} J)\n")
        f.write(f">>> Dynamic Energy:          {dynamic_energy_wh:.6f} Wh "
                f"({dynamic_energy_joules:.2f} J)  (Total - Baseline energy)\n\n")

        # 5. Files
        f.write("5. Saved Files\n")
        f.write("-" * 30 + "\n")
        f.write(f"Report file:    {report_file}\n")
        f.write(f"Stdout log:     {stdout_file}\n")
        f.write(f"Stderr log:     {stderr_file}\n")
        f.write(f"Raw power CSV:  {saved_power_file}\n")
        f.write(f"Summary JSON:   {summary_file}\n")

    # ── Colour codes ───────────────────────────────────────────────────────
    RESET       = "\033[0m"
    BOLD        = "\033[1m"
    G           = "\033[32m"         # green — normal text/values
    GB          = BOLD + "\033[32m"  # bold green — headers / highlight rows
    GBR         = "\033[32m"         # green reset shortcut
    GREEN       = "\033[32m"         # green — power & energy values
    GREENB      = BOLD + "\033[32m"  # bold green — dynamic power & energy values

    col1 = 36
    col2 = 22
    col3 = 22
    total_w = col1 + col2 + col3 + 4

    border   = GB + "+" + "-" * col1 + "+" + "-" * col2 + "+" + "-" * col3 + "+" + RESET
    border_h = GB + "+" + "=" * col1 + "+" + "=" * col2 + "+" + "=" * col3 + "+" + RESET

    # border for 2-column rows (no NOTE col) — spans col2+col3+3
    col_wide  = col2 + col3 + 3
    border2   = GB + "+" + "-" * col1 + "+" + "-" * col_wide + "+" + RESET
    border2_h = GB + "+" + "=" * col1 + "+" + "=" * col_wide + "+" + RESET

    def header_row(title, h1, h2):
        label = GB + f"  {title}" + RESET
        pad1  = col1 - 2 - len(title) - 2
        return (
            GB + "| " + RESET +
            label + G + " " * pad1 + RESET +
            GB + " | " + RESET +
            GB + f"{h1:<{col2-2}}" + RESET +
            GB + " | " + RESET +
            GB + f"{h2:<{col3-2}}" + RESET +
            GB + " |" + RESET
        )

    def header_row2(title, h1):
        """2-column header (no NOTE column)"""
        label = GB + f"  {title}" + RESET
        pad1  = col1 - 2 - len(title) - 2
        return (
            GB + "| " + RESET +
            label + G + " " * pad1 + RESET +
            GB + " | " + RESET +
            GB + f"{h1:<{col_wide-2}}" + RESET +
            GB + " |" + RESET
        )

    def data_row(label, val, note="", val_color=None):
        lbl_str  = G  + f"  {label:<{col1-4}}" + RESET
        vc       = val_color if val_color else G
        val_str  = vc + f"{val:<{col2-2}}"      + RESET
        note_str = G  + f"{note:<{col3-2}}"     + RESET
        sep = GB + " | " + RESET
        return GB + "| " + RESET + lbl_str + sep + val_str + sep + note_str + GB + " |" + RESET

    def data_row2(label, val):
        """2-column data row (no NOTE column)"""
        lbl_str = G + f"  {label:<{col1-4}}" + RESET
        val_str = G + f"{val:<{col_wide-2}}"  + RESET
        sep = GB + " | " + RESET
        return GB + "| " + RESET + lbl_str + sep + val_str + GB + " |" + RESET

    def highlight_row(label, val, note=""):
        lbl_str  = GB + f"  ★ {label:<{col1-6}}" + RESET
        val_str  = GREENB + f"{val:<{col2-2}}"    + RESET
        note_str = GREENB + f"{note:<{col3-2}}"   + RESET
        sep = GB + " | " + RESET
        return GB + "| " + RESET + lbl_str + sep + val_str + sep + note_str + GB + " |" + RESET

    # ── Title banner ───────────────────────────────────────────────────────
    print()
    print(GB + "+" + "=" * total_w + "+" + RESET)
    print(GB + "|" + f"{'  ENERGY MEASUREMENT SUMMARY':^{total_w}}" + "|" + RESET)
    print(G  + "|" + f"  App: {app_data['script']}  |  Run: {app_type}{'':{total_w - len(app_data['script']) - len(app_type) - 12}}" + "|" + RESET)
    print(GB + "+" + "=" * total_w + "+" + RESET)

    # ── TIMING section ─────────────────────────────────────────────────────
    print(border)
    print(header_row("TIMING", "VALUE", "NOTE"))
    print(border)
    print(data_row("Application execution time",
                   f"{execution_time:.3f} s", "printed by app"))
    print(border)

    # ── POWER section ──────────────────────────────────────────────────────
    print(header_row("POWER", "WATTS", "NOTE"))
    print(border)
    print(data_row("Baseline power",
                   f"{baseline_power:.2f} W",  "idle system",   val_color=GREEN))
    print(data_row("Total power (during app)",
                   f"{total_power:.2f} W",      "mean measured", val_color=GREEN))
    print(highlight_row("Dynamic Power",
                        f"{dynamic_power:.2f} W",
                        "Total - Baseline"))
    print(border)

    # ── ENERGY section ─────────────────────────────────────────────────────
    print(header_row("ENERGY", "Wh", "Joules"))
    print(border)
    print(data_row("Baseline energy (during app)",
                   f"{baseline_energy_during_app_wh:.6f} Wh",
                   f"{baseline_energy_during_app_joules:.2f} J",  val_color=GREEN))
    print(data_row("Total application energy",
                   f"{total_energy_wh:.6f} Wh",
                   f"{total_energy_joules:.2f} J",                val_color=GREEN))
    print(highlight_row("Dynamic Energy",
                        f"{dynamic_energy_wh:.6f} Wh",
                        f"{dynamic_energy_joules:.2f} J"))
    print(border)

    # ── SAMPLES section ────────────────────────────────────────────────────
    print(border2)
    print(header_row2("SAMPLES", "VALUE"))
    print(border2)
    print(data_row2("Power samples collected",
                    str(power_stats['sample_count'])))
    print(data_row2("Integration method",
                    power_stats['integration_method']))
    print(border2_h)

    # ── Saved files footer ─────────────────────────────────────────────────
    print()
    print(GB + "  Results saved in:" + RESET)
    print(GB + f"    {report_file}"      + RESET)
    print(GB + f"    {stdout_file}"      + RESET)
    print(GB + f"    {stderr_file}"      + RESET)
    print(GB + f"    {saved_power_file}" + RESET)
    print(GB + f"    {summary_file}"     + RESET)
    print()


# ─────────────────────────────────────────────
# Main measurement flow
# ─────────────────────────────────────────────

def measure_app_power(app_type, force_new_baseline=False):
    make_results_dir()

    run_id          = timestamp()
    temp_power_file = RESULTS_DIR / f"{run_id}_{app_type}_temp_power.csv"

    baseline_data = measure_baseline(force_new=force_new_baseline)

    monitor_process = None
    monitor_file    = None

    try:
        monitor_process, monitor_file = start_power_monitor(temp_power_file)

        # Small warmup buffer so power meter is stable before app starts
        time.sleep(0.5)

        app_data = run_application(app_type)

        # NO extra sleep here — stopping immediately after the app
        # avoids adding idle power samples to the measurement.

    finally:
        if monitor_process is not None and monitor_file is not None:
            stop_power_monitor(monitor_process, monitor_file)

    power_stats = process_power_data(temp_power_file)

    save_results(
        app_type=app_type,
        run_id=run_id,
        baseline_data=baseline_data,
        app_data=app_data,
        power_stats=power_stats,
        raw_power_file=temp_power_file,
    )

    if temp_power_file.exists():
        temp_power_file.unlink()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure application power and energy consumption."
    )

    parser.add_argument(
        "--app-type",
        choices=list(SCRIPT_MAP.keys()),
        required=True,
        help="Choose which Couette run script to execute.",
    )

    parser.add_argument(
        "--new-baseline",
        action="store_true",
        help="Force a new baseline measurement instead of using cached baseline.",
    )

    args = parser.parse_args()

    measure_app_power(
        app_type=args.app_type,
        force_new_baseline=args.new_baseline,
    )


if __name__ == "__main__":
    main()
