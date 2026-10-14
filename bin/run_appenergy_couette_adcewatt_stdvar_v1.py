#!/usr/bin/env python3

import subprocess
import time
import pandas as pd
import os
import argparse
import signal


BASELINE_REPETITIONS = 12
BASELINE_MEASUREMENT_TIME = 10
BASELINE_SLEEP = 2

BASELINE_TOTAL_MEASUREMENT_TIME = BASELINE_REPETITIONS * BASELINE_MEASUREMENT_TIME


def measure_baseline():
    """Measure average baseline power using the existing baseline script."""

    print("\nMeasuring Baseline Power Consumption...")
    print("=======================================")

    cmd = [
        "python3", "run_basepower_adcewatt_var_std.py",
        "--repetitions", str(BASELINE_REPETITIONS),
        "--measurement-time", str(BASELINE_MEASUREMENT_TIME),
        "--sleep", str(BASELINE_SLEEP)
    ]

    try:
        subprocess.run(cmd, check=True)

        baseline_file = "all_powerresults_adcewatt_varstd.txt"

        with open(baseline_file, "r") as f:
            baseline_data = f.read()

        baseline_power = None

        for line in baseline_data.splitlines():
            if "Average Power Consumption" in line:
                parts = line.split()
                for item in parts:
                    try:
                        baseline_power = float(item)
                        break
                    except ValueError:
                        continue
                break

        if baseline_power is None:
            raise RuntimeError("Could not find baseline average power.")

        print(f"Baseline average power: {baseline_power:.2f} W")
        print(f"Baseline measurement time: {BASELINE_TOTAL_MEASUREMENT_TIME:.2f} s")

        return baseline_power

    except Exception as e:
        print(f"Error measuring baseline: {e}")
        return None


def run_application(app_type):
    """Run selected application script and measure exact application runtime."""

    script_map = {
        "first": "./run_cuda_main_couette1.sh",
        "second": "./run_cuda_main_couette2.sh",
        "third": "./run_cuda_main_couette3.sh",
        "fourth": "./run_cuda_main_couette4.sh",
        "fifth": "./run_cuda_main_couette5.sh",
        "sixth": "./run_cuda_main_couette6.sh",
    }

    if app_type not in script_map:
        print(f"Error: unknown app-type '{app_type}'")
        return None, 0.0

    script = script_map[app_type]

    print(f"\nSelected run script: {script}")
    print("Running application...")

    cmd = ["bash", script]

    try:
        start_time = time.perf_counter()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate()

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        if process.returncode != 0:
            print("Error running application:")
            print(stderr.decode(errors="replace"))
            return None, 0.0

        return stdout.decode(errors="replace"), execution_time

    except Exception as e:
        print(f"Error running application: {e}")
        return None, 0.0


def process_power_data(csv_file):
    """Process power measurements from CSV file."""

    try:
        df = pd.read_csv(csv_file)

        power8 = df["#activepow8"]
        power9 = df["#activepow9"]

        total_power = power8 + power9

        stats = {
            "mean": total_power.mean(),
            "min": total_power.min(),
            "max": total_power.max(),
            "std": total_power.std(),
            "var": total_power.var(),

            "power8_mean": power8.mean(),
            "power8_std": power8.std(),
            "power8_var": power8.var(),

            "power9_mean": power9.mean(),
            "power9_std": power9.std(),
            "power9_var": power9.var(),
        }

        print("\nPlug 8 measurements:")
        print(f"  Average power: {stats['power8_mean']:.2f} W")
        print(f"  Standard deviation: {stats['power8_std']:.2f} W")
        print(f"  Variance: {stats['power8_var']:.2f} W²")

        print("\nPlug 9 measurements:")
        print(f"  Average power: {stats['power9_mean']:.2f} W")
        print(f"  Standard deviation: {stats['power9_std']:.2f} W")
        print(f"  Variance: {stats['power9_var']:.2f} W²")

        print("\nTotal measurements:")
        print(f"  Total average power: {stats['mean']:.2f} W")
        print(f"  Total standard deviation: {stats['std']:.2f} W")
        print(f"  Total variance: {stats['var']:.2f} W²")
        print(f"  Peak power: {stats['max']:.2f} W")

        return stats

    except Exception as e:
        print(f"Error processing CSV data: {e}")
        return None


def cleanup_previous_outputs():
    """Remove previous output files before starting new measurements."""

    files_to_remove = [
        "all_powerresults_adcewatt_varstd.txt",
        "all_powerresults_adceawatt_varstd.txt",
        "Adcewatt-output.csv",
        "mxm_power.csv",
        "power_first_mxm.txt",
        "power_second_mxm.txt",
        "power_third_mxm.txt",
        "power_fourth_mxm.txt",
        "power_fifth_mxm.txt",
        "power_sixth_mxm.txt",
    ]

    print("Cleaning up previous output files...")

    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"  Removed: {file}")
            except Exception as e:
                print(f"  Error removing {file}: {e}")

    print("Cleanup complete.\n")


def measure_app_power(app_type):
    """Measure baseline power, run application, then calculate energy."""

    output_dir = "."
    results_file = os.path.join(output_dir, f"power_{app_type}_mxm.txt")

    if os.path.exists(results_file):
        os.remove(results_file)

    baseline_power = measure_baseline()

    if baseline_power is None:
        print("Baseline measurement failed.")
        return

    print("\nMeasuring Application Power...")
    print("=============================")

    monitor_output = open("mxm_power.csv", "w")

    monitor_process = subprocess.Popen(
        [
            "/home/urooj/Adcewatts-stool/wattmetre-readmv2new",
            "-t", "/dev/ttyUSB0"
        ],
        stdout=monitor_output,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

    time.sleep(1)

    output, execution_time = run_application(app_type)

    time.sleep(1)

    try:
        os.killpg(os.getpgid(monitor_process.pid), signal.SIGTERM)
    except Exception:
        monitor_process.terminate()

    try:
        monitor_process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(monitor_process.pid), signal.SIGKILL)
        except Exception:
            monitor_process.kill()
        monitor_process.wait()

    monitor_output.close()

    if output is None:
        print("Application failed. No energy report generated.")
        return

    execution_stats = process_power_data("mxm_power.csv")

    if execution_stats is None:
        print("Failed to process power data.")
        return

    execution_power = execution_stats["mean"]
    power_increase = execution_power - baseline_power

    baseline_measurement_energy_joules = (
        baseline_power * BASELINE_TOTAL_MEASUREMENT_TIME
    )

    baseline_energy_during_app_joules = baseline_power * execution_time
    total_energy_joules = execution_power * execution_time
    dynamic_energy_joules = total_energy_joules - baseline_energy_during_app_joules

    baseline_measurement_energy_wh = baseline_measurement_energy_joules / 3600
    baseline_energy_during_app_wh = baseline_energy_during_app_joules / 3600
    total_energy_wh = total_energy_joules / 3600
    dynamic_energy_wh = dynamic_energy_joules / 3600

    script_number = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
    }[app_type]

    with open(results_file, "w") as f:
        f.write(f"APPLICATION ENERGY REPORT ({app_type})\n")
        f.write("=========================================\n\n")

        f.write("1. Baseline Measurements\n")
        f.write("------------------------\n")
        f.write(f"Baseline Average Power: {baseline_power:.2f} W\n")
        f.write(f"Baseline Measurement Time: {BASELINE_TOTAL_MEASUREMENT_TIME:.3f} s\n")
        f.write(
            f"Baseline Measurement Energy: "
            f"{baseline_measurement_energy_wh:.6f} Wh "
            f"({baseline_measurement_energy_joules:.2f} J)\n\n"
        )

        f.write("2. Application Measurements\n")
        f.write("---------------------------\n")
        f.write(f"Application Execution Time: {execution_time:.3f} s\n")
        f.write(f"Total Power During Application: {execution_power:.2f} W\n")
        f.write(f"Dynamic Power Increase: {power_increase:.2f} W\n")
        f.write(f"Peak Power: {execution_stats['max']:.2f} W\n")
        f.write(f"Power Standard Deviation: {execution_stats['std']:.2f} W\n")
        f.write(f"Power Variance: {execution_stats['var']:.2f} W²\n\n")

        f.write("3. Energy Analysis\n")
        f.write("------------------\n")
        f.write(
            f"Baseline Energy During Application: "
            f"{baseline_energy_during_app_wh:.6f} Wh "
            f"({baseline_energy_during_app_joules:.2f} J)\n"
        )
        f.write(
            f"Total Application Energy: "
            f"{total_energy_wh:.6f} Wh "
            f"({total_energy_joules:.2f} J)\n"
        )
        f.write(
            f"Dynamic Application Energy: "
            f"{dynamic_energy_wh:.6f} Wh "
            f"({dynamic_energy_joules:.2f} J)\n\n"
        )

        f.write("4. Formula Used\n")
        f.write("---------------\n")
        f.write("Total Energy = Total Power During Application × Application Execution Time\n")
        f.write("Baseline Energy During Application = Baseline Power × Application Execution Time\n")
        f.write("Dynamic Energy = Total Energy - Baseline Energy During Application\n\n")

        f.write("5. Application Details\n")
        f.write("----------------------\n")
        f.write(f"Application Type: {app_type}\n")
        f.write(f"Run Script: run_cuda_main_couette{script_number}.sh\n")

    print(f"\nResults saved to: {results_file}")

    print("\nENERGY CONSUMPTION SUMMARY")
    print("==========================")

    print("\n1. Baseline:")
    print(f"   Baseline average power: {baseline_power:.2f} W")
    print(f"   Baseline measurement time: {BASELINE_TOTAL_MEASUREMENT_TIME:.3f} s")
    print(
        f"   Baseline measurement energy: "
        f"{baseline_measurement_energy_wh:.6f} Wh "
        f"({baseline_measurement_energy_joules:.2f} J)"
    )

    print("\n2. Application:")
    print(f"   Application execution time: {execution_time:.3f} s")
    print(f"   Total power during application: {execution_power:.2f} W")
    print(f"   Dynamic power increase: {power_increase:.2f} W")

    print("\n3. Energy:")
    print(
        f"   Baseline energy during application: "
        f"{baseline_energy_during_app_wh:.6f} Wh "
        f"({baseline_energy_during_app_joules:.2f} J)"
    )
    print(
        f"   Total application energy: "
        f"{total_energy_wh:.6f} Wh "
        f"({total_energy_joules:.2f} J)"
    )
    print(
        f"   Dynamic application energy: "
        f"{dynamic_energy_wh:.6f} Wh "
        f"({dynamic_energy_joules:.2f} J)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Measure application energy consumption"
    )

    parser.add_argument(
        "--app-type",
        choices=["first", "second", "third", "fourth", "fifth", "sixth"],
        required=True,
        help="Choose which Couette run script to execute"
    )

    args = parser.parse_args()

    cleanup_previous_outputs()
    measure_app_power(args.app_type)


if __name__ == "__main__":
    main()
