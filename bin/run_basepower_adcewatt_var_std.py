#!/usr/bin/env python3
import subprocess
import time
import pandas as pd
import os
import argparse
from pathlib import Path
import signal
import sys

def run_wattmeter(output_dir, measurement_time=5):
    """Run the Adcewattmeter executable and save output to CSV"""
    meter_cmd = ["/home/urooj/Adcewatts-stool/wattmetre-readmv2new",
                 "-t", "/dev/ttyUSB0"]

    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "Adcewatt-output.csv")
    
    try:
        # Start the meter process
        process = subprocess.Popen(meter_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Collect output for measurement_time seconds
        start_time = time.time()
        output_lines = []
        
        while time.time() - start_time < measurement_time:
            line = process.stdout.readline()
            if line:
                output_lines.append(line)
        
        # Terminate the process
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        
        # Save collected output to file
        with open(output_file, 'w') as f:
            f.writelines(output_lines)
        
        return True
    except Exception as e:
        print(f"Error running wattmeter: {e}")
        return False

def process_power_data(csv_file):
    """Process the CSV file and calculate average power, variance, and standard deviation for columns 8 and 9"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Get active power values for plugs 8 and 9
        power8 = df['#activepow8']
        power9 = df['#activepow9']
        
        # Calculate statistics
        avg_power8 = power8.mean()
        avg_power9 = power9.mean()
        total_avg_power = avg_power8 + avg_power9
        
        # Calculate variance and standard deviation
        var_power8 = power8.var()
        var_power9 = power9.var()
        std_power8 = power8.std()
        std_power9 = power9.std()
        
        # Calculate total variance and std (assuming independence between plugs)
        total_var_power = var_power8 + var_power9
        total_std_power = (total_var_power) ** 0.5
        
        # Calculate energy consumption
        measurement_duration = (df['#timestamp'].max() - df['#timestamp'].min()) / 3600  # Convert to hours
        energy_consumption_wh = total_avg_power * measurement_duration
        energy_consumption_joules = energy_consumption_wh * 3600  # Convert Wh to Joules
        
        print(f"Plug 8 measurements:")
        print(f"  Average power: {avg_power8:.2f} W")
        print(f"  Standard deviation: {std_power8:.2f} W")
        print(f"  Variance: {var_power8:.2f} W²")
        
        print(f"\nPlug 9 measurements:")
        print(f"  Average power: {avg_power9:.2f} W")
        print(f"  Standard deviation: {std_power9:.2f} W")
        print(f"  Variance: {var_power9:.2f} W²")
        
        return {
            'total_avg_power': total_avg_power,
            'energy_wh': energy_consumption_wh,
            'energy_joules': energy_consumption_joules,
            'std_dev': total_std_power,
            'variance': total_var_power
        }
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        print("CSV contents:")
        try:
            with open(csv_file, 'r') as f:
                print(f.read())
        except Exception as read_error:
            print(f"Could not read CSV file: {read_error}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Monitor server power consumption')
    parser.add_argument('--repetitions', type=int, default=12,
                      help='Number of times to run the measurement')
    parser.add_argument('--sleep', type=int, default=0,
                      help='Sleep time between measurements in seconds')
    parser.add_argument('--measurement-time', type=int, default=5,
                      help='Time in seconds to collect measurements for each iteration')
    
    args = parser.parse_args()
    
    output_dir = "."
    results_file = os.path.join(output_dir, "all_powerresults_adcewatt_varstd.txt")
    
    # Remove old power-out.txt file if it exists
    if os.path.exists(results_file):
        os.remove(results_file)
    
    total_power = 0
    total_energy_wh = 0
    total_energy_joules = 0
    total_std_dev = 0
    total_variance = 0
    successful_runs = 0
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nCaught Ctrl+C, cleaning up...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Starting power monitoring for {args.repetitions} iterations...")
    
    for i in range(args.repetitions):
        print(f"\nIteration {i+1}/{args.repetitions}")
        
        if run_wattmeter(output_dir, args.measurement_time):
            csv_file = os.path.join(output_dir, "Adcewatt-output.csv")
            result = process_power_data(csv_file)
            
            if result is not None:
                total_power += result['total_avg_power']
                total_energy_wh += result['energy_wh']
                total_energy_joules += result['energy_joules']
                total_std_dev += result['std_dev']
                total_variance += result['variance']
                successful_runs += 1
                print(f"\nTotal measurements:")
                print(f"  Total average power: {result['total_avg_power']:.2f} W")
                print(f"  Total standard deviation: {result['std_dev']:.2f} W")
                print(f"  Total variance: {result['variance']:.2f} W²")
                print(f"  Energy consumption: {result['energy_wh']:.2f} Wh ({result['energy_joules']:.2f} J)")
        
        if i < args.repetitions - 1:  # Don't sleep after the last iteration
            print(f"Sleeping for {args.sleep} seconds...")
            time.sleep(args.sleep)
    
    if successful_runs > 0:
        final_avg_power = total_power / successful_runs
        final_std_dev = total_std_dev / successful_runs
        final_variance = total_variance / successful_runs
        # Calculate total energy from average power over total time
        total_time_seconds = args.repetitions * args.measurement_time
        total_energy_from_power = final_avg_power * total_time_seconds  # Energy in Joules
        
        # Save results to file
        with open(results_file, 'w') as f:
            f.write(f"Total Average Power Consumption of the HPC-Nexus Server: {final_avg_power:.2f} W\n")
            f.write(f"Total Time: {total_time_seconds} seconds\n")
            f.write(f"Total Energy (from avg power): {total_energy_from_power:.2f} Joules\n")
            f.write(f"\nStatistical Analysis:\n")
            f.write(f"Standard Deviation: {final_std_dev:.2f} W\n")
            f.write(f"Variance: {final_variance:.2f} W²\n")
        
        print(f"\nFinal Results:")
        print(f"Baseline/Average Power Consumption (Idle): {final_avg_power:.2f} W")
        print(f"Total Time: {total_time_seconds} seconds")
        print(f"Baseline/Total Energy (from avg power): {total_energy_from_power:.2f} Joules")
        print(f"Results saved to: {results_file}")
    else:
        print("No successful measurements were recorded.")

if __name__ == "__main__":
    main()

