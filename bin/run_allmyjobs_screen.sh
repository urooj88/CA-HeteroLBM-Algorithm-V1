#!/bin/bash

echo "Starting energy experiments..."

# ---------------- FIRST ----------------
echo "Running FIRST application..."
python3 run_appenergy_couette_adcewatt_stdvar.py --app-type first \
> ER100_output.log 2>&1

echo "FIRST completed."
echo "Sleeping for 3 minutes..."
sleep 180

# ---------------- SECOND ----------------
echo "Running SECOND application..."
python3 run_appenergy_couette_adcewatt_stdvar.py --app-type second \
> ER150_output.log 2>&1

echo "SECOND completed."
echo "Sleeping for 3 minutes..."
sleep 180

# ---------------- THIRD ----------------
echo "Running THIRD application..."
python3 run_appenergy_couette_adcewatt_stdvar.py --app-type third \
> ER200_output.log 2>&1

echo "THIRD completed."

echo "All experiments finished."
