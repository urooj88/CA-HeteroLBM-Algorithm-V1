# CA-HeteroLBM-Algorithm

**CA-HeteroLBM** is a communication-aware heterogeneous CPU-GPU Lattice Boltzmann Method (LBM) solver for 3D Couette flow simulation. It supports single-GPU, multi-GPU, heterogeneous CPU+GPU, optimized heterogeneous, and GPU-aware optimized heterogeneous execution within one runtime workflow.

## Key Features

- **Unified LBM workflow:** run single-GPU, multi-GPU, CPU+GPU heterogeneous, optimized heterogeneous, and GPU-aware optimized heterogeneous modes from one codebase.

- **Profile-guided heterogeneous workload distribution:** uses tuned Z-slab distributions for two NVIDIA L40S GPUs and one CPU worker.

- **Communication-aware halo optimization and GPU-aware MPI variant:** supports direct GPU-GPU halo exchange with CUDA-aware MPI while keeping CPU-GPU exchange host-staged.

- **Correctness-preserving design:** compares final timestep, final error, and centre velocity components against the single-GPU baseline.

- **Energy measurement workflow:** includes scripts for wall time, compute time, communication time, I/O time, power, and dynamic energy measurement.

## Repository Layout

```text
CA-HetroLBM-Git/
├── 3dbase/                 # Base 3D LBM support code
├── 3dgeneral/              # General 3D helper code
├── 3dsetup/                # 3D setup and initialization support
├── bin/                    # Python runners and shell helper scripts
├── cudafunc/               # CUDA kernels/helper code
├── docs/                   # Run instructions and folder descriptions
├── input/                  # Input/configuration files
├── results_summary/        # Final summary tables only, not raw runs
├── src/                    # Main CUDA/MPI/OpenMP solver source
├── Makefile                # Build file
└── README.md
```

## Build

From the repository root:

```bash
make clean
make
```

For CUDA-aware MPI runs:

```bash
make clean
make MPICXX=/home/urooj/openmpi-cuda/bin/mpicxx
export PATH=/home/urooj/openmpi-cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/urooj/openmpi-cuda/lib:$LD_LIBRARY_PATH
```

## Run Examples

Single GPU:

```bash
python3 bin/run_lbm_baseline.py --mode single-gpu --size 100 --gpus 0
```

Multi GPU:

```bash
python3 bin/run_lbm_baseline.py --mode multi-gpu --size 100 --gpus 0,1
```

Optimized heterogeneous CPU+GPU:

```bash
python3 bin/run_lbm_baseline.py \
  --mode hetero \
  --size 250 \
  --gpus 0,1 \
  --cpu-threads 0 \
  --hetero-layout gpu-gpu-cpu
```

GPU-aware optimized heterogeneous CPU+GPU:

```bash
python3 bin/run_lbm_baseline.py \
  --mode hetero \
  --size 250 \
  --gpus 0,1 \
  --cpu-threads 0 \
  --hetero-layout gpu-gpu-cpu \
  --gpu-aware-mpi
```

For full run and energy measurement instructions, see:

```text
docs/run_instructions.txt
docs/modes.txt
docs/folder_details.md
```

## Running With or Without the Power Meter

The solver can be used without the AdceWatt power meter. In that case, run the
normal baseline runner:

```bash
python3 bin/run_lbm_baseline.py \
  --mode single-gpu \
  --size 100 \
  --gpus 0
```

This produces timing and correctness results, but not measured energy values.

Energy measurement is optional and requires an AdceWatt power meter that is installed, connected, and readable from the machine running the solver. Before running a full energy experiment, check that the wattmeter path works:

```bash
python3 bin/run_basepower_adcewatt_var_std.py \
  --repetitions 3 \
  --measurement-time 5 \
  --sleep 1
```

If the meter is detected correctly, this command prints idle power measurements. Then run an energy experiment:

```bash
python3 bin/run_lbm_energy_adcewatt.py \
  --mode hetero \
  --sizes 100 \
  --gpus 0,1 \
  --cpu-threads 0 \
  --hetero-layout gpu-gpu-cpu \
  --new-baseline
```

For GPU-aware optimized heterogeneous energy measurement:

```bash
python3 bin/run_lbm_energy_adcewatt.py \
  --mode hetero \
  --sizes 100 \
  --gpus 0,1 \
  --cpu-threads 0 \
  --hetero-layout gpu-gpu-cpu \
  --gpu-aware-mpi \
  --new-baseline
```

Energy outputs are saved in:

```text
bin/energy_results/
```



