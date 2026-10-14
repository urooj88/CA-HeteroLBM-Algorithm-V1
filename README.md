# CA-HeteroLBM-Algorithm

### Communication-Aware Heterogeneous CPU–GPU Lattice Boltzmann Solver for 3D Couette Flow

**Developed by Urooj Asgher** 
Technological University Dublin, Ireland 
**ORCID:** [0000-0001-9218-3307](https://orcid.org/0000-0001-9218-3307)

![CUDA](https://img.shields.io/badge/CUDA-GPU%20Computing-76B900) ![MPI](https://img.shields.io/badge/MPI-Distributed%20Computing-blue) ![OpenMP](https://img.shields.io/badge/OpenMP-CPU%20Parallelism-orange) ![HPC](https://img.shields.io/badge/HPC-Research-blueviolet)

## Overview

**CA-HeteroLBM** is a communication-aware heterogeneous CPU–GPU Lattice Boltzmann Method (LBM) solver for **3D Couette flow simulation**. It supports single-GPU, multi-GPU, heterogeneous CPU+GPU, optimized heterogeneous, and GPU-aware optimized heterogeneous execution within one runtime workflow.

The framework includes profile-guided workload distribution, communication-aware halo exchange, correctness checks against a single-GPU baseline, and optional physical energy measurement using an AdceWatt power meter.

## Key Features

- **Unified LBM workflow:** Single-GPU, multi-GPU, CPU+GPU heterogeneous, optimized heterogeneous, and GPU-aware optimized heterogeneous execution from one codebase.
- **Profile-guided heterogeneous workload distribution:** Tuned Z-slab distributions for two NVIDIA L40S GPUs and one CPU worker.
- **Communication-aware halo optimization:** GPU-aware MPI supports direct GPU–GPU halo exchange while CPU–GPU exchange remains host-staged.
- **Correctness-preserving design:** Compares final timestep, final error, and centre velocity components with the single-GPU baseline.
- **Performance and energy evaluation:** Includes wall time, compute time, communication time, I/O time, power, and dynamic energy measurement workflows.

## Execution Configurations

| Execution mode | Description | Example |
|---|---|---|
| Single GPU | Baseline execution on one GPU | `--mode single-gpu --gpus 0` |
| Multi GPU | Execution across two GPUs | `--mode multi-gpu --gpus 0,1` |
| Heterogeneous CPU+GPU | Combined CPU and GPU execution | `--mode hetero` |
| Optimized heterogeneous | Profile-guided heterogeneous layout | `--hetero-layout gpu-gpu-cpu` |
| GPU-aware optimized heterogeneous | Heterogeneous execution with GPU-aware MPI | `--gpu-aware-mpi` |

## Repository Structure

```text
CA-HeteroLBM-Algorithm-V1/
├── 3dbase/                 # Base 3D LBM support code
├── 3dgeneral/              # General 3D helper code
├── 3dsetup/                # 3D setup and initialization support
├── bin/                    # Python runners and shell helper scripts
├── cudafunc/               # CUDA kernels and helper code
├── docs/                   # Run instructions and folder descriptions
├── input/                  # Input and configuration files
├── resource-check/         # Resource-check utilities
├── results_summary/        # Final summary tables, not raw runs
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

> **Note:** The CUDA-aware MPI paths above are specific to the original HPC environment. Adjust them for your own MPI installation.

## Usage

### 1. Single GPU

```bash
python3 bin/run_lbm_baseline.py --mode single-gpu --size 100 --gpus 0
```

### 2. Multi GPU

```bash
python3 bin/run_lbm_baseline.py --mode multi-gpu --size 100 --gpus 0,1
```

### 3. Optimized heterogeneous CPU+GPU

```bash
python3 bin/run_lbm_baseline.py \
  --mode hetero \
  --size 250 \
  --gpus 0,1 \
  --cpu-threads 0 \
  --hetero-layout gpu-gpu-cpu
```

### 4. GPU-aware optimized heterogeneous CPU+GPU

```bash
python3 bin/run_lbm_baseline.py \
  --mode hetero \
  --size 250 \
  --gpus 0,1 \
  --cpu-threads 0 \
  --hetero-layout gpu-gpu-cpu \
  --gpu-aware-mpi
```

For complete run instructions, see [`docs/run_instructions.txt`](docs/run_instructions.txt), [`docs/modes.txt`](docs/modes.txt), and [`docs/folder_details.md`](docs/folder_details.md).

## Performance and Correctness Evaluation

The workflow supports timing and correctness checks, including:

| Category | Recorded or checked information |
|---|---|
| Execution time | Wall time, compute time, communication time, I/O time |
| Correctness | Final timestep, final error, centre velocity components |
| Power and energy (optional) | Power and dynamic energy measurements |

Correctness is evaluated relative to the single-GPU baseline.

## Running With or Without the Power Meter

**The AdceWatt power meter is optional.** You can run the solver without it to obtain timing and correctness results, but not physically measured energy values.

### Run without the power meter

```bash
python3 bin/run_lbm_baseline.py \
  --mode single-gpu \
  --size 100 \
  --gpus 0
```

### Check the AdceWatt power meter

Energy measurement requires an AdceWatt power meter installed, connected, and readable from the execution machine. Before running an experiment, check idle power measurement:

```bash
python3 bin/run_basepower_adcewatt_var_std.py \
  --repetitions 3 \
  --measurement-time 5 \
  --sleep 1
```

If the meter is detected correctly, this command prints idle power measurements.

### Run a heterogeneous energy experiment

```bash
python3 bin/run_lbm_energy_adcewatt.py \
  --mode hetero \
  --sizes 100 \
  --gpus 0,1 \
  --cpu-threads 0 \
  --hetero-layout gpu-gpu-cpu \
  --new-baseline
```

### Run a GPU-aware heterogeneous energy experiment

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

Energy outputs are saved in `bin/energy_results/`.

## Documentation

- [`docs/run_instructions.txt`](docs/run_instructions.txt): Full execution instructions.
- [`docs/modes.txt`](docs/modes.txt): Execution mode descriptions.
- [`docs/folder_details.md`](docs/folder_details.md): Repository folder descriptions.

## Research Environment

The workload distribution described here is tuned for **two NVIDIA L40S GPUs and one CPU worker**. Execution requires a compatible CUDA, MPI, and OpenMP build environment. GPU-aware execution additionally requires CUDA-aware MPI.

## Citation

If you use this repository in research, please cite the associated publication when its bibliographic details are available. A publication citation has not been included here because no verified title, venue, or DOI was provided.

## Acknowledgements

This research was developed at the HPC-Nexus Lab, Technological University Dublin, Ireland.

**PhD Funding:** My PhD research is funded by the **Irish Research Council (IRC)**.

