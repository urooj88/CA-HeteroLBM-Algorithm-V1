# Folder Details

This file explains the purpose of each folder in the Git-ready
CA-HeteroLBM repository.

## Main Source and Build Files

`src/`
: Main CUDA/MPI/OpenMP source code for the LBM solver. This includes the
  heterogeneous CPU+GPU execution logic, optimized halo exchange, and
  GPU-aware MPI communication option.

`cudafunc/`
: CUDA helper functions and GPU-side support code used by the solver.

`3dbase/`, `3dgeneral/`, `3dsetup/`
: Supporting 3D LBM code, setup routines, and general helper functionality
  required by the solver.

`input/`
: Input/configuration files used by the LBM simulation.

`Makefile`
: Build file for compiling the solver executable.

## Runtime Scripts

`bin/`
: Python and shell scripts used to run the solver. Important files include:

- `run_lbm_baseline.py`: runs single-GPU, multi-GPU, and heterogeneous modes
  and saves timing/correctness reports.
- `run_lbm_energy_adcewatt.py`: runs energy measurements using the AdceWatt
  workflow.

The compiled executable is generated into this folder by `make`, but compiled
`.x` files should not be committed to Git.

## Documentation

`docs/run_instructions.txt`
: Main command reference for simple runs and energy measurement runs across all
  execution modes.

`docs/modes.txt`
: Summary of the currently supported execution modes.

`docs/build_and_run_on_another_system.txt`
: Notes for building and running the solver on another system.

`docs/energy_measurement_notes.txt`
: Notes related to AdceWatt energy measurement.

`docs/folder_details.md`
: This file.

## Results

`results_summary/`
: Place only final cleaned result summaries here, such as final CSV/XLSX tables
  used in the paper. Raw timestamped run folders should not be committed.

## Excluded From This Git Copy

The following are intentionally excluded:

- timestamped `baseline_results/` folders
- timestamped `bin/energy_results/` folders
- raw power logs
- compiled `.x` executables
- local tool folders such as `.codex/`, `.agents/`, and `.aws/`
- unrelated reference PDFs
