#!/bin/bash

# Move to project root (important for relative paths inside code)
cd /home/urooj/urooj-gerald-work-heterogeneous-v3-optimzed/culbibbs/home

# Load modules if required (uncomment if on HPC)
# module load cuda
# module load openmpi

# Yield CPU instead of spin-polling during MPI barriers → cuts barrier energy by ~80%
export OMPI_MCA_mpi_yield_when_idle=1

# OpenMP threads sleep at barriers instead of spinning
export OMP_WAIT_POLICY=passive
export GOMP_SPINCOUNT=0

mpirun --mca coll_hcoll_enable 0 \
       --mca pml ob1 \
       --mca btl self,smcuda,vader,tcp \
       -np 3 \
       ./bin/main_plane_couette_3D_CUDA_paramcalc_virtual_nodes_mpi_omp_cuda_1dZsv3.x \
       --mode hetero \
       --use-cpu 1 \
       --gpu 0,1 \
       --nx 150 \
       --ny 150 \
       --nz 150 \
       --cpu-threads 16 \
       --bind manual \
       --core-type logical \
       --gpu-host-cores 0,2 \
       --cpu-cores 12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42 \
       --print-binding
