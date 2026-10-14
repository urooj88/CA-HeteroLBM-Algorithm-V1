#!/bin/bash
set -e

cd /home/urooj/urooj-gerald-work-heterogeneous-v3-optimzed/culbibbs/home || exit 1

mpirun --mca coll_hcoll_enable 0 --mca pml ob1 --mca btl self,smcuda,vader,tcp -np 2 \
./bin/main_plane_couette_3D_CUDA_paramcalc_virtual_nodes_mpi_omp_cuda_1dZsv3.x \
--mode hetero \
--use-cpu 0 \
--gpu 0,1 \
--nx 100 --ny 100 --nz 100 \
--bind manual \
--core-type logical \
--gpu-host-cores 0,2 \
--print-binding
