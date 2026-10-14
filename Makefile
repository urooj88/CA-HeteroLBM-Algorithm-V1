# ==============================
# Compiler and paths
# ==============================

NVCC        = nvcc
MPICXX      = mpicxx
CUDA_ARCH   = sm_89

CUDA_HOME  ?= /usr/local/cuda

# ==============================
# MPI flags (cleaned for nvcc)
# ==============================

MPI_CFLAGS_RAW := $(shell $(MPICXX) --showme:compile 2>/dev/null)
MPI_LIBS_RAW   := $(shell $(MPICXX) --showme:link 2>/dev/null)

# nvcc cannot take -pthread directly here
MPI_BASE_INC := $(word 1,$(MPI_CFLAGS_RAW)) $(word 2,$(MPI_CFLAGS_RAW))
MPI_CFLAGS := $(filter-out -pthread,$(MPI_BASE_INC))

# pull only the library search paths from mpicxx
MPI_LINK_DIRS := $(filter -L%,$(MPI_LIBS_RAW))

# ==============================
# Flags
# ==============================

NVCCFLAGS = -O3 -std=c++17 -rdc=true \
            -arch=$(CUDA_ARCH) \
            -ccbin $(MPICXX) \
            -Xcompiler="-fopenmp -pthread"

INCLUDES = -I./3dbase \
           -I./3dgeneral \
           -I./3dsetup \
           -I./cudafunc \
           -I./src \
           -I$(CUDA_HOME)/include \
           $(MPI_CFLAGS)

# ==============================
# Optional NVML for adaptive DVFS
# Build with:  make USE_NVML=1   (default: enabled)
#              make USE_NVML=0   (disable — DVFS becomes no-op)
# ==============================
USE_NVML ?= 1
ifeq ($(USE_NVML), 1)
    NVCCFLAGS += -DUSE_NVML
    NVML_LIBS  = -lnvidia-ml
else
    NVML_LIBS  =
endif

# IMPORTANT:
# Use OpenMPI/HPC-X internal hwloc via libopen-pal
# not external -lhwloc
LIBS = -L$(CUDA_HOME)/lib64 \
       -lcudadevrt \
       -lcudart \
       $(MPI_LINK_DIRS) \
       -lmpi \
       -lopen-pal \
       -lhwloc \
       $(NVML_LIBS)

# ==============================
# Files
# ==============================

SRC = src/main_plane_couette_3D_CUDA_paramcalc_virtual_nodes_mpi_omp_cuda_1dZsv3.cu
BIN = bin/main_plane_couette_3D_CUDA_paramcalc_virtual_nodes_mpi_omp_cuda_1dZsv3.x

# ==============================
# Targets
# ==============================

all: mpi_omp_cuda

mpi_omp_cuda: $(BIN)

$(BIN): $(SRC)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SRC) -o $(BIN) $(LIBS)

clean:
	rm -rf bin/*.x
	
	
