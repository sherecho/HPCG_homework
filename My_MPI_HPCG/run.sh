#!/bin/bash
#SBATCH --job-name=hpcg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
source env.sh
cd ./bin
export OMP_NUM_THREADS=12
mpirun -n 8 ./xhpcg


