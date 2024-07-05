#!/bin/bash
#SBATCH --job-name=hpcg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
source env.sh
cd ./bin
START_THREADS=8
END_THREADS=16
for (( NUM_THREADS=$START_THREADS; NUM_THREADS<=$END_THREADS; NUM_THREADS++ ))
do
export OMP_NUM_THREADS=$NUM_THREADS
mpirun -n 8 ./xhpcg
done


