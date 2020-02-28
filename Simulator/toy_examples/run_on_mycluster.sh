#!/bin/bash
#SBATCH --ntasks-per-node=5
#SBATCH -N 1
#SBATCH -J toy
#SBATCH -o toy_10.out
#SBATCH -e toy_10.err
#SBATCH --time=12:35:00


#run the application:
#OpenMP settings:


mpirun -np 5 --mca btl_tcp_if_include enp97s0f1 \
python ./main.py --clients=10 --backend='mpi' --epochs=50