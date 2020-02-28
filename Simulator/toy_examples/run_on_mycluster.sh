#!/bin/bash
#SBATCH --ntasks-per-node=20
#SBATCH -N 1
#SBATCH -J toy
#SBATCH -o toy.out
#SBATCH -e toy.err
#SBATCH --time=1:35:00


#run the application:
#OpenMP settings:


mpirun -np 2 --mca btl_tcp_if_include enp97s0f1 python ./main.py --clients=4 --backend='mpi'