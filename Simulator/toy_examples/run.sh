#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH -N 3
#SBATCH -J toy
#SBATCH -o logs/toy.out
#SBATCH -e logs/toy.err
#SBATCH --time=1:35:00


#run the application:
#OpenMP settings:


mpirun -np 3 --mca btl_tcp_if_include enp97s0f1 python ./main.py