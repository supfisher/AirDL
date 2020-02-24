#!/bin/bash
#SBATCH --ntasks-per-node=2
#SBATCH -N 1
#SBATCH -J toy
#SBATCH -o logs/toy.out
#SBATCH -e logs/toy.err
#SBATCH --time=1:35:00


#run the application:
#OpenMP settings:


mpirun -np 2 --mca btl_tcp_if_include enp97s0f1 python ./main.py --dist_url='tcp://10.68.170.167:8091'