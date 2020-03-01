#!/bin/bash
#SBATCH --ntasks-per-node=5
#SBATCH -N 1
#SBATCH -J 10toy
#SBATCH -o toy_10_bs_64.out
#SBATCH -e toy_10_bs_64.err
#SBATCH --time=23:59:00


#run the application:
#OpenMP settings:


mpirun -np 5 --mca btl_tcp_if_include enp97s0f1 \
python ./main.py --clients=10 --backend='mpi' --epochs=50 --epsilon=0.2