#!/bin/bash
#SBATCH --ntasks-per-node=10
#SBATCH -N 1
#SBATCH -J 0.01toy
#SBATCH -o toy_40_epsilon_0.01.out
#SBATCH -e toy_40_epsilon_0.01.err
#SBATCH --time=23:59:00


#run the application:
#OpenMP settings:


mpirun -np 10 --mca btl_tcp_if_include enp97s0f1
python ./main.py --clients=40 --backend='mpi' --epochs=50 --epsilon=0.01