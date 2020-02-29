#!/bin/bash
#SBATCH --ntasks-per-node=10
#SBATCH -N 1
#SBATCH -J 3.2toy
#SBATCH -o toy_40_epsilon_3.2.out
#SBATCH -e toy_40_epsilon_3.2.err
#SBATCH --time=12:35:00


#run the application:
#OpenMP settings:


mpirun -np 10 --mca btl_tcp_if_include enp97s0f1
python ./main.py --clients=40 --backend='mpi' --epochs=50 --epsilon=3.2