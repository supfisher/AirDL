#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH -N 2
#SBATCH -J toy
#SBATCH -o logs/toy.out
#SBATCH -e logs/toy.err
#SBATCH --time=1:35:00


#Here you should load the modules you need to run it

mpirun -np 8 python ./main.py