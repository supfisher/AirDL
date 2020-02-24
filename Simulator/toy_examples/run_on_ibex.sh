#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH -N 2
#SBATCH -J toy
#SBATCH -o logs/toy.out
#SBATCH -e logs/toy.err
#SBATCH --time=1:35:00


#Here you should load the modules you need to run it
module load cudnn/7.5.0-cuda10.1.105
mpirun -np 2 python ./main.py --dist_url='tcp://10.109.58.61:8091'
