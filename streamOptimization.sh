#!/usr/bin/env zsh
#SBATCH --job-name=task
#SBATCH --partition=wacc
#SBATCH -c 1
#SBATCH --output=task.out --error task.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00

module load nvidia/cuda/11.6.0
module load anaconda/full/2021.05

#python luFactorization.py -d -n 4

make clean && make

let "size = 2**10"
let "thread = 25"
for s in {1..64}
do
    echo "Length of matrix : $size, Threads : $thread, numStreams $s"
    ./luFactorization $size $thread $s 64
done
