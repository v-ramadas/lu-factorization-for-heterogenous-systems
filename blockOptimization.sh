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
for b in {1..10}
do
    let "blockSize = 2**$b"
    echo "Length of matrix : $size, Threads : 25, blockSize : $blockSize"
    ./luFactorization $size 25 16 $blockSize
done
