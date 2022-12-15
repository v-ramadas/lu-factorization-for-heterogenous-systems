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

let "base = 2"
#./luFactorization 0 1
for i  in {1..13}
do
    let "number = $base ** $i"
    echo "Length of matrix : $number"
    ./luFactorization $number 20 16 16
done
