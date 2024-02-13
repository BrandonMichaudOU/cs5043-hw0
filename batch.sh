#!/bin/bash
# Brandon Michaud
#
# disc_dual_a100_students
#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH --mem=1G
#SBATCH --output=results/hw0_%j_stdout.txt
#SBATCH --error=results/hw0_%j_stderr.txt
#SBATCH --time=00:02:00
#SBATCH --job-name=hw0
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5043-hw0

. /home/fagg/tf_setup.sh
conda activate tf

# For running experiment
python hw0.py --project 'hw0' --lrate 0.005 --nonlinearity 'tanh' --nonlinearity_output 'tanh' --hidden 11 5 --epochs 1000 --exp 9 -vv

# For aggregating
# python hw0.py --project 'hw0' --aggregate
