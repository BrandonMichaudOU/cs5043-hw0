#!/bin/bash
# Brandon Michaud
#
# disc_dual_a100_students
#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH --mem=1G
# The %j is translated into the job number
#SBATCH --output=results/xor_%j_stdout.txt
#SBATCH --error=results/xor_%j_stderr.txt
#SBATCH --time=00:05:00
#SBATCH --job-name=hw0
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5043-hw0
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# Change this line to start an instance of your experiment
python hw0.py --project 'hw0' --lrate 0.005 --nonlinearity 'tanh' --nonlinearity_output 'tanh' --hidden 11 5 --epochs 1000 --exp 7 -vv
# python hw0.py --project 'hw0' --aggregate
