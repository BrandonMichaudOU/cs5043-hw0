#!/bin/bash
# Brandon Michaud
#
#SBATCH --partition=disc_dual_a100_students
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH --mem=1G
# The %j is translated into the job number
#SBATCH --output=results/xor_%j_stdout.txt
#SBATCH --error=results/xor_%j_stderr.txt
#SBATCH --time=00:15:00
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
python hw0.py --project 'hw0' --lrate 0.005 --nonlinearity 'tanh' --nonlinearity_output 'tanh' --hidden 9 5 --epochs 1000 --exp 6 -vv


# parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
# parser.add_argument('--nonlinearity', type=str, default='sigmoid', help='activation function')
# parser.add_argument('--nonlinearity_output', type=str, default='sigmoid', help='activation function in output layer')
# parser.add_argument('--project', type=str, default='HW0', help='Name for WandB')
# parser.add_argument('--label', type=str, default='None', help='Used for organization with WandB')
# parser.add_argument('--exp', type=int, default=0, help='Experiment number')
# parser.add_argument('--hidden', nargs='+', type=int, default=[5], help='Number of hidden units per layer (sequence of ints)')
# parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
# parser.add_argument('--cpus-per-task', type=int, default=None, help='Number of threads to use')
# parser.add_argument('--nogo', action='store_true', help='Do not execute the experiment')
# parser.add_argument('--gpu', action='store_true', help='Use a GPU')
# parser.add_argument('--aggregate', action='store_true', help='Aggregate the results instead of training')
# parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbosity level')