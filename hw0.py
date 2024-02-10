'''
Advanced Machine Learning HW0

Brandon Michaud
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time
import re
import socket

import argparse
import pickle
import wandb

# Tensorflow 2.x way of doing things
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential

#################################################################
# Default plotting parameters
FONTSIZE = 18
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

#################################################################
def build_model(n_inputs, hidden_layers, n_output, activation='elu', activation_output='elu', lrate=0.001):
	'''
	Construct a network with one hidden layer
	- Adam optimizer
	- MSE loss
	
	:param n_inputs: Number of input dimensions
	:param n_hidden: Number of units in the hidden layer
	:param n_output: Number of ouptut dimensions
	:param activation: Activation function to be used for hidden and output units
	:param lrate: Learning rate for Adam Optimizer
	'''
	model = Sequential()
	model.add(InputLayer(input_shape=(n_inputs,)))
	for i, n_hidden in enumerate(hidden_layers):
		model.add(Dense(n_hidden, use_bias=True, name='Hidden_%d'%i, activation=activation))
	model.add(Dense(n_output, use_bias=True, name='Output', activation=activation_output))
	
	# Optimizer
	opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)
	
	# Bind the optimizer and the loss function to the model
	model.compile(loss='mse', optimizer=opt)
	
	# Generate an ASCII representation of the architecture
	print(model.summary())
	return model

def args2string(args):
	'''
	Translate the current set of arguments
	
	:param args: Command line arguments
	'''
	return "exp_%02d_hidden_"%(args.exp) + '_'.join([str(h) for h in args.hidden])
	
	
########################################################
def execute_exp(args):
	'''
	Execute a single instance of an experiment.  The details are specified in the args object
	
	:param args: Command line arguments
	'''

	##############################
	# Run the experiment

	# Describe arguments
	argstring = args2string(args)
	print("EXPERIMENT: %s"%argstring)

	# Initialize WANDB
	wandb.init(project=args.project, name='hw0_%d'%(args.exp), notes=argstring, config=vars(args))

	# Load training set
	fp = open("hw0_dataset.pkl", "rb")
	training_set = pickle.load(fp)
	fp.close()

	ins = training_set['ins']
	outs = training_set['outs']
	
	model = build_model(ins.shape[1], args.hidden, outs.shape[1], activation=args.nonlinearity, activation_output=args.nonlinearity_output, lrate=args.lrate)

	# Callbacks
	cbs = []

	#early_stopping_cb = # TODO

	wandb.log({'hostname': socket.gethostname()})

	cbs.append(wandb.keras.WandbMetricsLogger()) 
	
	# Only execute if we are 'going'
	if not args.nogo:
		# Training
		print("Training...")        
		history = model.fit(x=ins, y=outs, epochs=args.epochs, verbose=args.verbose >= 2, callbacks=cbs)
		print("Done Training")

		# Compute training error statistics
		preds = model.predict(ins)
		errors = outs - preds
		abs_errors = np.abs(errors)
		abs_errors_table = wandb.Table(data=abs_errors, columns=["abs_errors"])
		max_abs_error = np.max(abs_errors)
		sum_abs_errors = np.sum(abs_errors)
		num_abs_errors_over_threshold = len([abs_error for abs_error in abs_errors if abs_error > 0.1])

		# Log trainging error statistics to WandB
		obj = {'abs_errors': abs_errors_table,
			   'max_abs_error': max_abs_error,
			   'sum_abs_errors': sum_abs_errors,
			   'num_abs_errors_over_threshold': num_abs_errors_over_threshold
		}
		wandb.log(obj)

		# Save the training history
		# with open('results/hw0_results_%s.pkl'%(argstring), "wb") as fp:
		# 	pickle.dump(obj, fp)

	# Close WANDB
	wandb.finish()
	
# def display_learning_curve(fname):
#     '''
#     Display the learning curve that is stored in fname
	
#     :param fname: Results file to load and dipslay
	
#     '''
	
#     # Load the history file and display it
#     fp = #TODO
#     # TODO
#     fp.close()
	
#     # Display
#     plt.plot(history['loss'])
#     plt.ylabel('MSE')
#     plt.xlabel('epochs')

# def display_learning_curve_set(dir, base):
#     '''
#     Plot the learning curves for a set of results
	
#     :param base: Directory containing a set of results files
#     '''
#     # Find the list of files in the local directory that match base_[\d]+.pkl
#     files = [f for f in os.listdir(dir) if re.match(r'%s.+.pkl'%(base), f)]
#     files.sort()
	
#     for f in files:
#         with open("%s/%s"%(dir,f), "rb") as fp:
#             history = pickle.load(fp)
#             plt.plot(history['loss'])
#     plt.ylabel('MSE')
#     plt.xlabel('epochs')
#     plt.legend(files)
	
def create_parser():
	'''
	Create a command line parser for the XOR experiment
	'''
	parser = argparse.ArgumentParser(description='XOR Learner')
	parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
	parser.add_argument('--nonlinearity', type=str, default='sigmoid', help='activation function')
	parser.add_argument('--nonlinearity_output', type=str, default='sigmoid', help='activation function in output layer')
	parser.add_argument('--project', type=str, default='HW0', help='Name for WandB')
	parser.add_argument('--label', type=str, default='None', help='Used for organization with WandB')
	parser.add_argument('--exp', type=int, default=0, help='Experiment number')
	parser.add_argument('--hidden', nargs='+', type=int, default=[5], help='Number of hidden units per layer (sequence of ints)')
	parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
	parser.add_argument('--cpus-per-task', type=int, default=None, help='Number of threads to use')
	parser.add_argument('--nogo', action='store_true', help='Do not execute the experiment')
	parser.add_argument('--gpu', action='store_true', help='Use a GPU')
	parser.add_argument('--aggregate', action='store_true', help='Aggregate the results instead of training')
	parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbosity level')
	return parser

'''
This next bit of code is executed only if this python file itself is executed
(if it is imported into another file, then the code below is not executed)
'''
if __name__ == "__main__":
	# Parse the command-line arguments
	parser = create_parser()
	args = parser.parse_args()
	
	# GPU check                                                                                              
	visible_devices = tf.config.get_visible_devices('GPU')
	n_visible_devices = len(visible_devices)

	print('GPUS:', visible_devices)
	if(n_visible_devices > 0):
		for device in visible_devices:
			tf.config.experimental.set_memory_growth(device, True)
		print('We have %d GPUs\n'%n_visible_devices)
	else:
		print('NO GPU')

	# Set number of threads, if it is specified                                                              
	if args.cpus_per_task is not None:
		tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
		tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)


	# Do the work
	if args.aggregate:
		print() # build figures
	else:
		execute_exp(args)
