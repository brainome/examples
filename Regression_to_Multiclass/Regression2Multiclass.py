import os
import numpy as np
import argparse
import capacity
import subprocess
import sys

def discretize(value, buckets):
	for bucket, real_label in buckets.items():
		if value <= bucket[1]:
			return real_label
	return real_label
def run(args):
	#parse args
	np.random.seed(0)
	parser = argparse.ArgumentParser(description='Convert Regression to Classification')
	parser.add_argument('CLEANDATA', type=str, help='Clean Data to Discretize')
	parser.add_argument('NUMCLASSES', type=int, help='Number of Classes to Discretize to')
	parser.add_argument('-target_col', type=int, default=-1, help='Target Column Index')
	#Parse Args
	args = parser.parse_args(args)
	#Total Tensor Loaded
	tot_tensor = np.loadtxt(args.CLEANDATA, delimiter=',', dtype='float64')
	target = tot_tensor[:, args.target_col]
	non_target_idxs = [i for i in range(int(tot_tensor.shape[1])) if i!=args.target_col]
	tot_tensor = np.concatenate((tot_tensor[:, non_target_idxs], target.reshape(-1, 1)), axis=1)
	print('Unique Values:',list(set(list(tot_tensor[:, -1]))))
	minny = np.min(target)
	maxxy = np.max(target)
	left_num = minny
	delta = (float(maxxy) - float(minny)) / args.NUMCLASSES
	right_num = minny + delta
	buckets = {}
	#Make the quantized buckets
	for i in range(args.NUMCLASSES):
		buckets[(left_num, right_num)] = i
		left_num += delta
		right_num += delta
	print("Discrete Buckets:", buckets)
	#Fill the Buckets
	for i, label in enumerate(target):
		discrete = discretize(label, buckets)
		if discrete not in list(range(args.NUMCLASSES)):
			print('wackylabel:',label)
		tot_tensor[i, -1] = discrete
	#Save Discretized Data
	np.savetxt(args.CLEANDATA[:-4] + '_discrete.csv', tot_tensor, delimiter=',', fmt='%s')



if __name__ == '__main__':
	run(sys.argv[1:])
