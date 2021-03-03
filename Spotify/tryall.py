import sys
import argparse
import os
import glob
import json
import numpy as np
import inspect
import importlib
import textwrap
import time

def safe_class_split(arr):
	np.random.seed(0)
	train_arr = []
	val_arr = []
	for i in range(len(np.unique(arr[:,-1].reshape(-1)))):
		indices_of_class_i = np.random.permutation(np.argwhere(arr[:,-1].reshape(-1) == i).reshape(-1))
		to_add_train_arr = arr[indices_of_class_i[:int(.5*float(indices_of_class_i.shape[0]))],:]
		to_add_val_arr = arr[indices_of_class_i[int(.5*float(indices_of_class_i.shape[0])):],:]
		train_arr.append(to_add_train_arr)
		val_arr.append(to_add_val_arr)
	return np.concatenate(train_arr,axis=0), np.concatenate(val_arr,axis=0)


def run_a_btc(client, trainfile, testarr, flag='DT'):
	outfile = str(flag.replace('-','').replace(' ','')) + '.py'
	compiler_cmd = './' + str(client) + ' \'' + str(trainfile) + '\' ' + ('-headerless -f ') + str(flag) + ' -o ' + outfile + ' -riskoverfit --yes'
	print(compiler_cmd)
	os.system(compiler_cmd)
	jsonfile = str(flag.replace('-','').replace(' ','')) + '.json'
	#Access its classification method
	current_lib = importlib.import_module(str(flag.replace('-','').replace(' ','')))
	current_pred = current_lib.classify
	important_idxs = current_lib.important_idxs
	try:
		transform_or_not = current_lib.transform_true
		preds = np.array(current_pred(testarr[:,important_idxs],transform_true=transform_or_not))

	except:
		preds = np.array(current_pred(testarr[:,important_idxs]))
	return 	round(100.0 * float(np.sum(preds.reshape(-1) == testarr[:,-1].reshape(-1)))/float(testarr.shape[0]), 2)



def ensemble_predictors(predictors):
	def ensemble_function(rows):
		predictions = []
		#Calculate How Many Classes Are Present in the Data
		n_classes = len(np.unique(rows[:,-1].reshape(-1)))
		#Loop over each predictor in the ensemble
		for predictor in predictors:
			#Access its classification method
			current_lib = importlib.import_module(str(predictor.replace('-','').replace(' ','')))
			current_pred = current_lib.classify
			important_idxs = current_lib.important_idxs
			try:
				transform_or_not = current_lib.transform_true
				#Append its decisions to a list of decisions for each predictor
				predictions.append(np.array(current_pred(rows[:,important_idxs],transform_true=transform_or_not)).reshape(-1,1))
			except:
				#Append its decisions to a list of decisions for each predictor
				predictions.append(np.array(current_pred(rows[:,important_idxs])).reshape(-1,1))				
		#Treat each predictors decisions as a column in a table
		classifications = np.concatenate(predictions,axis=1).astype('int64')
		#Tally the decisions for each instance being voted on
		bincounts = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_classes), axis=1, arr=classifications)
		#Return the Majority Ruling
		return np.argmax(bincounts, axis=1)
	
	return ensemble_function, textwrap.dedent(inspect.getsource(ensemble_function))

def run(args):
	parser = argparse.ArgumentParser(description='Try all RF, DT, NN, RF -rank, DT -rank, NN -rank, and pick the highest validation accuracy')
	parser.add_argument('DATAFILE', type=str, help="Data file")
	parser.add_argument('BTC_CLIENT', type=str, help="Btc client to be used")
	parser.add_argument('-headerless', action="store_true", help="Wether the data is headerless")
	parser.add_argument('-ensemble',action="store_true",help="Ensemble any model with better than best guess on heldout data together")
	args = parser.parse_args(args)
	since=time.time()
	fileList = glob.glob('clean*')
	for filePath in fileList:
		try:
			os.remove(filePath)
		except:
			print("Error while deleting file : ", filePath)
	print("Cleaning...")
	np.random.seed(0)
	os.system('./' + args.BTC_CLIENT + ' \'' + str(args.DATAFILE) + '\' -cleanonly --yes ' + (' -headerless' if args.headerless else '') )
	fileList = glob.glob('clean*')
	for filePath in fileList:
		if '.csv' in filePath:
			cleanfile = filePath
	print("Done Cleaning!")
	print("Splitting Data...")
	trainarr,valarr = safe_class_split(np.loadtxt(cleanfile,delimiter=',',dtype='float64'))
	train_file = 'train.csv'
	np.savetxt(train_file, trainarr, delimiter=',', fmt='%s')
	val_file = 'val.csv'
	np.savetxt(val_file, valarr, delimiter=',', fmt='%s')	
	print("Done Splitting Data!")
	run_dict = {}
	best_flag = None
	best_acc = -1
	flags = ['DT -rank','NN -rank','RF -rank','RF','NN','DT']
	print("#"*50)
	if args.ensemble:
		to_ensemble = []
		improvements_over_best_guess = []
	for flag in flags:
		print("Running:",flag)
		val_acc =  run_a_btc(args.BTC_CLIENT, train_file, valarr, flag=flag)
		run_dict[flag] = float(val_acc)
		if args.ensemble:
			to_ensemble.append(flag)
		if val_acc > best_acc:
			best_acc = val_acc
			best_flag = flag
		print('Testing on heldout data...')
		print("Using ",flag,"achieved",str(val_acc)+'%','test accuracy')
		print("#"*50)

	print("Done Running!")
	print("Summary:")
	print(run_dict)
	print("Best Test Accuracy:",best_acc)
	print("Using:",best_flag)
	if args.ensemble:
		if best_acc == 100.0:
			print("No use in ensembling, 100% Test Accuracy Achieved")
		else:
			to_ensemble = [flag for flag in to_ensemble if abs(run_dict[flag] - best_acc) < 1]
			if len(to_ensemble) <= 1:
				print("No use in ensembling since the optimal model is significantly better than the others")
			else:
				new_predictor, new_predictor_code = ensemble_predictors(to_ensemble)
				ensembled_preds = new_predictor(valarr[:,:-1])
				ensemble_acc = round(100.0 * float(np.sum(ensembled_preds.reshape(-1) == valarr[:,-1].reshape(-1)))/float(valarr.shape[0]), 2)
				print("List of Models Included in the Ensemble:",to_ensemble)
				print("Test Accuracy if ensembled:", str(ensemble_acc) + '%')
				print("Test Accuracy Improvement from Ensembling:", str(round(ensemble_acc - best_acc,2)) + '%')
				print("New Ensembled Classifier Code:")
				print("#"*35)
				print("predictors = " + str(to_ensemble))
				print(new_predictor_code)
				print("#"*35)

	print("Total Time Elapsed:",int(time.time()-since),'seconds')
	fileList = glob.glob('clean*')
	for filePath in fileList:
		try:
			os.remove(filePath)
		except:
			print("Error while deleting file : ", filePath)
	os.remove(train_file)
	os.remove(val_file)





if __name__ == "__main__":
	run(sys.argv[1:])

