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
import tempfile
import csv
import binascii
IOBUF = 100000000


def clean_labels(arr):
	#makes the labels 0,...,c-1
	unique_labels = np.unique((arr[:, -1].reshape(-1)))
	if (unique_labels == np.array(list(range(len(unique_labels))))).all():
		return arr
	mappings = {}
	for i, label in enumerate(sorted(list(unique_labels))):
		mappings[label] = i
	for row in arr:
		row[-1]=mappings[row[-1]]

	return arr

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
	compiler_cmd = './' + str(client) + ' \'' + str(trainfile) + '\' -headerless -f ' + str(flag) + ' -o ' + outfile + ' -riskoverfit --yes'
	print(compiler_cmd)
	os.system(compiler_cmd)
	jsonfile = str(flag.replace('-','').replace(' ','')) + '.json'
	#Access its classification method
	current_lib = importlib.import_module(str(flag.replace('-','').replace(' ','')))
	current_pred = current_lib.classify
	important_idxs = current_lib.important_idxs
	try:
		transform_or_not = current_lib.transform_true
		Normalize = current_lib.Normalize
		testarr = Normalize(testarr)
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
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[], trim=False, targetstr='', ignorecolumnstr='',ignorelabelstr=''):
	#This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
	#Precursor to clean

	il=[]

	
	ignorelabelstr = eval(ignorelabelstr.split('=')[-1])
	ignorecolumns = eval(ignorecolumnstr.split('=')[-1])
	targetstr = eval(targetstr.split('=')[-1])
	important_idxs = []
	if ignorelabels == [] and ignorecolumns == [] and target == "":
		return -1
	if not trim:
		ignorecolumns = []
	if (testfile):
		target = ''
		hc = -1 
	with open(outputcsvfile, "w+", encoding='utf-8') as outputfile:
		with open(inputcsvfile, "r", encoding='utf-8') as csvfile:      # hardcoded utf-8 encoding per #717
			reader = csv.reader(csvfile)
			if (headerless == False):
				header=next(reader, None)
				try:
					if not testfile:
						if (target != ''): 
							hc = header.index(target)
						else:
							hc = len(header) - 1
							target=header[hc]
				except:
					raise NameError("Target '" + target + "' not found! Header must be same as in file passed to btc.")
				for i in range(0, len(ignorecolumns)):
					try:
						col = header.index(ignorecolumns[i])
						if not testfile:
							if (col == hc):
								raise ValueError("Attribute '" + ignorecolumns[i] + "' is the target. Header must be same as in file passed to btc.")
						il = il + [col]
					except ValueError:
						raise
					except:
						raise NameError("Attribute '" + ignorecolumns[i] + "' not found in header. Header must be same as in file passed to btc.")
				first = True
				for i in range(0, len(header)):

					if (i == hc):
						continue
					if (i in il):
						continue
					if first:
						first = False
					else:
						print(",", end='', file=outputfile)
					print(header[i], end='', file=outputfile)
				if not testfile:
					print("," + header[hc], file=outputfile)
				else:
					print("", file=outputfile)

				for row in csv.DictReader(open(inputcsvfile, encoding='utf-8')):
					if target and (row[target] in ignorelabels):
						continue
					first = True
					for name in header:
						if (name in ignorecolumns):
							continue
						if (name == target):
							continue
						if first:
							first = False
						else:
							print(",", end='', file=outputfile)
						if (',' in row[name]):
							print('"' + row[name].replace('"', '') + '"', end='', file=outputfile)
						else:
							print(row[name].replace('"', ''), end='', file=outputfile)
					if not testfile:
						print("," + row[target], file=outputfile)
					else:
						if len(important_idxs) == 1:
							print(",", file=outputfile)
						else:
							print("", file=outputfile)

			else:
				try:
					if (target != ""): 
						hc = int(target)
					else:
						hc = -1
				except:
					raise NameError("No header found but attribute name given as target. Header must be same as in file passed to btc.")
				for i in range(0, len(ignorecolumns)):
					try:
						col = int(ignorecolumns[i])
						if (col == hc):
							raise ValueError("Attribute " + str(col) + " is the target. Cannot ignore. Header must be same as in file passed to btc.")
						il = il + [col]
					except ValueError:
						raise
					except:
						raise ValueError("No header found but attribute name given in ignore column list. Header must be same as in file passed to btc.")
				for row in reader:
					first = True
					if (hc == -1) and (not testfile):
						hc = len(row) - 1
					if (row[hc] in ignorelabels):
						continue
					for i in range(0, len(row)):
						if (i in il):
							continue
						if (i == hc):
							continue
						if first:
							first = False
						else:
							print(",", end='', file=outputfile)
						if (',' in row[i]):
							print('"' + row[i].replace('"', '') + '"', end='', file=outputfile)
						else:
							print(row[i].replace('"', ''), end = '', file=outputfile)
					if not testfile:
						print("," + row[hc], file=outputfile)
					else:
						if len(important_idxs) == 1:
							print(",", file=outputfile)
						else:
							print("", file=outputfile)


def cleaner(filename, outfile, rounding=-1, headerless=False, testfile=False, trim=False, mappingstr='', num_attrstr='', n_classesstr='', ignorecolumnstr=''):
	#This function takes a preprocessed csv and cleans it to real numbers for prediction or validation

	transform_true = False
	clean.classlist = []
	clean.testfile = testfile
	clean.mapping = {}
	clean.mapping = eval(mappingstr.split('=')[-1])
	ignorecolumns = eval(ignorecolumnstr.split('=')[-1])
	num_attr = int(num_attrstr.split('=')[-1])
	n_classes = int(n_classesstr.split('=')[-1])
	def convert(cell):
		value = str(cell)
		try:
			result = int(value)
			return result
		except:
			try:
				result=float(value)
				if math.isnan(result):
					#if nan parse to string
					raise ValueError('')
				if (rounding != -1):
					result = int(result * math.pow(10, rounding)) / math.pow(10, rounding)
				return result
			except:
				result = (binascii.crc32(value.encode('utf8')) % (1 << 32))
				return result

	#Function to return key for any value 
	def get_key(val, clean_classmapping):
		if clean_classmapping == {}:
			return val
		for key, value in clean_classmapping.items(): 
			if val == value:
				return key
		if val not in list(clean_classmapping.values):
			raise ValueError("Label key does not exist")


	#Function to convert the class label
	def convertclassid(cell):
		if (clean.testfile):
			return convert(cell)
		value = str(cell)
		if (value == ''):
			raise ValueError("All cells in the target column must contain a class label.")

		if (not clean.mapping == {}):
			result = -1
			try:
				result = clean.mapping[cell]
			except:
				raise ValueError("Class label '" + value + "' encountered in input not defined in user-provided mapping.")
			if (not result == int(result)):
				raise ValueError("Class labels must be mapped to integer.")
			if (not str(result) in clean.classlist):
				clean.classlist = clean.classlist + [str(result)]
			return result
		try:
			result = float(cell)
			if (rounding != -1):
				result = int(result * math.pow(10, rounding)) / math.pow(10, rounding)
			else:
				result = int(int(result * 100) / 100)  # round classes to two digits

			if (not str(result) in clean.classlist):
				clean.classlist = clean.classlist + [str(result)]
		except:
			result = (binascii.crc32(value.encode('utf8')) % (1 << 32))
			if (result in clean.classlist):
				result = clean.classlist.index(result)
			else:
				clean.classlist = clean.classlist + [result]
				result = clean.classlist.index(result)
			if (not result == int(result)):
				raise ValueError("Class labels must be mappable to integer.")
		return result


	#Main Cleaning Code
	rowcount = 0
	with open(filename, encoding='utf-8') as csv_file:
		reader = csv.reader(csv_file)
		f = open(outfile, "w+", encoding='utf-8')
		if (headerless == False):
			next(reader, None)
		outbuf = []
		for row in reader:
			if (row == []):  # Skip empty rows
				continue
			rowcount = rowcount + 1
			if not transform_true:
				rowlen = num_attr if trim else num_attr + len(ignorecolumns)
			else:
				rowlen = num_attr_before_transform if trim else num_attr_before_transform + len(ignorecolumns)
			if (not testfile):
				rowlen = rowlen + 1    
			if ((len(row) - (1 if ((testfile and len(important_idxs) == 1)) else 0))  != rowlen) and not (row == ['','']):
				raise ValueError("Column count must match trained predictor. Row " + str(rowcount) + " differs. Expected Row length: " + str(rowlen) + ", Actual Row Length: " + str(len(row)))
			i = 0
			for elem in row:
				if(i + 1 < len(row)):
					outbuf.append(str(convert(elem)))
					outbuf.append(',')
				else:
					classid = str(convertclassid(elem))
					outbuf.append(classid)
				i = i + 1
			if (len(outbuf) < IOBUF):
				outbuf.append(os.linesep)
			else:
				print(''.join(outbuf), file=f)
				outbuf = []
		print(''.join(outbuf), end="", file=f)
		f.close()

		if (testfile == False and not len(clean.classlist) >= 2):
			raise ValueError("Number of classes must be at least 2.")

		return get_key, clean.mapping

def clean(BTC_CLIENT, DATAFILE, headerless, ignorecolumns, ignorelabels, state=None):
	if state is None:
		client_dir = os.path.split(BTC_CLIENT)[0]
		clean_cmd = './' + BTC_CLIENT + ' \'' + str(DATAFILE) + '\' -cleanonly --yes' + (' -headerless' if headerless else '') + ((' -ignorecolumns ' +str(','.join([x.replace(' ','\\ ') for x in ignorecolumns.split(',')]))) if ignorecolumns else "") + ((' -ignorelabels ' +str(','.join([x.replace(' ','\\ ') for x in ignorelabels.split(',')]))) if ignorelabels else "")
		print(clean_cmd)
		os.system(clean_cmd)
		fileList = list(glob.glob(client_dir + 'clean*')) + list(glob.glob('clean*'))
		for filePath in fileList:
			if '.csv' in filePath:
				cleanfile = filePath
		return cleanfile
	else:
		mappingstr = str(state[3])
		targetstr = str(state[4])
		ignorelabelstr = str(state[5])
		ignorecolumnstr = str(state[6])
		num_attrstr = str(state[1])
		n_classesstr = str(state[2])
		print("State:",state[1:])
		cleanfile = tempfile.NamedTemporaryFile().name
		preprocessedfile = tempfile.NamedTemporaryFile().name
		output = preprocess(DATAFILE,preprocessedfile,ignorecolumnstr=ignorecolumnstr,ignorelabelstr=ignorelabelstr,targetstr=targetstr)
		get_key, classmapping = cleaner(preprocessedfile if output!=-1 else DATAFILE, cleanfile,mappingstr=mappingstr, num_attrstr=num_attrstr, n_classesstr=n_classesstr, ignorecolumnstr=ignorecolumnstr)
		return cleanfile

def run(args):
	parser = argparse.ArgumentParser(description='Try all RF, DT, NN, RF -rank, DT -rank, NN -rank, and pick the highest validation accuracy')
	parser.add_argument('DATAFILE', type=str, help="Data file")
	parser.add_argument('BTC_CLIENT', type=str, help="Btc client to be used")
	parser.add_argument('-headerless', action="store_true", help="Wether the data is headerless")
	parser.add_argument('-ensemble',action="store_true",help="Ensemble any model with better than best guess on heldout data together")
	parser.add_argument('-ignorecolumns', default="", type=str, help='Comma-separated list of attributes to ignore (names or numbers).')
	parser.add_argument('-ignorelabels', default="", type=str, help='Comma-separated list of rows of classes to ignore.')
	parser.add_argument('-testfile',default="",type=str,help='Filename of an already predefined test set(Assumes DATAFILE is the training set)')
	args = parser.parse_args(args)
	since=time.time()
	client_dir = os.path.split(args.BTC_CLIENT)[0]
	fileList = list(glob.glob(client_dir + 'clean*')) + list(glob.glob('clean*')) 
	for filePath in fileList:
		try:
			os.remove(filePath)
		except:
			print("Error while deleting file : ", filePath)
	print("Cleaning...")
	np.random.seed(0)
	cleanfile = clean(args.BTC_CLIENT, args.DATAFILE, args.headerless, args.ignorecolumns, args.ignorelabels)
	cleanarr = clean_labels(np.loadtxt(cleanfile,delimiter=',',dtype='float64'))
	if not args.testfile:
		print("Done Cleaning!")
		print("Splitting Data...")
		trainarr,valarr = safe_class_split(cleanarr)
		train_file = 'train.csv'
		np.savetxt(train_file, trainarr, delimiter=',', fmt='%s')
		val_file = 'val.csv'
		np.savetxt(val_file, valarr, delimiter=',', fmt='%s')	
		print("Done Splitting Data!")
	else:
		print("Test set already provided. Not Splitting. Treating DATAFILE as train set and -testfile as test set")
		trainarr = cleanarr
		train_file = 'train.csv'
		os.system('mv ' + str(cleanfile) + ' ' + str(train_file))
		cleantestfile = clean(args.BTC_CLIENT, args.testfile, args.headerless, args.ignorecolumns, args.ignorelabels, state=open('clean.state', 'r', encoding='utf-8').read().splitlines())
		valarr = clean_labels(np.loadtxt(cleantestfile,delimiter=',',dtype='float64'))
		val_file = 'val.csv'
		np.savetxt(val_file, valarr, delimiter=',', fmt='%s')		
		print("Done Preparing Data!")

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
		print("Testing on heldout data using ",flag,"achieved",str(val_acc)+'%','test accuracy')
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
	fileList = list(glob.glob(client_dir + 'clean*')) + list(glob.glob('clean*'))
	for filePath in fileList:
		try:
			os.remove(filePath)
		except:
			print("Error while deleting file : ", filePath)
	os.remove(train_file)
	os.remove(val_file)





if __name__ == "__main__":
	run(sys.argv[1:])

