#Make sure your predictor is in the same folder and named predictor.py
from predictor import *
import sys
import argparse
parser = argparse.ArgumentParser(description='Get soft probabilities for a CSV containing rows of features unlabeled')
parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
parser.add_argument('-file_has_labels', action='store_true', help='By default we assume your data is only features, no labels. If it contains labels use this flag to notify our system')
args = parser.parse_args()
if not args.cleanfile:
    cleanfile = tempfile.NamedTemporaryFile().name
    preprocessedfile = tempfile.NamedTemporaryFile().name
    output = preprocess(args.csvfile,preprocessedfile,args.headerless,testfile=(not args.file_has_labels), trim=True)
    get_key, classmapping = clean(preprocessedfile if output!=-1 else args.csvfile, cleanfile, -1, args.headerless, testfile=(not args.file_has_labels), trim=True)
else:
    cleanfile=args.csvfile
    preprocessedfile=args.csvfile
    get_key = lambda x, y: x
    classmapping = {}
arr = np.loadtxt(cleanfile,delimiter=',',dtype='float64')
print("Soft Probabilites for Each Row:")
try:
	from predictor import transform_true
	arr = Normalize(arr)
	o = classify(arr, transform=transform_true, soft_probabilities=True)
	for prob in o:
		print(list(prob))
except:
	o = classify(arr, soft_probabilities=True)
	for prob in o:
		print(list(prob))
#Clean Up
if not args.cleanfile:
    os.remove(cleanfile)
    if output!=-1:
        os.remove(preprocessedfile)
