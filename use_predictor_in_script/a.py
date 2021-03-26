#!/usr/bin/env python3
#
# This code has been produced by a free evaluation version of Brainome Table Compiler(tm).
# Portions of this code copyright (c) 2019-2021 by Brainome, Inc. All Rights Reserved.
# Brainome grants an exclusive (subject to our continuing rights to use and modify models),
# worldwide, non-sublicensable, and non-transferable limited license to use and modify this
# predictor produced through the input of your data:
# (i) for users accessing the service through a free evaluation account, solely for your
# own non-commercial purposes, including for the purpose of evaluating this service, and
# (ii) for users accessing the service through a paid, commercial use account, for your
# own internal  and commercial purposes.
# Please contact support@brainome.ai with any questions.
# Use of predictions results at your own risk.
#
# Output of Brainome Table Compiler v0.993-rishi-rishi.
# Invocation: btc /home/rishi/mlmeter-data/data/titanic_train_labeled.csv
# Total compiler execution time: 0:00:10.08. Finished on: Mar-26-2021 00:28:15.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        a.py
    Classifier Type:              Random Forest
    System Type:                  Binary classifier
    Training / Validation Split:  60% : 40%
    Accuracy:
      Best-guess accuracy:        61.61%
      Training accuracy:         100.00% (534/534 correct)
      Validation Accuracy:        80.67% (288/357 correct)
      Combined Model Accuracy:    92.25% (822/891 correct)

    Model Capacity (MEC):         12    bits

    Generalization Ratio:         42.07 bits/bit
    Generalization Index:         20.71
    Percent of Data Memorized:     4.83%
    Resilience to Noise:          -1.64 dB
    System Meter Runtime Duration:    1s

    Training Confusion Matrix:
              Actual | Predicted
                Dead |  340    0 
               Alive |    0  194 

    Validation Confusion Matrix:
              Actual | Predicted
                Dead |  185   24 
               Alive |   45  103 

    Combined Confusion Matrix:
              Actual | Predicted
                Dead |  525   24 
               Alive |   45  297 

    Training Accuracy by Class:
               class |   TP   FP   TN   FN     TPR      TNR      PPV      NPV       F1       TS 
                Dead |  340    0  194    0  100.00%  100.00%  100.00%  100.00%  100.00%  100.00%
               Alive |  194    0  340    0  100.00%  100.00%  100.00%  100.00%  100.00%  100.00%

    Validation Accuracy by Class:
               class |   TP   FP   TN   FN     TPR      TNR      PPV      NPV       F1       TS 
                Dead |  185   45  103   24   88.52%   81.10%   80.43%   81.10%   84.28%   72.83%
               Alive |  103   24  185   45   69.59%   80.43%   81.10%   80.43%   74.91%   59.88%

    Combined Accuracy by Class:
               class |   TP   FP   TN   FN     TPR      TNR      PPV      NPV       F1       TS 
                Dead |  525   45  297   24   95.63%   92.52%   92.11%   92.52%   93.83%   88.38%
               Alive |  297   24  525   45   86.84%   92.11%   92.52%   92.11%   89.59%   81.15%

    Attribute Ranking:
                                Sex :   33.92%
                             Pclass :   23.76%
                              Cabin :    7.21%
                           Embarked :    6.58%
                              Parch :    6.43%
                             Ticket :    4.28%
                                Age :    3.73%
                        PassengerId :    3.64%
                               Name :    3.54%
                               Fare :    3.46%
                              SibSp :    3.44%
         
"""

# Imports -- Python3 standard library
import sys
import math
import os
import argparse
import tempfile
import csv
import binascii
import faulthandler


# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF = 100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE = "titanic_train_labeled.csv"

try:
    import numpy as np # For numpy see: http://numpy.org
    from numpy import array
except:
    print("This predictor requires the Numpy library. For installation instructions please refer to: http://numpy.org")

try:
    import multiprocessing
    availmem = 32347435008
    if np.prod(xs.shape) * 30 * 64 * multiprocessing.cpu_count() > availmem:
        procs = math.floor(availmem/np.prod(xs.shape) * 30 * 64)
        if procs == 1: 
            pool = -1
        else:
            pool = multiprocessing.Pool(processes=procs)
    else:
        pool = multiprocessing.Pool()
except:
    pool = -1

#Number of attributes
num_attr = 11
n_classes = 2
transform_true = False

# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target=""
important_idxs=[0,1,2,3,4,5,6,7,8,9,10]
mapping = {}
mapping={'Dead': 0, 'Alive': 1}

def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[], trim=False):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target=""
    important_idxs=[0,1,2,3,4,5,6,7,8,9,10]
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


def clean(filename, outfile, rounding=-1, headerless=False, testfile=False, trim=False):
    #This function takes a preprocessed csv and cleans it to real numbers for prediction or validation


    clean.classlist = []
    clean.testfile = testfile
    clean.mapping = mapping
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
        finally:
            if (result < 0):
                raise ValueError("Integer class labels must be positive and contiguous.")

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
                rowlen = num_attr_before_transform if trim else num_attr_before_transform + len(ignorecolumns)      # noqa
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
def clean_one_line(listy):
    def get_key(val, clean_classmapping):
        if clean_classmapping == {}:
            return val
        for key, value in clean_classmapping.items(): 
            if val == value:
                return key
        if val not in list(clean_classmapping.values):
            raise ValueError("Label key does not exist")
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
    use_cols = [listy[i] for i in important_idxs]
    use_cols = [float(convert(i)) for i in use_cols]
    return use_cols, get_key, mapping

def predict_instance(listy,prob=False):
    clean_row, get_key, cleanmapping = list(clean_one_line(listy))
    returnval = classify([clean_row],return_probabilities=prob)
    if prob:
        return dict([(get_key(i,cleanmapping),float(probab)) for i,probab in enumerate(returnval.reshape(-1))])
    else:
        return get_key(int(returnval),cleanmapping)



# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)
# Classifier
def apply(f, x):
    return f(x)

def booster_0(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [770297728.0, 5.0, 6.0, 9.0], [56.0, 7.0, 8.0, 4.0], [790394816.0, 9.0, 10.0, 10.0], [2666.0, 11.0, 12.0, 7.0], [586.0, 13.0, 14.0, 0.0], [-0.291620851, 0.0, 0.0, 0.0], [-0.0608600043, 0.0, 0.0, 0.0], [347279.0, 15.0, 16.0, 7.0], [17.0, 17.0, 18.0, 4.0], [0.5, 19.0, 20.0, 6.0], [4154891260.0, 21.0, 22.0, 2.0], [1041301120.0, 23.0, 24.0, 2.0], [16.96875, 25.0, 26.0, 8.0], [0.220355168, 0.0, 0.0, 0.0], [0.0190187506, 0.0, 0.0, 0.0], [-0.136410356, 0.0, 0.0, 0.0], [0.169055566, 0.0, 0.0, 0.0], [0.080078952, 0.0, 0.0, 0.0], [-0.182580009, 0.0, 0.0, 0.0], [0.245256722, 0.0, 0.0, 0.0], [-0.0276636388, 0.0, 0.0, 0.0], [0.248972729, 0.0, 0.0, 0.0], [0.0524655208, 0.0, 0.0, 0.0], [0.152150005, 0.0, 0.0, 0.0], [-0.182580009, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_1(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [770297728.0, 5.0, 6.0, 9.0], [56.0, 7.0, 8.0, 4.0], [790394816.0, 9.0, 10.0, 10.0], [2666.0, 11.0, 12.0, 7.0], [586.0, 13.0, 14.0, 0.0], [0.291620851, 0.0, 0.0, 0.0], [0.0608600043, 0.0, 0.0, 0.0], [347279.0, 15.0, 16.0, 7.0], [17.0, 17.0, 18.0, 4.0], [0.5, 19.0, 20.0, 6.0], [4154891260.0, 21.0, 22.0, 2.0], [1041301120.0, 23.0, 24.0, 2.0], [16.96875, 25.0, 26.0, 8.0], [-0.220355168, 0.0, 0.0, 0.0], [-0.0190187506, 0.0, 0.0, 0.0], [0.136410356, 0.0, 0.0, 0.0], [-0.169055566, 0.0, 0.0, 0.0], [-0.080078952, 0.0, 0.0, 0.0], [0.182580009, 0.0, 0.0, 0.0], [-0.245256722, 0.0, 0.0, 0.0], [0.0276636388, 0.0, 0.0, 0.0], [-0.248972729, 0.0, 0.0, 0.0], [-0.0524655208, 0.0, 0.0, 0.0], [-0.152150005, 0.0, 0.0, 0.0], [0.182580009, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_2(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [15.6458502, 5.0, 6.0, 8.0], [56.0, 7.0, 8.0, 4.0], [23.3500004, 9.0, 10.0, 8.0], [341.0, 11.0, 12.0, 0.0], [30066.5, 13.0, 14.0, 7.0], [-0.224820107, 0.0, 0.0, 0.0], [-0.0498724319, 0.0, 0.0, 0.0], [366226.0, 15.0, 16.0, 7.0], [322.0, 17.0, 18.0, 0.0], [314014.0, 19.0, 20.0, 7.0], [2626.0, 21.0, 22.0, 7.0], [1957446270.0, 23.0, 24.0, 2.0], [1.5, 25.0, 26.0, 6.0], [0.0524233468, 0.0, 0.0, 0.0], [-0.119040966, 0.0, 0.0, 0.0], [0.0207303204, 0.0, 0.0, 0.0], [0.229807496, 0.0, 0.0, 0.0], [0.0236836225, 0.0, 0.0, 0.0], [0.162192181, 0.0, 0.0, 0.0], [-0.00302581978, 0.0, 0.0, 0.0], [0.212858126, 0.0, 0.0, 0.0], [-0.199639589, 0.0, 0.0, 0.0], [0.0437711664, 0.0, 0.0, 0.0], [0.14976564, 0.0, 0.0, 0.0], [-0.0550918207, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_3(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [15.6458502, 5.0, 6.0, 8.0], [56.0, 7.0, 8.0, 4.0], [23.3500004, 9.0, 10.0, 8.0], [341.0, 11.0, 12.0, 0.0], [30066.5, 13.0, 14.0, 7.0], [0.224820107, 0.0, 0.0, 0.0], [0.0498724505, 0.0, 0.0, 0.0], [366226.0, 15.0, 16.0, 7.0], [322.0, 17.0, 18.0, 0.0], [314014.0, 19.0, 20.0, 7.0], [2626.0, 21.0, 22.0, 7.0], [1957446270.0, 23.0, 24.0, 2.0], [1.5, 25.0, 26.0, 6.0], [-0.0524233505, 0.0, 0.0, 0.0], [0.119040959, 0.0, 0.0, 0.0], [-0.0207303204, 0.0, 0.0, 0.0], [-0.229807496, 0.0, 0.0, 0.0], [-0.0236836337, 0.0, 0.0, 0.0], [-0.162192196, 0.0, 0.0, 0.0], [0.00302583771, 0.0, 0.0, 0.0], [-0.21285814, 0.0, 0.0, 0.0], [0.199639589, 0.0, 0.0, 0.0], [-0.0437711664, 0.0, 0.0, 0.0], [-0.149765655, 0.0, 0.0, 0.0], [0.0550917983, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_4(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [770297728.0, 5.0, 6.0, 9.0], [56.0, 7.0, 8.0, 4.0], [790394816.0, 9.0, 10.0, 10.0], [2621.5, 11.0, 12.0, 7.0], [550.5, 13.0, 14.0, 0.0], [-0.192658827, 0.0, 0.0, 0.0], [-0.0410240665, 0.0, 0.0, 0.0], [13.0, 15.0, 16.0, 0.0], [17.0, 17.0, 18.0, 4.0], [-0.135374069, 0.0, 0.0, 0.0], [4154891260.0, 19.0, 20.0, 2.0], [50.5, 21.0, 22.0, 4.0], [26.3375015, 23.0, 24.0, 8.0], [-0.175663337, 0.0, 0.0, 0.0], [0.0995580256, 0.0, 0.0, 0.0], [-0.10219007, 0.0, 0.0, 0.0], [0.117633455, 0.0, 0.0, 0.0], [0.14635174, 0.0, 0.0, 0.0], [-0.0514572971, 0.0, 0.0, 0.0], [0.0209148359, 0.0, 0.0, 0.0], [0.205661982, 0.0, 0.0, 0.0], [0.0213493239, 0.0, 0.0, 0.0], [-0.132966951, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 11, 19, 20, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_5(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [770297728.0, 5.0, 6.0, 9.0], [56.0, 7.0, 8.0, 4.0], [790394816.0, 9.0, 10.0, 10.0], [2621.5, 11.0, 12.0, 7.0], [550.5, 13.0, 14.0, 0.0], [0.192658797, 0.0, 0.0, 0.0], [0.0410240665, 0.0, 0.0, 0.0], [13.0, 15.0, 16.0, 0.0], [17.0, 17.0, 18.0, 4.0], [0.135374069, 0.0, 0.0, 0.0], [4154891260.0, 19.0, 20.0, 2.0], [50.5, 21.0, 22.0, 4.0], [26.3375015, 23.0, 24.0, 8.0], [0.175663337, 0.0, 0.0, 0.0], [-0.0995580256, 0.0, 0.0, 0.0], [0.10219007, 0.0, 0.0, 0.0], [-0.117633455, 0.0, 0.0, 0.0], [-0.14635174, 0.0, 0.0, 0.0], [0.0514573008, 0.0, 0.0, 0.0], [-0.0209148303, 0.0, 0.0, 0.0], [-0.205661967, 0.0, 0.0, 0.0], [-0.0213493239, 0.0, 0.0, 0.0], [0.132966965, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 11, 19, 20, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_6(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [15.6458502, 5.0, 6.0, 8.0], [11.25, 7.0, 8.0, 8.0], [371.5, 9.0, 10.0, 0.0], [1842319620.0, 11.0, 12.0, 7.0], [549.5, 13.0, 14.0, 0.0], [-0.0397450887, 0.0, 0.0, 0.0], [-0.174968034, 0.0, 0.0, 0.0], [37.5, 15.0, 16.0, 4.0], [2673.0, 17.0, 18.0, 7.0], [126.5, 19.0, 20.0, 0.0], [3517299200.0, 21.0, 22.0, 2.0], [12.5, 23.0, 24.0, 4.0], [652.0, 25.0, 26.0, 0.0], [-0.0663251802, 0.0, 0.0, 0.0], [0.160766929, 0.0, 0.0, 0.0], [-0.152066052, 0.0, 0.0, 0.0], [0.100633621, 0.0, 0.0, 0.0], [0.0296037048, 0.0, 0.0, 0.0], [0.155602366, 0.0, 0.0, 0.0], [0.00349260145, 0.0, 0.0, 0.0], [0.144036129, 0.0, 0.0, 0.0], [-0.0060859262, 0.0, 0.0, 0.0], [0.124500223, 0.0, 0.0, 0.0], [-0.170466244, 0.0, 0.0, 0.0], [0.00624009641, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_7(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [15.6458502, 5.0, 6.0, 8.0], [11.25, 7.0, 8.0, 8.0], [371.5, 9.0, 10.0, 0.0], [1842319620.0, 11.0, 12.0, 7.0], [549.5, 13.0, 14.0, 0.0], [0.03974507, 0.0, 0.0, 0.0], [0.174968019, 0.0, 0.0, 0.0], [37.5, 15.0, 16.0, 4.0], [2673.0, 17.0, 18.0, 7.0], [126.5, 19.0, 20.0, 0.0], [3517299200.0, 21.0, 22.0, 2.0], [12.5, 23.0, 24.0, 4.0], [652.0, 25.0, 26.0, 0.0], [0.0663251653, 0.0, 0.0, 0.0], [-0.160766944, 0.0, 0.0, 0.0], [0.152066037, 0.0, 0.0, 0.0], [-0.100633629, 0.0, 0.0, 0.0], [-0.0296037234, 0.0, 0.0, 0.0], [-0.155602366, 0.0, 0.0, 0.0], [-0.00349262264, 0.0, 0.0, 0.0], [-0.144036129, 0.0, 0.0, 0.0], [0.00608592806, 0.0, 0.0, 0.0], [-0.12450023, 0.0, 0.0, 0.0], [0.170466244, 0.0, 0.0, 0.0], [-0.00624009501, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_8(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [770297728.0, 5.0, 6.0, 9.0], [56.0, 7.0, 8.0, 4.0], [23.3500004, 9.0, 10.0, 8.0], [2666.0, 11.0, 12.0, 7.0], [1041301120.0, 13.0, 14.0, 2.0], [-0.159039497, 0.0, 0.0, 0.0], [-0.0114730503, 0.0, 0.0, 0.0], [366226.0, 15.0, 16.0, 7.0], [322.0, 17.0, 18.0, 0.0], [0.209999993, 19.0, 20.0, 4.0], [4154891260.0, 21.0, 22.0, 2.0], [558.0, 23.0, 24.0, 0.0], [1796258300.0, 25.0, 26.0, 2.0], [0.0307832304, 0.0, 0.0, 0.0], [-0.0969388038, 0.0, 0.0, 0.0], [0.00952887069, 0.0, 0.0, 0.0], [0.165661499, 0.0, 0.0, 0.0], [0.0680314526, 0.0, 0.0, 0.0], [-0.129550904, 0.0, 0.0, 0.0], [0.109164104, 0.0, 0.0, 0.0], [-0.0583747141, 0.0, 0.0, 0.0], [0.166952595, 0.0, 0.0, 0.0], [-0.0515312776, 0.0, 0.0, 0.0], [-0.150314927, 0.0, 0.0, 0.0], [0.0155766597, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_9(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [770297728.0, 5.0, 6.0, 9.0], [56.0, 7.0, 8.0, 4.0], [23.3500004, 9.0, 10.0, 8.0], [2666.0, 11.0, 12.0, 7.0], [1041301120.0, 13.0, 14.0, 2.0], [0.159039482, 0.0, 0.0, 0.0], [0.0114730587, 0.0, 0.0, 0.0], [366226.0, 15.0, 16.0, 7.0], [322.0, 17.0, 18.0, 0.0], [0.209999993, 19.0, 20.0, 4.0], [4154891260.0, 21.0, 22.0, 2.0], [558.0, 23.0, 24.0, 0.0], [1796258300.0, 25.0, 26.0, 2.0], [-0.0307832211, 0.0, 0.0, 0.0], [0.096938774, 0.0, 0.0, 0.0], [-0.00952888466, 0.0, 0.0, 0.0], [-0.165661514, 0.0, 0.0, 0.0], [-0.0680314526, 0.0, 0.0, 0.0], [0.129550904, 0.0, 0.0, 0.0], [-0.109164096, 0.0, 0.0, 0.0], [0.0583747067, 0.0, 0.0, 0.0], [-0.166952595, 0.0, 0.0, 0.0], [0.051531285, 0.0, 0.0, 0.0], [0.150314927, 0.0, 0.0, 0.0], [-0.0155766541, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_10(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [7.91040039, 5.0, 6.0, 8.0], [11.25, 7.0, 8.0, 8.0], [4164530180.0, 9.0, 10.0, 2.0], [3854757890.0, 11.0, 12.0, 7.0], [20718.0, 13.0, 14.0, 7.0], [-0.011293645, 0.0, 0.0, 0.0], [3199817730.0, 15.0, 16.0, 2.0], [235.0, 17.0, 18.0, 0.0], [0.194792509, 0.0, 0.0, 0.0], [219056032.0, 19.0, 20.0, 2.0], [334.5, 21.0, 22.0, 0.0], [49.5, 23.0, 24.0, 4.0], [1.5, 25.0, 26.0, 6.0], [-0.163254261, 0.0, 0.0, 0.0], [-0.078315191, 0.0, 0.0, 0.0], [-0.0675730482, 0.0, 0.0, 0.0], [0.0331216492, 0.0, 0.0, 0.0], [-0.0392777473, 0.0, 0.0, 0.0], [0.132309988, 0.0, 0.0, 0.0], [-0.162089676, 0.0, 0.0, 0.0], [0.112177156, 0.0, 0.0, 0.0], [-0.0801943466, 0.0, 0.0, 0.0], [0.0940114036, 0.0, 0.0, 0.0], [0.0704100728, 0.0, 0.0, 0.0], [-0.0897729993, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 10, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_11(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [7.91040039, 5.0, 6.0, 8.0], [11.25, 7.0, 8.0, 8.0], [4164530180.0, 9.0, 10.0, 2.0], [3854757890.0, 11.0, 12.0, 7.0], [20718.0, 13.0, 14.0, 7.0], [0.0112936758, 0.0, 0.0, 0.0], [3199817730.0, 15.0, 16.0, 2.0], [235.0, 17.0, 18.0, 0.0], [-0.194792509, 0.0, 0.0, 0.0], [219056032.0, 19.0, 20.0, 2.0], [334.5, 21.0, 22.0, 0.0], [49.5, 23.0, 24.0, 4.0], [1.5, 25.0, 26.0, 6.0], [0.163254261, 0.0, 0.0, 0.0], [0.0783151984, 0.0, 0.0, 0.0], [0.0675730482, 0.0, 0.0, 0.0], [-0.033121638, 0.0, 0.0, 0.0], [0.0392777435, 0.0, 0.0, 0.0], [-0.132309988, 0.0, 0.0, 0.0], [0.162089661, 0.0, 0.0, 0.0], [-0.112177156, 0.0, 0.0, 0.0], [0.0801943466, 0.0, 0.0, 0.0], [-0.0940114111, 0.0, 0.0, 0.0], [-0.0704100728, 0.0, 0.0, 0.0], [0.0897729993, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 10, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_12(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [7.76249981, 5.0, 6.0, 8.0], [769.5, 7.0, 8.0, 0.0], [2673.0, 9.0, 10.0, 7.0], [3854757890.0, 11.0, 12.0, 7.0], [1.5, 13.0, 14.0, 5.0], [3199817730.0, 15.0, 16.0, 2.0], [0.00194126577, 0.0, 0.0, 0.0], [0.5, 17.0, 18.0, 5.0], [10932.5, 19.0, 20.0, 7.0], [2666.0, 21.0, 22.0, 7.0], [287.5, 23.0, 24.0, 0.0], [0.5, 25.0, 26.0, 6.0], [3.5, 27.0, 28.0, 4.0], [-0.156031892, 0.0, 0.0, 0.0], [-0.0625822321, 0.0, 0.0, 0.0], [-0.191779092, 0.0, 0.0, 0.0], [-0.0167160258, 0.0, 0.0, 0.0], [0.128427312, 0.0, 0.0, 0.0], [0.000392786402, 0.0, 0.0, 0.0], [0.0183555111, 0.0, 0.0, 0.0], [0.167632192, 0.0, 0.0, 0.0], [-0.167661846, 0.0, 0.0, 0.0], [0.113406897, 0.0, 0.0, 0.0], [0.0365235582, 0.0, 0.0, 0.0], [-0.0941972136, 0.0, 0.0, 0.0], [0.00229494367, 0.0, 0.0, 0.0], [0.174049199, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_13(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [7.76249981, 5.0, 6.0, 8.0], [769.5, 7.0, 8.0, 0.0], [2673.0, 9.0, 10.0, 7.0], [3854757890.0, 11.0, 12.0, 7.0], [1.5, 13.0, 14.0, 5.0], [3199817730.0, 15.0, 16.0, 2.0], [-0.00194131152, 0.0, 0.0, 0.0], [0.5, 17.0, 18.0, 5.0], [10932.5, 19.0, 20.0, 7.0], [2666.0, 21.0, 22.0, 7.0], [287.5, 23.0, 24.0, 0.0], [0.5, 25.0, 26.0, 6.0], [3.5, 27.0, 28.0, 4.0], [0.156031877, 0.0, 0.0, 0.0], [0.0625822917, 0.0, 0.0, 0.0], [0.191779092, 0.0, 0.0, 0.0], [0.0167160258, 0.0, 0.0, 0.0], [-0.128427312, 0.0, 0.0, 0.0], [-0.00039279167, 0.0, 0.0, 0.0], [-0.0183555204, 0.0, 0.0, 0.0], [-0.167632207, 0.0, 0.0, 0.0], [0.16766189, 0.0, 0.0, 0.0], [-0.113406911, 0.0, 0.0, 0.0], [-0.0365235694, 0.0, 0.0, 0.0], [0.0941972062, 0.0, 0.0, 0.0], [-0.0022949311, 0.0, 0.0, 0.0], [-0.174049199, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_14(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [7.76249981, 5.0, 6.0, 8.0], [56.0, 7.0, 8.0, 4.0], [13.9354496, 9.0, 10.0, 8.0], [3854757890.0, 11.0, 12.0, 7.0], [39.5, 13.0, 14.0, 4.0], [3199817730.0, 15.0, 16.0, 2.0], [0.025711026, 0.0, 0.0, 0.0], [10.8249998, 17.0, 18.0, 8.0], [366037.5, 19.0, 20.0, 7.0], [2666.0, 21.0, 22.0, 7.0], [287.5, 23.0, 24.0, 0.0], [73120080.0, 25.0, 26.0, 9.0], [113282.0, 27.0, 28.0, 7.0], [-0.14697583, 0.0, 0.0, 0.0], [-0.0571794175, 0.0, 0.0, 0.0], [0.0116327051, 0.0, 0.0, 0.0], [-0.229748368, 0.0, 0.0, 0.0], [0.123888969, 0.0, 0.0, 0.0], [-0.0427642986, 0.0, 0.0, 0.0], [0.0143090924, 0.0, 0.0, 0.0], [0.156936124, 0.0, 0.0, 0.0], [-0.128215656, 0.0, 0.0, 0.0], [0.102900356, 0.0, 0.0, 0.0], [0.0289141815, 0.0, 0.0, 0.0], [-0.0927724242, 0.0, 0.0, 0.0], [0.0230915546, 0.0, 0.0, 0.0], [0.141801015, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_15(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [7.76249981, 5.0, 6.0, 8.0], [56.0, 7.0, 8.0, 4.0], [13.9354496, 9.0, 10.0, 8.0], [3854757890.0, 11.0, 12.0, 7.0], [39.5, 13.0, 14.0, 4.0], [3199817730.0, 15.0, 16.0, 2.0], [-0.025711026, 0.0, 0.0, 0.0], [10.8249998, 17.0, 18.0, 8.0], [366037.5, 19.0, 20.0, 7.0], [2666.0, 21.0, 22.0, 7.0], [287.5, 23.0, 24.0, 0.0], [73120080.0, 25.0, 26.0, 9.0], [113282.0, 27.0, 28.0, 7.0], [0.146975845, 0.0, 0.0, 0.0], [0.0571794398, 0.0, 0.0, 0.0], [-0.0116327051, 0.0, 0.0, 0.0], [0.229748368, 0.0, 0.0, 0.0], [-0.123888955, 0.0, 0.0, 0.0], [0.0427642949, 0.0, 0.0, 0.0], [-0.0143091055, 0.0, 0.0, 0.0], [-0.156936109, 0.0, 0.0, 0.0], [0.128215656, 0.0, 0.0, 0.0], [-0.102900371, 0.0, 0.0, 0.0], [-0.0289141778, 0.0, 0.0, 0.0], [0.0927724242, 0.0, 0.0, 0.0], [-0.0230915472, 0.0, 0.0, 0.0], [-0.141801015, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_16(xs):
    #Predicts Class 0
    function_dict = np.array([[2668.5, 1.0, 2.0, 7.0], [0.209999993, 3.0, 4.0, 4.0], [7575.5, 5.0, 6.0, 7.0], [14.8520498, 7.0, 8.0, 8.0], [30.5, 9.0, 10.0, 4.0], [9.70624924, 11.0, 12.0, 8.0], [20718.0, 13.0, 14.0, 7.0], [2628.0, 15.0, 16.0, 7.0], [-0.0923106, 0.0, 0.0, 0.0], [1962866940.0, 17.0, 18.0, 2.0], [0.0061526401, 0.0, 0.0, 0.0], [1342256510.0, 19.0, 20.0, 3.0], [0.171203315, 0.0, 0.0, 0.0], [1.5, 21.0, 22.0, 5.0], [36.5, 23.0, 24.0, 4.0], [0.0169533677, 0.0, 0.0, 0.0], [0.1452647, 0.0, 0.0, 0.0], [-0.0188788213, 0.0, 0.0, 0.0], [-0.26293081, 0.0, 0.0, 0.0], [-0.113734894, 0.0, 0.0, 0.0], [0.12649785, 0.0, 0.0, 0.0], [-0.0984016582, 0.0, 0.0, 0.0], [0.085992001, 0.0, 0.0, 0.0], [0.00429882202, 0.0, 0.0, 0.0], [0.0814842358, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 10, 19, 20, 12, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_17(xs):
    #Predicts Class 1
    function_dict = np.array([[2668.5, 1.0, 2.0, 7.0], [0.209999993, 3.0, 4.0, 4.0], [7575.5, 5.0, 6.0, 7.0], [14.8520498, 7.0, 8.0, 8.0], [30.5, 9.0, 10.0, 4.0], [9.70624924, 11.0, 12.0, 8.0], [20718.0, 13.0, 14.0, 7.0], [2628.0, 15.0, 16.0, 7.0], [0.0923106, 0.0, 0.0, 0.0], [1962866940.0, 17.0, 18.0, 2.0], [-0.00615264382, 0.0, 0.0, 0.0], [1342256510.0, 19.0, 20.0, 3.0], [-0.171203315, 0.0, 0.0, 0.0], [1.5, 21.0, 22.0, 5.0], [36.5, 23.0, 24.0, 4.0], [-0.016953364, 0.0, 0.0, 0.0], [-0.145264715, 0.0, 0.0, 0.0], [0.0188788027, 0.0, 0.0, 0.0], [0.26293081, 0.0, 0.0, 0.0], [0.113734894, 0.0, 0.0, 0.0], [-0.12649785, 0.0, 0.0, 0.0], [0.0984016731, 0.0, 0.0, 0.0], [-0.0859920159, 0.0, 0.0, 0.0], [-0.00429883692, 0.0, 0.0, 0.0], [-0.0814842358, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 10, 19, 20, 12, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_18(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [1860468100.0, 5.0, 6.0, 7.0], [11.25, 7.0, 8.0, 8.0], [790394816.0, 9.0, 10.0, 10.0], [347085.5, 11.0, 12.0, 7.0], [3443318270.0, 13.0, 14.0, 2.0], [0.0167964566, 0.0, 0.0, 0.0], [315.0, 15.0, 16.0, 0.0], [347074.0, 17.0, 18.0, 7.0], [17.0, 19.0, 20.0, 4.0], [3233670910.0, 21.0, 22.0, 2.0], [27.0541496, 23.0, 24.0, 8.0], [4045140480.0, 25.0, 26.0, 7.0], [4067799810.0, 27.0, 28.0, 2.0], [-0.049691882, 0.0, 0.0, 0.0], [-0.143812612, 0.0, 0.0, 0.0], [0.132144362, 0.0, 0.0, 0.0], [-0.00175075489, 0.0, 0.0, 0.0], [-0.0876934081, 0.0, 0.0, 0.0], [0.0595273636, 0.0, 0.0, 0.0], [0.040819075, 0.0, 0.0, 0.0], [-0.0886681303, 0.0, 0.0, 0.0], [0.0588619709, 0.0, 0.0, 0.0], [0.163209155, 0.0, 0.0, 0.0], [-0.141950712, 0.0, 0.0, 0.0], [0.113330409, 0.0, 0.0, 0.0], [0.135357931, 0.0, 0.0, 0.0], [-0.00835862663, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_19(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [1860468100.0, 5.0, 6.0, 7.0], [11.25, 7.0, 8.0, 8.0], [790394816.0, 9.0, 10.0, 10.0], [347085.5, 11.0, 12.0, 7.0], [3443318270.0, 13.0, 14.0, 2.0], [-0.0167964362, 0.0, 0.0, 0.0], [315.0, 15.0, 16.0, 0.0], [347074.0, 17.0, 18.0, 7.0], [17.0, 19.0, 20.0, 4.0], [3233670910.0, 21.0, 22.0, 2.0], [27.0541496, 23.0, 24.0, 8.0], [4045140480.0, 25.0, 26.0, 7.0], [4067799810.0, 27.0, 28.0, 2.0], [0.0496919118, 0.0, 0.0, 0.0], [0.143812612, 0.0, 0.0, 0.0], [-0.132144362, 0.0, 0.0, 0.0], [0.0017507528, 0.0, 0.0, 0.0], [0.0876933932, 0.0, 0.0, 0.0], [-0.0595273525, 0.0, 0.0, 0.0], [-0.040819075, 0.0, 0.0, 0.0], [0.0886681452, 0.0, 0.0, 0.0], [-0.0588619597, 0.0, 0.0, 0.0], [-0.163209155, 0.0, 0.0, 0.0], [0.141950697, 0.0, 0.0, 0.0], [-0.113330416, 0.0, 0.0, 0.0], [-0.135357916, 0.0, 0.0, 0.0], [0.0083586229, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_20(xs):
    #Predicts Class 0
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [372928.0, 3.0, 4.0, 7.0], [9.5, 5.0, 6.0, 4.0], [7.68335009, 7.0, 8.0, 8.0], [0.5, 9.0, 10.0, 5.0], [2.5, 11.0, 12.0, 5.0], [4164530180.0, 13.0, 14.0, 2.0], [391.0, 15.0, 16.0, 0.0], [7.88749981, 17.0, 18.0, 8.0], [0.198918909, 0.0, 0.0, 0.0], [0.0534947552, 0.0, 0.0, 0.0], [15.5728998, 19.0, 20.0, 8.0], [347085.0, 21.0, 22.0, 7.0], [1.5, 23.0, 24.0, 1.0], [0.157571658, 0.0, 0.0, 0.0], [0.0126298713, 0.0, 0.0, 0.0], [0.142149583, 0.0, 0.0, 0.0], [-0.140375257, 0.0, 0.0, 0.0], [0.0295109823, 0.0, 0.0, 0.0], [-0.0644784868, 0.0, 0.0, 0.0], [-0.279010177, 0.0, 0.0, 0.0], [-0.169943765, 0.0, 0.0, 0.0], [0.133189544, 0.0, 0.0, 0.0], [-0.0467531309, 0.0, 0.0, 0.0], [0.0225172434, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 19, 20, 21, 22, 23, 24, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_21(xs):
    #Predicts Class 1
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [372928.0, 3.0, 4.0, 7.0], [9.5, 5.0, 6.0, 4.0], [7.68335009, 7.0, 8.0, 8.0], [0.5, 9.0, 10.0, 5.0], [2.5, 11.0, 12.0, 5.0], [4164530180.0, 13.0, 14.0, 2.0], [391.0, 15.0, 16.0, 0.0], [7.88749981, 17.0, 18.0, 8.0], [-0.198918909, 0.0, 0.0, 0.0], [-0.0534947477, 0.0, 0.0, 0.0], [15.5728998, 19.0, 20.0, 8.0], [347085.0, 21.0, 22.0, 7.0], [1.5, 23.0, 24.0, 1.0], [-0.157571658, 0.0, 0.0, 0.0], [-0.0126298787, 0.0, 0.0, 0.0], [-0.142149583, 0.0, 0.0, 0.0], [0.140375271, 0.0, 0.0, 0.0], [-0.0295109879, 0.0, 0.0, 0.0], [0.0644784719, 0.0, 0.0, 0.0], [0.279010177, 0.0, 0.0, 0.0], [0.16994375, 0.0, 0.0, 0.0], [-0.133189559, 0.0, 0.0, 0.0], [0.0467531309, 0.0, 0.0, 0.0], [-0.0225172509, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 19, 20, 21, 22, 23, 24, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_22(xs):
    #Predicts Class 0
    function_dict = np.array([[7.1333499, 1.0, 2.0, 8.0], [287.5, 3.0, 4.0, 0.0], [2668.5, 5.0, 6.0, 7.0], [0.0222701151, 0.0, 0.0, 0.0], [0.132363603, 0.0, 0.0, 0.0], [0.209999993, 7.0, 8.0, 4.0], [7575.5, 9.0, 10.0, 7.0], [14.8520498, 11.0, 12.0, 8.0], [24.5, 13.0, 14.0, 4.0], [9.70624924, 15.0, 16.0, 8.0], [37.7520981, 17.0, 18.0, 8.0], [0.0945735201, 0.0, 0.0, 0.0], [-0.0705159977, 0.0, 0.0, 0.0], [-0.198263809, 0.0, 0.0, 0.0], [-0.0252764858, 0.0, 0.0, 0.0], [0.00813086517, 0.0, 0.0, 0.0], [0.152452469, 0.0, 0.0, 0.0], [-0.0111361723, 0.0, 0.0, 0.0], [0.0449309796, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 11, 12, 13, 14, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 2, 5, 7, 8, 6, 9, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_23(xs):
    #Predicts Class 1
    function_dict = np.array([[7.1333499, 1.0, 2.0, 8.0], [287.5, 3.0, 4.0, 0.0], [2668.5, 5.0, 6.0, 7.0], [-0.0222701039, 0.0, 0.0, 0.0], [-0.132363603, 0.0, 0.0, 0.0], [0.209999993, 7.0, 8.0, 4.0], [7575.5, 9.0, 10.0, 7.0], [14.8520498, 11.0, 12.0, 8.0], [24.5, 13.0, 14.0, 4.0], [9.70624924, 15.0, 16.0, 8.0], [37.7520981, 17.0, 18.0, 8.0], [-0.0945735276, 0.0, 0.0, 0.0], [0.0705159977, 0.0, 0.0, 0.0], [0.198263809, 0.0, 0.0, 0.0], [0.0252764802, 0.0, 0.0, 0.0], [-0.00813086517, 0.0, 0.0, 0.0], [-0.152452469, 0.0, 0.0, 0.0], [0.0111361686, 0.0, 0.0, 0.0], [-0.0449309945, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 11, 12, 13, 14, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 2, 5, 7, 8, 6, 9, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_24(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [1.5, 3.0, 4.0, 1.0], [7.76249981, 5.0, 6.0, 8.0], [-0.136059031, 0.0, 0.0, 0.0], [250650.0, 7.0, 8.0, 7.0], [3854757890.0, 9.0, 10.0, 7.0], [7.7979002, 11.0, 12.0, 8.0], [0.5, 13.0, 14.0, 5.0], [235.0, 15.0, 16.0, 0.0], [2666.0, 17.0, 18.0, 7.0], [27.5, 19.0, 20.0, 4.0], [394.0, 21.0, 22.0, 0.0], [1860468100.0, 23.0, 24.0, 7.0], [-0.143128708, 0.0, 0.0, 0.0], [0.0502856672, 0.0, 0.0, 0.0], [-0.0595022328, 0.0, 0.0, 0.0], [0.0619424433, 0.0, 0.0, 0.0], [0.00177583063, 0.0, 0.0, 0.0], [0.143989146, 0.0, 0.0, 0.0], [-0.107233711, 0.0, 0.0, 0.0], [0.111677974, 0.0, 0.0, 0.0], [-0.300581008, 0.0, 0.0, 0.0], [0.0868356526, 0.0, 0.0, 0.0], [0.0290604327, 0.0, 0.0, 0.0], [-0.0502469949, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 10, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_25(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [1.5, 3.0, 4.0, 1.0], [7.76249981, 5.0, 6.0, 8.0], [0.136059016, 0.0, 0.0, 0.0], [250650.0, 7.0, 8.0, 7.0], [3854757890.0, 9.0, 10.0, 7.0], [7.7979002, 11.0, 12.0, 8.0], [0.5, 13.0, 14.0, 5.0], [235.0, 15.0, 16.0, 0.0], [2666.0, 17.0, 18.0, 7.0], [27.5, 19.0, 20.0, 4.0], [394.0, 21.0, 22.0, 0.0], [1860468100.0, 23.0, 24.0, 7.0], [0.143128708, 0.0, 0.0, 0.0], [-0.0502856635, 0.0, 0.0, 0.0], [0.0595022365, 0.0, 0.0, 0.0], [-0.0619424321, 0.0, 0.0, 0.0], [-0.0017758332, 0.0, 0.0, 0.0], [-0.143989146, 0.0, 0.0, 0.0], [0.107233718, 0.0, 0.0, 0.0], [-0.111677982, 0.0, 0.0, 0.0], [0.300581008, 0.0, 0.0, 0.0], [-0.0868356675, 0.0, 0.0, 0.0], [-0.0290604364, 0.0, 0.0, 0.0], [0.0502470136, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 10, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_26(xs):
    #Predicts Class 0
    function_dict = np.array([[914675200.0, 1.0, 2.0, 2.0], [18.375, 3.0, 4.0, 8.0], [1323098110.0, 5.0, 6.0, 2.0], [31.5, 7.0, 8.0, 4.0], [292438.5, 9.0, 10.0, 7.0], [349623.0, 11.0, 12.0, 7.0], [18.5, 13.0, 14.0, 4.0], [7.76249981, 15.0, 16.0, 8.0], [8.35210037, 17.0, 18.0, 8.0], [380.0, 19.0, 20.0, 0.0], [1342256510.0, 21.0, 22.0, 3.0], [1046070780.0, 23.0, 24.0, 2.0], [9.0, 25.0, 26.0, 4.0], [0.209999993, 27.0, 28.0, 4.0], [86.0, 29.0, 30.0, 0.0], [0.0264813751, 0.0, 0.0, 0.0], [0.154593736, 0.0, 0.0, 0.0], [-0.183400705, 0.0, 0.0, 0.0], [0.0613520928, 0.0, 0.0, 0.0], [0.0222489499, 0.0, 0.0, 0.0], [-0.200602934, 0.0, 0.0, 0.0], [-0.0524659343, 0.0, 0.0, 0.0], [0.0715919062, 0.0, 0.0, 0.0], [0.117227688, 0.0, 0.0, 0.0], [-0.0764657855, 0.0, 0.0, 0.0], [0.0285915006, 0.0, 0.0, 0.0], [-0.261451334, 0.0, 0.0, 0.0], [0.0373168513, 0.0, 0.0, 0.0], [-0.0913549587, 0.0, 0.0, 0.0], [-0.0919404775, 0.0, 0.0, 0.0], [0.0385848545, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_27(xs):
    #Predicts Class 1
    function_dict = np.array([[914675200.0, 1.0, 2.0, 2.0], [18.375, 3.0, 4.0, 8.0], [1323098110.0, 5.0, 6.0, 2.0], [31.5, 7.0, 8.0, 4.0], [292438.5, 9.0, 10.0, 7.0], [349623.0, 11.0, 12.0, 7.0], [18.5, 13.0, 14.0, 4.0], [7.76249981, 15.0, 16.0, 8.0], [8.35210037, 17.0, 18.0, 8.0], [380.0, 19.0, 20.0, 0.0], [1342256510.0, 21.0, 22.0, 3.0], [1046070780.0, 23.0, 24.0, 2.0], [9.0, 25.0, 26.0, 4.0], [0.209999993, 27.0, 28.0, 4.0], [86.0, 29.0, 30.0, 0.0], [-0.0264813658, 0.0, 0.0, 0.0], [-0.154593736, 0.0, 0.0, 0.0], [0.183400705, 0.0, 0.0, 0.0], [-0.0613521002, 0.0, 0.0, 0.0], [-0.0222489517, 0.0, 0.0, 0.0], [0.200602934, 0.0, 0.0, 0.0], [0.052465938, 0.0, 0.0, 0.0], [-0.0715919062, 0.0, 0.0, 0.0], [-0.117227688, 0.0, 0.0, 0.0], [0.0764657855, 0.0, 0.0, 0.0], [-0.0285915211, 0.0, 0.0, 0.0], [0.261451334, 0.0, 0.0, 0.0], [-0.0373168476, 0.0, 0.0, 0.0], [0.0913549513, 0.0, 0.0, 0.0], [0.09194047, 0.0, 0.0, 0.0], [-0.0385848545, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_28(xs):
    #Predicts Class 0
    function_dict = np.array([[58474200.0, 1.0, 2.0, 9.0], [654.5, 3.0, 4.0, 0.0], [818.5, 5.0, 6.0, 0.0], [65.5, 7.0, 8.0, 0.0], [7.73960018, 9.0, 10.0, 8.0], [11.4250002, 11.0, 12.0, 8.0], [-0.232204139, 0.0, 0.0, 0.0], [3891039740.0, 13.0, 14.0, 2.0], [88.0, 15.0, 16.0, 0.0], [7.2270999, 17.0, 18.0, 8.0], [474191744.0, 19.0, 20.0, 7.0], [0.128943413, 0.0, 0.0, 0.0], [26.46875, 21.0, 22.0, 8.0], [0.130549669, 0.0, 0.0, 0.0], [-0.136046112, 0.0, 0.0, 0.0], [-0.145444572, 0.0, 0.0, 0.0], [-0.00474275462, 0.0, 0.0, 0.0], [0.0920705497, 0.0, 0.0, 0.0], [-0.14892827, 0.0, 0.0, 0.0], [0.106914073, 0.0, 0.0, 0.0], [-0.0412102118, 0.0, 0.0, 0.0], [-0.184650436, 0.0, 0.0, 0.0], [-0.000426885439, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 11, 21, 22, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_29(xs):
    #Predicts Class 1
    function_dict = np.array([[58474200.0, 1.0, 2.0, 9.0], [654.5, 3.0, 4.0, 0.0], [818.5, 5.0, 6.0, 0.0], [65.5, 7.0, 8.0, 0.0], [7.73960018, 9.0, 10.0, 8.0], [11.4250002, 11.0, 12.0, 8.0], [0.232204139, 0.0, 0.0, 0.0], [3891039740.0, 13.0, 14.0, 2.0], [88.0, 15.0, 16.0, 0.0], [7.2270999, 17.0, 18.0, 8.0], [474191744.0, 19.0, 20.0, 7.0], [-0.128943399, 0.0, 0.0, 0.0], [26.46875, 21.0, 22.0, 8.0], [-0.130549669, 0.0, 0.0, 0.0], [0.136046112, 0.0, 0.0, 0.0], [0.145444557, 0.0, 0.0, 0.0], [0.00474274717, 0.0, 0.0, 0.0], [-0.0920705497, 0.0, 0.0, 0.0], [0.148928255, 0.0, 0.0, 0.0], [-0.106914058, 0.0, 0.0, 0.0], [0.0412101969, 0.0, 0.0, 0.0], [0.184650436, 0.0, 0.0, 0.0], [0.000426887738, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 11, 21, 22, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_30(xs):
    #Predicts Class 0
    function_dict = np.array([[4.5, 1.0, 2.0, 5.0], [35.5, 3.0, 4.0, 4.0], [0.127950653, 0.0, 0.0, 0.0], [33.5, 5.0, 6.0, 4.0], [184.0, 7.0, 8.0, 0.0], [2626.5, 9.0, 10.0, 7.0], [26.1437492, 11.0, 12.0, 8.0], [0.147010729, 0.0, 0.0, 0.0], [578.5, 13.0, 14.0, 0.0], [-0.140757248, 0.0, 0.0, 0.0], [0.00265440135, 0.0, 0.0, 0.0], [-0.0343902744, 0.0, 0.0, 0.0], [-0.259776533, 0.0, 0.0, 0.0], [-0.0546604209, 0.0, 0.0, 0.0], [0.0773425996, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 7, 13, 14, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_31(xs):
    #Predicts Class 1
    function_dict = np.array([[4.5, 1.0, 2.0, 5.0], [35.5, 3.0, 4.0, 4.0], [-0.127950668, 0.0, 0.0, 0.0], [33.5, 5.0, 6.0, 4.0], [184.0, 7.0, 8.0, 0.0], [2626.5, 9.0, 10.0, 7.0], [26.1437492, 11.0, 12.0, 8.0], [-0.147010729, 0.0, 0.0, 0.0], [578.5, 13.0, 14.0, 0.0], [0.140757248, 0.0, 0.0, 0.0], [-0.00265440368, 0.0, 0.0, 0.0], [0.0343902819, 0.0, 0.0, 0.0], [0.259776533, 0.0, 0.0, 0.0], [0.0546604246, 0.0, 0.0, 0.0], [-0.0773425922, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 7, 13, 14, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_32(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [1.5, 3.0, 4.0, 1.0], [3236726780.0, 5.0, 6.0, 2.0], [-0.12541528, 0.0, 0.0, 0.0], [3222181890.0, 7.0, 8.0, 2.0], [2904583680.0, 9.0, 10.0, 2.0], [3437313790.0, 11.0, 12.0, 2.0], [71805064.0, 13.0, 14.0, 7.0], [7.82289982, 15.0, 16.0, 8.0], [388.0, 17.0, 18.0, 0.0], [0.155103058, 0.0, 0.0, 0.0], [275975.5, 19.0, 20.0, 7.0], [3776773630.0, 21.0, 22.0, 2.0], [-0.0548408218, 0.0, 0.0, 0.0], [0.0421917364, 0.0, 0.0, 0.0], [-0.115035921, 0.0, 0.0, 0.0], [0.104278944, 0.0, 0.0, 0.0], [0.0540658273, 0.0, 0.0, 0.0], [-0.0202206038, 0.0, 0.0, 0.0], [-0.0495283715, 0.0, 0.0, 0.0], [-0.322985858, 0.0, 0.0, 0.0], [0.150979042, 0.0, 0.0, 0.0], [-0.0290122386, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 10, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_33(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [1.5, 3.0, 4.0, 1.0], [3236726780.0, 5.0, 6.0, 2.0], [0.125415266, 0.0, 0.0, 0.0], [3222181890.0, 7.0, 8.0, 2.0], [2904583680.0, 9.0, 10.0, 2.0], [3437313790.0, 11.0, 12.0, 2.0], [71805064.0, 13.0, 14.0, 7.0], [7.82289982, 15.0, 16.0, 8.0], [388.0, 17.0, 18.0, 0.0], [-0.155103058, 0.0, 0.0, 0.0], [275975.5, 19.0, 20.0, 7.0], [3776773630.0, 21.0, 22.0, 2.0], [0.054840818, 0.0, 0.0, 0.0], [-0.0421917215, 0.0, 0.0, 0.0], [0.115035921, 0.0, 0.0, 0.0], [-0.104278937, 0.0, 0.0, 0.0], [-0.0540658273, 0.0, 0.0, 0.0], [0.0202206038, 0.0, 0.0, 0.0], [0.0495283939, 0.0, 0.0, 0.0], [0.322985828, 0.0, 0.0, 0.0], [-0.150979057, 0.0, 0.0, 0.0], [0.0290122367, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 10, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_34(xs):
    #Predicts Class 0
    function_dict = np.array([[3233494530.0, 1.0, 2.0, 9.0], [639408256.0, 3.0, 4.0, 2.0], [298428.0, 5.0, 6.0, 7.0], [36.125, 7.0, 8.0, 8.0], [4035160830.0, 9.0, 10.0, 7.0], [45.4312515, 11.0, 12.0, 8.0], [0.0313047431, 0.0, 0.0, 0.0], [246298784.0, 13.0, 14.0, 2.0], [0.5, 15.0, 16.0, 6.0], [3962790400.0, 17.0, 18.0, 7.0], [0.129476279, 0.0, 0.0, 0.0], [-0.173281983, 0.0, 0.0, 0.0], [-0.0140814446, 0.0, 0.0, 0.0], [0.00273985136, 0.0, 0.0, 0.0], [0.133326158, 0.0, 0.0, 0.0], [0.000400870777, 0.0, 0.0, 0.0], [-0.124796495, 0.0, 0.0, 0.0], [-0.00401091389, 0.0, 0.0, 0.0], [-0.144707009, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 10, 11, 12, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_35(xs):
    #Predicts Class 1
    function_dict = np.array([[3233494530.0, 1.0, 2.0, 9.0], [639408256.0, 3.0, 4.0, 2.0], [298428.0, 5.0, 6.0, 7.0], [36.125, 7.0, 8.0, 8.0], [4035160830.0, 9.0, 10.0, 7.0], [45.4312515, 11.0, 12.0, 8.0], [-0.0313047543, 0.0, 0.0, 0.0], [246298784.0, 13.0, 14.0, 2.0], [0.5, 15.0, 16.0, 6.0], [3962790400.0, 17.0, 18.0, 7.0], [-0.129476279, 0.0, 0.0, 0.0], [0.173281997, 0.0, 0.0, 0.0], [0.0140814576, 0.0, 0.0, 0.0], [-0.00273985136, 0.0, 0.0, 0.0], [-0.133326158, 0.0, 0.0, 0.0], [-0.000400867575, 0.0, 0.0, 0.0], [0.124796495, 0.0, 0.0, 0.0], [0.00401091576, 0.0, 0.0, 0.0], [0.144707009, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 10, 11, 12, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_36(xs):
    #Predicts Class 0
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [372928.0, 3.0, 4.0, 7.0], [9.5, 5.0, 6.0, 4.0], [366038.5, 7.0, 8.0, 7.0], [0.158841521, 0.0, 0.0, 0.0], [2.5, 9.0, 10.0, 5.0], [0.5, 11.0, 12.0, 5.0], [124.5, 13.0, 14.0, 0.0], [1342256510.0, 15.0, 16.0, 3.0], [15.5728998, 17.0, 18.0, 8.0], [347085.0, 19.0, 20.0, 7.0], [658073216.0, 21.0, 22.0, 2.0], [959511616.0, 23.0, 24.0, 9.0], [-0.0775765106, 0.0, 0.0, 0.0], [0.0515033156, 0.0, 0.0, 0.0], [-0.13589552, 0.0, 0.0, 0.0], [0.00323171332, 0.0, 0.0, 0.0], [-0.0421041809, 0.0, 0.0, 0.0], [-0.191150606, 0.0, 0.0, 0.0], [-0.0999387503, 0.0, 0.0, 0.0], [0.111937642, 0.0, 0.0, 0.0], [0.068696335, 0.0, 0.0, 0.0], [-0.0273390133, 0.0, 0.0, 0.0], [0.0858022273, 0.0, 0.0, 0.0], [-0.0719902664, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 17, 18, 19, 20, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 9, 10, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_37(xs):
    #Predicts Class 1
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [372928.0, 3.0, 4.0, 7.0], [9.5, 5.0, 6.0, 4.0], [366038.5, 7.0, 8.0, 7.0], [-0.158841521, 0.0, 0.0, 0.0], [2.5, 9.0, 10.0, 5.0], [0.5, 11.0, 12.0, 5.0], [124.5, 13.0, 14.0, 0.0], [1342256510.0, 15.0, 16.0, 3.0], [15.5728998, 17.0, 18.0, 8.0], [347085.0, 19.0, 20.0, 7.0], [658073216.0, 21.0, 22.0, 2.0], [959511616.0, 23.0, 24.0, 9.0], [0.0775765106, 0.0, 0.0, 0.0], [-0.0515033156, 0.0, 0.0, 0.0], [0.13589552, 0.0, 0.0, 0.0], [-0.00323172635, 0.0, 0.0, 0.0], [0.0421041772, 0.0, 0.0, 0.0], [0.191150606, 0.0, 0.0, 0.0], [0.0999387652, 0.0, 0.0, 0.0], [-0.111937642, 0.0, 0.0, 0.0], [-0.0686963424, 0.0, 0.0, 0.0], [0.0273390114, 0.0, 0.0, 0.0], [-0.0858022422, 0.0, 0.0, 0.0], [0.071990259, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 17, 18, 19, 20, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 9, 10, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_38(xs):
    #Predicts Class 0
    function_dict = np.array([[4218452480.0, 1.0, 2.0, 2.0], [4062966020.0, 3.0, 4.0, 2.0], [0.0984578803, 0.0, 0.0, 0.0], [4017780220.0, 5.0, 6.0, 2.0], [70438032.0, 7.0, 8.0, 7.0], [1342256510.0, 9.0, 10.0, 3.0], [0.133433536, 0.0, 0.0, 0.0], [849733120.0, 11.0, 12.0, 9.0], [3668878850.0, 13.0, 14.0, 7.0], [-0.0290861838, 0.0, 0.0, 0.0], [0.0136775365, 0.0, 0.0, 0.0], [-0.192683384, 0.0, 0.0, 0.0], [-0.0497660786, 0.0, 0.0, 0.0], [0.0135213789, 0.0, 0.0, 0.0], [0.0806882083, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 6, 11, 12, 13, 14, 2])
    branch_indices = np.array([0, 1, 3, 5, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_39(xs):
    #Predicts Class 1
    function_dict = np.array([[4218452480.0, 1.0, 2.0, 2.0], [4062966020.0, 3.0, 4.0, 2.0], [-0.0984578878, 0.0, 0.0, 0.0], [4017780220.0, 5.0, 6.0, 2.0], [70438032.0, 7.0, 8.0, 7.0], [1342256510.0, 9.0, 10.0, 3.0], [-0.133433521, 0.0, 0.0, 0.0], [849733120.0, 11.0, 12.0, 9.0], [3668878850.0, 13.0, 14.0, 7.0], [0.0290861838, 0.0, 0.0, 0.0], [-0.0136775328, 0.0, 0.0, 0.0], [0.192683384, 0.0, 0.0, 0.0], [0.0497660786, 0.0, 0.0, 0.0], [-0.0135213789, 0.0, 0.0, 0.0], [-0.080688186, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 6, 11, 12, 13, 14, 2])
    branch_indices = np.array([0, 1, 3, 5, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_40(xs):
    #Predicts Class 0
    function_dict = np.array([[3.0, 1.0, 2.0, 6.0], [7.1333499, 3.0, 4.0, 8.0], [0.109385893, 0.0, 0.0, 0.0], [287.5, 5.0, 6.0, 0.0], [0.209999993, 7.0, 8.0, 4.0], [-0.00286557339, 0.0, 0.0, 0.0], [0.115676679, 0.0, 0.0, 0.0], [372928.0, 9.0, 10.0, 7.0], [5.5, 11.0, 12.0, 4.0], [-0.00423527462, 0.0, 0.0, 0.0], [0.141931757, 0.0, 0.0, 0.0], [-0.100640886, 0.0, 0.0, 0.0], [-0.00665144529, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([5, 6, 9, 10, 11, 12, 2])
    branch_indices = np.array([0, 1, 3, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_41(xs):
    #Predicts Class 1
    function_dict = np.array([[3.0, 1.0, 2.0, 6.0], [7.1333499, 3.0, 4.0, 8.0], [-0.1093859, 0.0, 0.0, 0.0], [287.5, 5.0, 6.0, 0.0], [0.209999993, 7.0, 8.0, 4.0], [0.00286557293, 0.0, 0.0, 0.0], [-0.115676686, 0.0, 0.0, 0.0], [372928.0, 9.0, 10.0, 7.0], [5.5, 11.0, 12.0, 4.0], [0.00423527556, 0.0, 0.0, 0.0], [-0.141931757, 0.0, 0.0, 0.0], [0.100640886, 0.0, 0.0, 0.0], [0.00665144436, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([5, 6, 9, 10, 11, 12, 2])
    branch_indices = np.array([0, 1, 3, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_42(xs):
    #Predicts Class 0
    function_dict = np.array([[1860468100.0, 1.0, 2.0, 7.0], [820.5, 3.0, 4.0, 0.0], [753.5, 5.0, 6.0, 0.0], [789.5, 7.0, 8.0, 0.0], [172283520.0, 9.0, 10.0, 9.0], [3435824640.0, 11.0, 12.0, 7.0], [29.0, 13.0, 14.0, 4.0], [7.88749981, 15.0, 16.0, 8.0], [0.120991051, 0.0, 0.0, 0.0], [841.5, 17.0, 18.0, 0.0], [-0.184961721, 0.0, 0.0, 0.0], [177.5, 19.0, 20.0, 0.0], [22.5, 21.0, 22.0, 4.0], [0.0235754158, 0.0, 0.0, 0.0], [0.114909843, 0.0, 0.0, 0.0], [-0.035141442, 0.0, 0.0, 0.0], [0.0204437263, 0.0, 0.0, 0.0], [-0.136591256, 0.0, 0.0, 0.0], [0.0731272176, 0.0, 0.0, 0.0], [0.0153190577, 0.0, 0.0, 0.0], [-0.17520541, 0.0, 0.0, 0.0], [0.125720233, 0.0, 0.0, 0.0], [-0.0530574024, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 10, 19, 20, 21, 22, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_43(xs):
    #Predicts Class 1
    function_dict = np.array([[1860468100.0, 1.0, 2.0, 7.0], [820.5, 3.0, 4.0, 0.0], [753.5, 5.0, 6.0, 0.0], [789.5, 7.0, 8.0, 0.0], [172283520.0, 9.0, 10.0, 9.0], [3435824640.0, 11.0, 12.0, 7.0], [29.0, 13.0, 14.0, 4.0], [7.88749981, 15.0, 16.0, 8.0], [-0.120991051, 0.0, 0.0, 0.0], [841.5, 17.0, 18.0, 0.0], [0.184961721, 0.0, 0.0, 0.0], [177.5, 19.0, 20.0, 0.0], [22.5, 21.0, 22.0, 4.0], [-0.0235754289, 0.0, 0.0, 0.0], [-0.114909843, 0.0, 0.0, 0.0], [0.0351414494, 0.0, 0.0, 0.0], [-0.0204437245, 0.0, 0.0, 0.0], [0.136591256, 0.0, 0.0, 0.0], [-0.0731272101, 0.0, 0.0, 0.0], [-0.0153190568, 0.0, 0.0, 0.0], [0.175205395, 0.0, 0.0, 0.0], [-0.125720233, 0.0, 0.0, 0.0], [0.0530574247, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 10, 19, 20, 21, 22, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_44(xs):
    #Predicts Class 0
    function_dict = np.array([[2664.0, 1.0, 2.0, 7.0], [7.2270999, 3.0, 4.0, 8.0], [7575.5, 5.0, 6.0, 7.0], [0.0493827648, 0.0, 0.0, 0.0], [18.7104492, 7.0, 8.0, 8.0], [9.70624924, 9.0, 10.0, 8.0], [7.69165039, 11.0, 12.0, 8.0], [432.0, 13.0, 14.0, 0.0], [-0.00603349414, 0.0, 0.0, 0.0], [4134.5, 15.0, 16.0, 7.0], [19.0228996, 17.0, 18.0, 8.0], [2668476420.0, 19.0, 20.0, 2.0], [36.8770981, 21.0, 22.0, 8.0], [-0.14691624, 0.0, 0.0, 0.0], [-0.0473162048, 0.0, 0.0, 0.0], [-0.0837258026, 0.0, 0.0, 0.0], [0.104759395, 0.0, 0.0, 0.0], [0.145092666, 0.0, 0.0, 0.0], [0.00342057133, 0.0, 0.0, 0.0], [0.117725089, 0.0, 0.0, 0.0], [-0.012479594, 0.0, 0.0, 0.0], [-0.0179714523, 0.0, 0.0, 0.0], [0.033397343, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 8, 15, 16, 17, 18, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 4, 7, 2, 5, 9, 10, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_45(xs):
    #Predicts Class 1
    function_dict = np.array([[2664.0, 1.0, 2.0, 7.0], [7.2270999, 3.0, 4.0, 8.0], [7575.5, 5.0, 6.0, 7.0], [-0.0493827499, 0.0, 0.0, 0.0], [18.7104492, 7.0, 8.0, 8.0], [9.70624924, 9.0, 10.0, 8.0], [7.69165039, 11.0, 12.0, 8.0], [432.0, 13.0, 14.0, 0.0], [0.00603348855, 0.0, 0.0, 0.0], [4134.5, 15.0, 16.0, 7.0], [19.0228996, 17.0, 18.0, 8.0], [2668476420.0, 19.0, 20.0, 2.0], [36.8770981, 21.0, 22.0, 8.0], [0.14691624, 0.0, 0.0, 0.0], [0.0473161861, 0.0, 0.0, 0.0], [0.0837257728, 0.0, 0.0, 0.0], [-0.104759403, 0.0, 0.0, 0.0], [-0.145092666, 0.0, 0.0, 0.0], [-0.00342057319, 0.0, 0.0, 0.0], [-0.117725089, 0.0, 0.0, 0.0], [0.012479593, 0.0, 0.0, 0.0], [0.0179714523, 0.0, 0.0, 0.0], [-0.0333973505, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 8, 15, 16, 17, 18, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 4, 7, 2, 5, 9, 10, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_46(xs):
    #Predicts Class 0
    function_dict = np.array([[56.1978989, 1.0, 2.0, 8.0], [37.7520981, 3.0, 4.0, 8.0], [2635741440.0, 5.0, 6.0, 2.0], [1.5, 7.0, 8.0, 1.0], [699.0, 9.0, 10.0, 0.0], [21.5, 11.0, 12.0, 4.0], [1342256510.0, 13.0, 14.0, 3.0], [53.0, 15.0, 16.0, 4.0], [387371.5, 17.0, 18.0, 7.0], [1342256510.0, 19.0, 20.0, 3.0], [-0.0812739283, 0.0, 0.0, 0.0], [0.0630117729, 0.0, 0.0, 0.0], [50.5, 21.0, 22.0, 4.0], [-0.081293568, 0.0, 0.0, 0.0], [28458.0, 23.0, 24.0, 7.0], [-0.132812858, 0.0, 0.0, 0.0], [0.0785280019, 0.0, 0.0, 0.0], [0.027841676, 0.0, 0.0, 0.0], [-0.0366444401, 0.0, 0.0, 0.0], [0.0286970921, 0.0, 0.0, 0.0], [0.181566522, 0.0, 0.0, 0.0], [-0.155592382, 0.0, 0.0, 0.0], [-0.0245955624, 0.0, 0.0, 0.0], [-0.0442169495, 0.0, 0.0, 0.0], [0.123547286, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 10, 11, 21, 22, 13, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 12, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_47(xs):
    #Predicts Class 1
    function_dict = np.array([[56.1978989, 1.0, 2.0, 8.0], [37.7520981, 3.0, 4.0, 8.0], [2635741440.0, 5.0, 6.0, 2.0], [1.5, 7.0, 8.0, 1.0], [699.0, 9.0, 10.0, 0.0], [21.5, 11.0, 12.0, 4.0], [1342256510.0, 13.0, 14.0, 3.0], [53.0, 15.0, 16.0, 4.0], [387371.5, 17.0, 18.0, 7.0], [1342256510.0, 19.0, 20.0, 3.0], [0.0812739208, 0.0, 0.0, 0.0], [-0.0630118102, 0.0, 0.0, 0.0], [50.5, 21.0, 22.0, 4.0], [0.081293568, 0.0, 0.0, 0.0], [28458.0, 23.0, 24.0, 7.0], [0.132812873, 0.0, 0.0, 0.0], [-0.0785280019, 0.0, 0.0, 0.0], [-0.0278416779, 0.0, 0.0, 0.0], [0.0366444476, 0.0, 0.0, 0.0], [-0.0286970884, 0.0, 0.0, 0.0], [-0.181566522, 0.0, 0.0, 0.0], [0.155592382, 0.0, 0.0, 0.0], [0.024595568, 0.0, 0.0, 0.0], [0.0442169495, 0.0, 0.0, 0.0], [-0.123547286, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 10, 11, 21, 22, 13, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 12, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_48(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [250650.0, 3.0, 4.0, 7.0], [7.76249981, 5.0, 6.0, 8.0], [9660.0, 7.0, 8.0, 7.0], [366226.0, 9.0, 10.0, 7.0], [19.5, 11.0, 12.0, 4.0], [7.86460018, 13.0, 14.0, 8.0], [0.5, 15.0, 16.0, 5.0], [-0.13844341, 0.0, 0.0, 0.0], [3222181890.0, 17.0, 18.0, 2.0], [2305741310.0, 19.0, 20.0, 2.0], [0.121560611, 0.0, 0.0, 0.0], [27.0, 21.0, 22.0, 4.0], [2331814910.0, 23.0, 24.0, 2.0], [7.91040039, 25.0, 26.0, 8.0], [-0.0681802705, 0.0, 0.0, 0.0], [0.0738398135, 0.0, 0.0, 0.0], [0.00656485558, 0.0, 0.0, 0.0], [0.149280235, 0.0, 0.0, 0.0], [-0.110808894, 0.0, 0.0, 0.0], [0.0112891328, 0.0, 0.0, 0.0], [-0.0604930185, 0.0, 0.0, 0.0], [0.121095151, 0.0, 0.0, 0.0], [-0.163536176, 0.0, 0.0, 0.0], [0.00224879058, 0.0, 0.0, 0.0], [0.12395177, 0.0, 0.0, 0.0], [0.00382297533, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 11, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_49(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [250650.0, 3.0, 4.0, 7.0], [7.76249981, 5.0, 6.0, 8.0], [9660.0, 7.0, 8.0, 7.0], [366226.0, 9.0, 10.0, 7.0], [19.5, 11.0, 12.0, 4.0], [7.86460018, 13.0, 14.0, 8.0], [0.5, 15.0, 16.0, 5.0], [0.13844341, 0.0, 0.0, 0.0], [3222181890.0, 17.0, 18.0, 2.0], [2305741310.0, 19.0, 20.0, 2.0], [-0.121560611, 0.0, 0.0, 0.0], [27.0, 21.0, 22.0, 4.0], [2331814910.0, 23.0, 24.0, 2.0], [7.91040039, 25.0, 26.0, 8.0], [0.0681802407, 0.0, 0.0, 0.0], [-0.0738398209, 0.0, 0.0, 0.0], [-0.00656486163, 0.0, 0.0, 0.0], [-0.14928022, 0.0, 0.0, 0.0], [0.110808901, 0.0, 0.0, 0.0], [-0.0112891225, 0.0, 0.0, 0.0], [0.0604930185, 0.0, 0.0, 0.0], [-0.121095151, 0.0, 0.0, 0.0], [0.163536161, 0.0, 0.0, 0.0], [-0.00224880083, 0.0, 0.0, 0.0], [-0.123951778, 0.0, 0.0, 0.0], [-0.0038229744, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 11, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_50(xs):
    #Predicts Class 0
    function_dict = np.array([[2664.0, 1.0, 2.0, 7.0], [0.209999993, 3.0, 4.0, 4.0], [61.5, 5.0, 6.0, 4.0], [14.8520498, 7.0, 8.0, 8.0], [24.5, 9.0, 10.0, 4.0], [7575.5, 11.0, 12.0, 7.0], [1.5, 13.0, 14.0, 1.0], [0.0533952191, 0.0, 0.0, 0.0], [-0.0245665982, 0.0, 0.0, 0.0], [-0.129352331, 0.0, 0.0, 0.0], [2112.0, 15.0, 16.0, 7.0], [1637187840.0, 17.0, 18.0, 2.0], [20718.0, 19.0, 20.0, 7.0], [0.0263492092, 0.0, 0.0, 0.0], [-0.197549924, 0.0, 0.0, 0.0], [-0.0114089092, 0.0, 0.0, 0.0], [0.0036126757, 0.0, 0.0, 0.0], [0.0121808685, 0.0, 0.0, 0.0], [0.128046855, 0.0, 0.0, 0.0], [-0.06505505, 0.0, 0.0, 0.0], [0.0108500682, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 15, 16, 17, 18, 19, 20, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 10, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_51(xs):
    #Predicts Class 1
    function_dict = np.array([[2664.0, 1.0, 2.0, 7.0], [0.209999993, 3.0, 4.0, 4.0], [61.5, 5.0, 6.0, 4.0], [14.8520498, 7.0, 8.0, 8.0], [24.5, 9.0, 10.0, 4.0], [7575.5, 11.0, 12.0, 7.0], [1.5, 13.0, 14.0, 1.0], [-0.0533952229, 0.0, 0.0, 0.0], [0.0245665982, 0.0, 0.0, 0.0], [0.129352331, 0.0, 0.0, 0.0], [2112.0, 15.0, 16.0, 7.0], [1637187840.0, 17.0, 18.0, 2.0], [20718.0, 19.0, 20.0, 7.0], [-0.0263492037, 0.0, 0.0, 0.0], [0.197549954, 0.0, 0.0, 0.0], [0.0114089157, 0.0, 0.0, 0.0], [-0.00361266476, 0.0, 0.0, 0.0], [-0.0121808862, 0.0, 0.0, 0.0], [-0.128046855, 0.0, 0.0, 0.0], [0.06505505, 0.0, 0.0, 0.0], [-0.0108500682, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 15, 16, 17, 18, 19, 20, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 10, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_52(xs):
    #Predicts Class 0
    function_dict = np.array([[35.5, 1.0, 2.0, 4.0], [33.5, 3.0, 4.0, 4.0], [184.0, 5.0, 6.0, 0.0], [302.5, 7.0, 8.0, 0.0], [26.1437492, 9.0, 10.0, 8.0], [59.0, 11.0, 12.0, 0.0], [578.5, 13.0, 14.0, 0.0], [1051402750.0, 15.0, 16.0, 2.0], [533.5, 17.0, 18.0, 0.0], [322.0, 19.0, 20.0, 0.0], [-0.194407985, 0.0, 0.0, 0.0], [0.0323136263, 0.0, 0.0, 0.0], [0.133341327, 0.0, 0.0, 0.0], [11.75, 21.0, 22.0, 8.0], [2805942530.0, 23.0, 24.0, 9.0], [0.0644311309, 0.0, 0.0, 0.0], [-0.0640825704, 0.0, 0.0, 0.0], [0.0691772774, 0.0, 0.0, 0.0], [-0.0139714181, 0.0, 0.0, 0.0], [-0.0974352732, 0.0, 0.0, 0.0], [0.0681125522, 0.0, 0.0, 0.0], [-0.123209208, 0.0, 0.0, 0.0], [0.0133733833, 0.0, 0.0, 0.0], [0.137718201, 0.0, 0.0, 0.0], [-0.115170829, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 10, 11, 12, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_53(xs):
    #Predicts Class 1
    function_dict = np.array([[35.5, 1.0, 2.0, 4.0], [33.5, 3.0, 4.0, 4.0], [184.0, 5.0, 6.0, 0.0], [302.5, 7.0, 8.0, 0.0], [26.1437492, 9.0, 10.0, 8.0], [59.0, 11.0, 12.0, 0.0], [578.5, 13.0, 14.0, 0.0], [1051402750.0, 15.0, 16.0, 2.0], [533.5, 17.0, 18.0, 0.0], [322.0, 19.0, 20.0, 0.0], [0.19440797, 0.0, 0.0, 0.0], [-0.0323136188, 0.0, 0.0, 0.0], [-0.133341327, 0.0, 0.0, 0.0], [11.75, 21.0, 22.0, 8.0], [2805942530.0, 23.0, 24.0, 9.0], [-0.0644311234, 0.0, 0.0, 0.0], [0.0640825704, 0.0, 0.0, 0.0], [-0.0691772774, 0.0, 0.0, 0.0], [0.0139714116, 0.0, 0.0, 0.0], [0.0974352807, 0.0, 0.0, 0.0], [-0.0681125447, 0.0, 0.0, 0.0], [0.123209216, 0.0, 0.0, 0.0], [-0.0133733982, 0.0, 0.0, 0.0], [-0.137718201, 0.0, 0.0, 0.0], [0.115170836, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 10, 11, 12, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_54(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [45.25, 5.0, 6.0, 4.0], [0.5, 7.0, 8.0, 6.0], [20.6625004, 9.0, 10.0, 8.0], [44.5, 11.0, 12.0, 4.0], [558.5, 13.0, 14.0, 0.0], [603.5, 15.0, 16.0, 0.0], [-0.00223208684, 0.0, 0.0, 0.0], [207465920.0, 17.0, 18.0, 2.0], [5.5, 19.0, 20.0, 4.0], [2621.5, 21.0, 22.0, 7.0], [-0.124295883, 0.0, 0.0, 0.0], [49.5, 23.0, 24.0, 4.0], [50.0, 25.0, 26.0, 4.0], [-0.1265551, 0.0, 0.0, 0.0], [-0.000927297282, 0.0, 0.0, 0.0], [0.0980014727, 0.0, 0.0, 0.0], [-0.0321952216, 0.0, 0.0, 0.0], [-0.0240395144, 0.0, 0.0, 0.0], [0.124238327, 0.0, 0.0, 0.0], [-0.118577212, 0.0, 0.0, 0.0], [0.0137179792, 0.0, 0.0, 0.0], [0.00398031995, 0.0, 0.0, 0.0], [0.1493368, 0.0, 0.0, 0.0], [0.0463541523, 0.0, 0.0, 0.0], [-0.0959756598, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 21, 22, 12, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_55(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [45.25, 5.0, 6.0, 4.0], [0.5, 7.0, 8.0, 6.0], [20.6625004, 9.0, 10.0, 8.0], [44.5, 11.0, 12.0, 4.0], [558.5, 13.0, 14.0, 0.0], [603.5, 15.0, 16.0, 0.0], [0.00223208708, 0.0, 0.0, 0.0], [207465920.0, 17.0, 18.0, 2.0], [5.5, 19.0, 20.0, 4.0], [2621.5, 21.0, 22.0, 7.0], [0.124295905, 0.0, 0.0, 0.0], [49.5, 23.0, 24.0, 4.0], [50.0, 25.0, 26.0, 4.0], [0.126555115, 0.0, 0.0, 0.0], [0.000927315268, 0.0, 0.0, 0.0], [-0.0980014727, 0.0, 0.0, 0.0], [0.0321952179, 0.0, 0.0, 0.0], [0.0240395088, 0.0, 0.0, 0.0], [-0.124238312, 0.0, 0.0, 0.0], [0.118577212, 0.0, 0.0, 0.0], [-0.013717982, 0.0, 0.0, 0.0], [-0.00398032041, 0.0, 0.0, 0.0], [-0.149336815, 0.0, 0.0, 0.0], [-0.046354156, 0.0, 0.0, 0.0], [0.0959756598, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 21, 22, 12, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_56(xs):
    #Predicts Class 0
    function_dict = np.array([[52.5499992, 1.0, 2.0, 8.0], [37.7520981, 3.0, 4.0, 8.0], [1.5, 5.0, 6.0, 5.0], [1.5, 7.0, 8.0, 1.0], [38.0, 9.0, 10.0, 4.0], [2280339710.0, 11.0, 12.0, 2.0], [0.0714704767, 0.0, 0.0, 0.0], [111739.0, 13.0, 14.0, 7.0], [59.5, 15.0, 16.0, 4.0], [0.143233687, 0.0, 0.0, 0.0], [0.0201237239, 0.0, 0.0, 0.0], [790394816.0, 17.0, 18.0, 10.0], [36960.0, 19.0, 20.0, 7.0], [-0.109859422, 0.0, 0.0, 0.0], [0.0325220861, 0.0, 0.0, 0.0], [0.0100772548, 0.0, 0.0, 0.0], [-0.127563864, 0.0, 0.0, 0.0], [-0.0382297784, 0.0, 0.0, 0.0], [-0.127461493, 0.0, 0.0, 0.0], [-0.0698947087, 0.0, 0.0, 0.0], [0.0507040769, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 17, 18, 19, 20, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_57(xs):
    #Predicts Class 1
    function_dict = np.array([[52.5499992, 1.0, 2.0, 8.0], [37.7520981, 3.0, 4.0, 8.0], [1.5, 5.0, 6.0, 5.0], [1.5, 7.0, 8.0, 1.0], [38.0, 9.0, 10.0, 4.0], [2280339710.0, 11.0, 12.0, 2.0], [-0.0714704618, 0.0, 0.0, 0.0], [111739.0, 13.0, 14.0, 7.0], [59.5, 15.0, 16.0, 4.0], [-0.143233702, 0.0, 0.0, 0.0], [-0.0201237258, 0.0, 0.0, 0.0], [790394816.0, 17.0, 18.0, 10.0], [36960.0, 19.0, 20.0, 7.0], [0.109859429, 0.0, 0.0, 0.0], [-0.0325220861, 0.0, 0.0, 0.0], [-0.0100772567, 0.0, 0.0, 0.0], [0.127563894, 0.0, 0.0, 0.0], [0.0382297896, 0.0, 0.0, 0.0], [0.127461478, 0.0, 0.0, 0.0], [0.0698947012, 0.0, 0.0, 0.0], [-0.0507040992, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 17, 18, 19, 20, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_58(xs):
    #Predicts Class 0
    function_dict = np.array([[45.25, 1.0, 2.0, 4.0], [1.5, 3.0, 4.0, 1.0], [47.5, 5.0, 6.0, 4.0], [30.75, 7.0, 8.0, 8.0], [341.0, 9.0, 10.0, 0.0], [0.122165956, 0.0, 0.0, 0.0], [1342256510.0, 11.0, 12.0, 3.0], [111739.0, 13.0, 14.0, 7.0], [52.5499992, 15.0, 16.0, 8.0], [267.5, 17.0, 18.0, 0.0], [206900272.0, 19.0, 20.0, 2.0], [-0.06947653, 0.0, 0.0, 0.0], [558.5, 21.0, 22.0, 0.0], [-0.168406874, 0.0, 0.0, 0.0], [-0.0224535763, 0.0, 0.0, 0.0], [0.104027174, 0.0, 0.0, 0.0], [-0.0465760715, 0.0, 0.0, 0.0], [-0.0027197192, 0.0, 0.0, 0.0], [-0.1267481, 0.0, 0.0, 0.0], [-0.110166766, 0.0, 0.0, 0.0], [0.0370004438, 0.0, 0.0, 0.0], [0.0973357633, 0.0, 0.0, 0.0], [-0.0434841029, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 5, 11, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 6, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_59(xs):
    #Predicts Class 1
    function_dict = np.array([[45.25, 1.0, 2.0, 4.0], [1.5, 3.0, 4.0, 1.0], [47.5, 5.0, 6.0, 4.0], [30.75, 7.0, 8.0, 8.0], [341.0, 9.0, 10.0, 0.0], [-0.122165948, 0.0, 0.0, 0.0], [1342256510.0, 11.0, 12.0, 3.0], [111739.0, 13.0, 14.0, 7.0], [52.5499992, 15.0, 16.0, 8.0], [267.5, 17.0, 18.0, 0.0], [206900272.0, 19.0, 20.0, 2.0], [0.0694765151, 0.0, 0.0, 0.0], [558.5, 21.0, 22.0, 0.0], [0.168406874, 0.0, 0.0, 0.0], [0.0224535856, 0.0, 0.0, 0.0], [-0.104027145, 0.0, 0.0, 0.0], [0.0465760715, 0.0, 0.0, 0.0], [0.00271972199, 0.0, 0.0, 0.0], [0.126748085, 0.0, 0.0, 0.0], [0.110166758, 0.0, 0.0, 0.0], [-0.0370004475, 0.0, 0.0, 0.0], [-0.0973357558, 0.0, 0.0, 0.0], [0.0434841029, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 5, 11, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 6, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_60(xs):
    #Predicts Class 0
    function_dict = np.array([[39.5, 1.0, 2.0, 4.0], [2656.0, 3.0, 4.0, 7.0], [20718.0, 5.0, 6.0, 7.0], [7.8729496, 7.0, 8.0, 8.0], [13131.5, 9.0, 10.0, 7.0], [2645232640.0, 11.0, 12.0, 2.0], [3310912510.0, 13.0, 14.0, 7.0], [0.00873832591, 0.0, 0.0, 0.0], [-0.121872172, 0.0, 0.0, 0.0], [2004903680.0, 15.0, 16.0, 2.0], [31.5, 17.0, 18.0, 4.0], [-0.0961778536, 0.0, 0.0, 0.0], [3538064380.0, 19.0, 20.0, 2.0], [2874498050.0, 21.0, 22.0, 2.0], [-0.0653961748, 0.0, 0.0, 0.0], [-0.0322437137, 0.0, 0.0, 0.0], [0.133254811, 0.0, 0.0, 0.0], [0.000548068783, 0.0, 0.0, 0.0], [-0.064942643, 0.0, 0.0, 0.0], [0.112740412, 0.0, 0.0, 0.0], [-0.018323129, 0.0, 0.0, 0.0], [0.0938425437, 0.0, 0.0, 0.0], [0.00435094815, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 11, 19, 20, 21, 22, 14])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_61(xs):
    #Predicts Class 1
    function_dict = np.array([[39.5, 1.0, 2.0, 4.0], [2656.0, 3.0, 4.0, 7.0], [20718.0, 5.0, 6.0, 7.0], [7.8729496, 7.0, 8.0, 8.0], [13131.5, 9.0, 10.0, 7.0], [2645232640.0, 11.0, 12.0, 2.0], [3310912510.0, 13.0, 14.0, 7.0], [-0.00873832963, 0.0, 0.0, 0.0], [0.121872179, 0.0, 0.0, 0.0], [2004903680.0, 15.0, 16.0, 2.0], [31.5, 17.0, 18.0, 4.0], [0.0961778685, 0.0, 0.0, 0.0], [3538064380.0, 19.0, 20.0, 2.0], [2874498050.0, 21.0, 22.0, 2.0], [0.0653961301, 0.0, 0.0, 0.0], [0.0322436988, 0.0, 0.0, 0.0], [-0.133254826, 0.0, 0.0, 0.0], [-0.000548068318, 0.0, 0.0, 0.0], [0.0649426356, 0.0, 0.0, 0.0], [-0.112740405, 0.0, 0.0, 0.0], [0.0183231309, 0.0, 0.0, 0.0], [-0.0938425213, 0.0, 0.0, 0.0], [-0.00435097609, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 11, 19, 20, 21, 22, 14])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_62(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [10.8249998, 3.0, 4.0, 8.0], [1.5, 5.0, 6.0, 6.0], [348353.0, 7.0, 8.0, 7.0], [14.15625, 9.0, 10.0, 8.0], [479872640.0, 11.0, 12.0, 9.0], [2.0, 13.0, 14.0, 5.0], [8.03960037, 15.0, 16.0, 8.0], [387370.0, 17.0, 18.0, 7.0], [-0.142356589, 0.0, 0.0, 0.0], [366037.5, 19.0, 20.0, 7.0], [11.3708496, 21.0, 22.0, 8.0], [700.5, 23.0, 24.0, 0.0], [-0.109225295, 0.0, 0.0, 0.0], [0.0301713906, 0.0, 0.0, 0.0], [-0.127567574, 0.0, 0.0, 0.0], [0.0848374739, 0.0, 0.0, 0.0], [0.151625067, 0.0, 0.0, 0.0], [8.79906584e-05, 0.0, 0.0, 0.0], [0.0387257822, 0.0, 0.0, 0.0], [-0.0788910538, 0.0, 0.0, 0.0], [-0.00475040451, 0.0, 0.0, 0.0], [0.0619827956, 0.0, 0.0, 0.0], [0.0134576373, 0.0, 0.0, 0.0], [-0.110418238, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 21, 22, 23, 24, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_63(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [10.8249998, 3.0, 4.0, 8.0], [1.5, 5.0, 6.0, 6.0], [348353.0, 7.0, 8.0, 7.0], [14.15625, 9.0, 10.0, 8.0], [479872640.0, 11.0, 12.0, 9.0], [2.0, 13.0, 14.0, 5.0], [8.03960037, 15.0, 16.0, 8.0], [387370.0, 17.0, 18.0, 7.0], [0.142356589, 0.0, 0.0, 0.0], [366037.5, 19.0, 20.0, 7.0], [11.3708496, 21.0, 22.0, 8.0], [700.5, 23.0, 24.0, 0.0], [0.109225303, 0.0, 0.0, 0.0], [-0.0301713906, 0.0, 0.0, 0.0], [0.127567589, 0.0, 0.0, 0.0], [-0.0848374888, 0.0, 0.0, 0.0], [-0.151625052, 0.0, 0.0, 0.0], [-8.80052758e-05, 0.0, 0.0, 0.0], [-0.0387257636, 0.0, 0.0, 0.0], [0.0788910463, 0.0, 0.0, 0.0], [0.00475040171, 0.0, 0.0, 0.0], [-0.0619827881, 0.0, 0.0, 0.0], [-0.013457641, 0.0, 0.0, 0.0], [0.11041823, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 21, 22, 23, 24, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_64(xs):
    #Predicts Class 0
    function_dict = np.array([[7.1333499, 1.0, 2.0, 8.0], [2668476420.0, 3.0, 4.0, 2.0], [2668.5, 5.0, 6.0, 7.0], [0.100666381, 0.0, 0.0, 0.0], [-0.000834777078, 0.0, 0.0, 0.0], [0.209999993, 7.0, 8.0, 4.0], [7575.5, 9.0, 10.0, 7.0], [14.8520498, 11.0, 12.0, 8.0], [24.5, 13.0, 14.0, 4.0], [4134.5, 15.0, 16.0, 7.0], [196179056.0, 17.0, 18.0, 2.0], [0.0607309453, 0.0, 0.0, 0.0], [-0.0310981255, 0.0, 0.0, 0.0], [-0.133484885, 0.0, 0.0, 0.0], [-0.0025235957, 0.0, 0.0, 0.0], [0.0241116267, 0.0, 0.0, 0.0], [0.101079054, 0.0, 0.0, 0.0], [0.0608566366, 0.0, 0.0, 0.0], [-0.00780198164, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 11, 12, 13, 14, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 2, 5, 7, 8, 6, 9, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_65(xs):
    #Predicts Class 1
    function_dict = np.array([[7.1333499, 1.0, 2.0, 8.0], [2668476420.0, 3.0, 4.0, 2.0], [2668.5, 5.0, 6.0, 7.0], [-0.100666389, 0.0, 0.0, 0.0], [0.000834796811, 0.0, 0.0, 0.0], [0.209999993, 7.0, 8.0, 4.0], [7575.5, 9.0, 10.0, 7.0], [14.8520498, 11.0, 12.0, 8.0], [24.5, 13.0, 14.0, 4.0], [4134.5, 15.0, 16.0, 7.0], [196179056.0, 17.0, 18.0, 2.0], [-0.0607309379, 0.0, 0.0, 0.0], [0.0310981218, 0.0, 0.0, 0.0], [0.133484885, 0.0, 0.0, 0.0], [0.00252361177, 0.0, 0.0, 0.0], [-0.0241116397, 0.0, 0.0, 0.0], [-0.101079054, 0.0, 0.0, 0.0], [-0.0608566552, 0.0, 0.0, 0.0], [0.00780198677, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 11, 12, 13, 14, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 2, 5, 7, 8, 6, 9, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_66(xs):
    #Predicts Class 0
    function_dict = np.array([[52.5, 1.0, 2.0, 0.0], [2166785020.0, 3.0, 4.0, 2.0], [99.5, 5.0, 6.0, 0.0], [0.123762928, 0.0, 0.0, 0.0], [246535.5, 7.0, 8.0, 7.0], [8.77499962, 9.0, 10.0, 8.0], [107.0, 11.0, 12.0, 0.0], [0.0257730298, 0.0, 0.0, 0.0], [-0.11018347, 0.0, 0.0, 0.0], [0.0424985215, 0.0, 0.0, 0.0], [73.5, 13.0, 14.0, 0.0], [0.117152996, 0.0, 0.0, 0.0], [20.5, 15.0, 16.0, 4.0], [-0.0176163223, 0.0, 0.0, 0.0], [-0.175392568, 0.0, 0.0, 0.0], [-0.0216799621, 0.0, 0.0, 0.0], [0.0125732003, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 13, 14, 11, 15, 16])
    branch_indices = np.array([0, 1, 4, 2, 5, 10, 6, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_67(xs):
    #Predicts Class 1
    function_dict = np.array([[52.5, 1.0, 2.0, 0.0], [2166785020.0, 3.0, 4.0, 2.0], [99.5, 5.0, 6.0, 0.0], [-0.123762935, 0.0, 0.0, 0.0], [246535.5, 7.0, 8.0, 7.0], [8.77499962, 9.0, 10.0, 8.0], [107.0, 11.0, 12.0, 0.0], [-0.0257730428, 0.0, 0.0, 0.0], [0.11018347, 0.0, 0.0, 0.0], [-0.0424985066, 0.0, 0.0, 0.0], [73.5, 13.0, 14.0, 0.0], [-0.117152996, 0.0, 0.0, 0.0], [20.5, 15.0, 16.0, 4.0], [0.0176163111, 0.0, 0.0, 0.0], [0.175392583, 0.0, 0.0, 0.0], [0.0216799602, 0.0, 0.0, 0.0], [-0.0125731993, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 13, 14, 11, 15, 16])
    branch_indices = np.array([0, 1, 4, 2, 5, 10, 6, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_68(xs):
    #Predicts Class 0
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [372928.0, 3.0, 4.0, 7.0], [9.5, 5.0, 6.0, 4.0], [7.68335009, 7.0, 8.0, 8.0], [0.131234124, 0.0, 0.0, 0.0], [2.5, 9.0, 10.0, 5.0], [196179056.0, 11.0, 12.0, 2.0], [0.0786671415, 0.0, 0.0, 0.0], [7.88749981, 13.0, 14.0, 8.0], [15.5728998, 15.0, 16.0, 8.0], [30.2000008, 17.0, 18.0, 8.0], [0.100128494, 0.0, 0.0, 0.0], [1283933570.0, 19.0, 20.0, 2.0], [-0.0931782573, 0.0, 0.0, 0.0], [0.00467253476, 0.0, 0.0, 0.0], [-0.028328428, 0.0, 0.0, 0.0], [-0.155177966, 0.0, 0.0, 0.0], [0.0928170308, 0.0, 0.0, 0.0], [-0.0720287114, 0.0, 0.0, 0.0], [-0.0488648079, 0.0, 0.0, 0.0], [0.0135684023, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 13, 14, 4, 15, 16, 17, 18, 11, 19, 20])
    branch_indices = np.array([0, 1, 3, 8, 2, 5, 9, 10, 6, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_69(xs):
    #Predicts Class 1
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [372928.0, 3.0, 4.0, 7.0], [9.5, 5.0, 6.0, 4.0], [7.68335009, 7.0, 8.0, 8.0], [-0.131234139, 0.0, 0.0, 0.0], [2.5, 9.0, 10.0, 5.0], [196179056.0, 11.0, 12.0, 2.0], [-0.0786671415, 0.0, 0.0, 0.0], [7.88749981, 13.0, 14.0, 8.0], [15.5728998, 15.0, 16.0, 8.0], [30.2000008, 17.0, 18.0, 8.0], [-0.100128494, 0.0, 0.0, 0.0], [1283933570.0, 19.0, 20.0, 2.0], [0.0931782499, 0.0, 0.0, 0.0], [-0.00467252266, 0.0, 0.0, 0.0], [0.0283284169, 0.0, 0.0, 0.0], [0.155177966, 0.0, 0.0, 0.0], [-0.0928170383, 0.0, 0.0, 0.0], [0.072028704, 0.0, 0.0, 0.0], [0.0488648117, 0.0, 0.0, 0.0], [-0.0135684023, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 13, 14, 4, 15, 16, 17, 18, 11, 19, 20])
    branch_indices = np.array([0, 1, 3, 8, 2, 5, 9, 10, 6, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_70(xs):
    #Predicts Class 0
    function_dict = np.array([[820.5, 1.0, 2.0, 0.0], [789.5, 3.0, 4.0, 0.0], [841.0, 5.0, 6.0, 0.0], [1342256510.0, 7.0, 8.0, 3.0], [0.117511161, 0.0, 0.0, 0.0], [-0.135369107, 0.0, 0.0, 0.0], [24.6896, 9.0, 10.0, 8.0], [250650.0, 11.0, 12.0, 7.0], [1860468100.0, 13.0, 14.0, 7.0], [1342256510.0, 15.0, 16.0, 3.0], [-0.0681751743, 0.0, 0.0, 0.0], [-0.0733265951, 0.0, 0.0, 0.0], [-0.00216619391, 0.0, 0.0, 0.0], [0.0191006418, 0.0, 0.0, 0.0], [-0.0293875076, 0.0, 0.0, 0.0], [-0.0229571946, 0.0, 0.0, 0.0], [0.0979901329, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 15, 16, 10])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 9])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_71(xs):
    #Predicts Class 1
    function_dict = np.array([[820.5, 1.0, 2.0, 0.0], [789.5, 3.0, 4.0, 0.0], [841.0, 5.0, 6.0, 0.0], [1342256510.0, 7.0, 8.0, 3.0], [-0.117511176, 0.0, 0.0, 0.0], [0.135369122, 0.0, 0.0, 0.0], [24.6896, 9.0, 10.0, 8.0], [250650.0, 11.0, 12.0, 7.0], [1860468100.0, 13.0, 14.0, 7.0], [1342256510.0, 15.0, 16.0, 3.0], [0.0681751668, 0.0, 0.0, 0.0], [0.0733265877, 0.0, 0.0, 0.0], [0.0021661995, 0.0, 0.0, 0.0], [-0.0191006418, 0.0, 0.0, 0.0], [0.0293875169, 0.0, 0.0, 0.0], [0.0229571965, 0.0, 0.0, 0.0], [-0.0979901329, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 15, 16, 10])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 9])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_72(xs):
    #Predicts Class 0
    function_dict = np.array([[4035160830.0, 1.0, 2.0, 7.0], [3977575420.0, 3.0, 4.0, 7.0], [13.1750002, 5.0, 6.0, 8.0], [4062966020.0, 7.0, 8.0, 2.0], [-0.117003672, 0.0, 0.0, 0.0], [0.0888300762, 0.0, 0.0, 0.0], [0.00791363139, 0.0, 0.0, 0.0], [3842726400.0, 9.0, 10.0, 2.0], [4218452480.0, 11.0, 12.0, 2.0], [-0.000750233536, 0.0, 0.0, 0.0], [0.0644999593, 0.0, 0.0, 0.0], [-0.105748527, 0.0, 0.0, 0.0], [0.060246069, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 4, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_73(xs):
    #Predicts Class 1
    function_dict = np.array([[4035160830.0, 1.0, 2.0, 7.0], [3977575420.0, 3.0, 4.0, 7.0], [13.1750002, 5.0, 6.0, 8.0], [4062966020.0, 7.0, 8.0, 2.0], [0.117003679, 0.0, 0.0, 0.0], [-0.0888300836, 0.0, 0.0, 0.0], [-0.00791363139, 0.0, 0.0, 0.0], [3842726400.0, 9.0, 10.0, 2.0], [4218452480.0, 11.0, 12.0, 2.0], [0.000750233943, 0.0, 0.0, 0.0], [-0.0644999593, 0.0, 0.0, 0.0], [0.105748512, 0.0, 0.0, 0.0], [-0.0602460615, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 4, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_74(xs):
    #Predicts Class 0
    function_dict = np.array([[3236726780.0, 1.0, 2.0, 2.0], [3082723330.0, 3.0, 4.0, 2.0], [3275228670.0, 5.0, 6.0, 2.0], [65.5, 7.0, 8.0, 0.0], [9.36250019, 9.0, 10.0, 8.0], [-0.211011291, 0.0, 0.0, 0.0], [346416.5, 11.0, 12.0, 7.0], [2037452540.0, 13.0, 14.0, 2.0], [2223089660.0, 15.0, 16.0, 7.0], [0.0137717342, 0.0, 0.0, 0.0], [0.178330764, 0.0, 0.0, 0.0], [233.0, 17.0, 18.0, 0.0], [18.5, 19.0, 20.0, 4.0], [0.111310892, 0.0, 0.0, 0.0], [-0.0552347302, 0.0, 0.0, 0.0], [0.00265356922, 0.0, 0.0, 0.0], [-0.0478870161, 0.0, 0.0, 0.0], [-0.0999498591, 0.0, 0.0, 0.0], [0.00864860136, 0.0, 0.0, 0.0], [-0.0131797455, 0.0, 0.0, 0.0], [0.0821293443, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_75(xs):
    #Predicts Class 1
    function_dict = np.array([[3236726780.0, 1.0, 2.0, 2.0], [3082723330.0, 3.0, 4.0, 2.0], [3275228670.0, 5.0, 6.0, 2.0], [65.5, 7.0, 8.0, 0.0], [9.36250019, 9.0, 10.0, 8.0], [0.211011291, 0.0, 0.0, 0.0], [346416.5, 11.0, 12.0, 7.0], [2037452540.0, 13.0, 14.0, 2.0], [2223089660.0, 15.0, 16.0, 7.0], [-0.0137717482, 0.0, 0.0, 0.0], [-0.178330749, 0.0, 0.0, 0.0], [233.0, 17.0, 18.0, 0.0], [18.5, 19.0, 20.0, 4.0], [-0.111310899, 0.0, 0.0, 0.0], [0.0552347451, 0.0, 0.0, 0.0], [-0.00265356665, 0.0, 0.0, 0.0], [0.0478870086, 0.0, 0.0, 0.0], [0.0999498591, 0.0, 0.0, 0.0], [-0.00864860136, 0.0, 0.0, 0.0], [0.0131797409, 0.0, 0.0, 0.0], [-0.0821293443, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_76(xs):
    #Predicts Class 0
    function_dict = np.array([[2661.5, 1.0, 2.0, 7.0], [14.8520498, 3.0, 4.0, 8.0], [1.5, 5.0, 6.0, 1.0], [1989208580.0, 7.0, 8.0, 2.0], [-0.0931688473, 0.0, 0.0, 0.0], [27.1353989, 9.0, 10.0, 8.0], [59.5, 11.0, 12.0, 4.0], [0.0657268614, 0.0, 0.0, 0.0], [-0.0552299507, 0.0, 0.0, 0.0], [37.0, 13.0, 14.0, 4.0], [627.0, 15.0, 16.0, 0.0], [408.5, 17.0, 18.0, 0.0], [-0.0928795412, 0.0, 0.0, 0.0], [-0.162837327, 0.0, 0.0, 0.0], [-0.0225658622, 0.0, 0.0, 0.0], [0.032260634, 0.0, 0.0, 0.0], [-0.0650752783, 0.0, 0.0, 0.0], [-0.00472669955, 0.0, 0.0, 0.0], [0.0401697792, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 13, 14, 15, 16, 17, 18, 12])
    branch_indices = np.array([0, 1, 3, 2, 5, 9, 10, 6, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_77(xs):
    #Predicts Class 1
    function_dict = np.array([[2661.5, 1.0, 2.0, 7.0], [14.8520498, 3.0, 4.0, 8.0], [1.5, 5.0, 6.0, 1.0], [1989208580.0, 7.0, 8.0, 2.0], [0.0931688473, 0.0, 0.0, 0.0], [27.1353989, 9.0, 10.0, 8.0], [59.5, 11.0, 12.0, 4.0], [-0.0657268614, 0.0, 0.0, 0.0], [0.0552299507, 0.0, 0.0, 0.0], [37.0, 13.0, 14.0, 4.0], [627.0, 15.0, 16.0, 0.0], [408.5, 17.0, 18.0, 0.0], [0.0928795412, 0.0, 0.0, 0.0], [0.162837327, 0.0, 0.0, 0.0], [0.022565851, 0.0, 0.0, 0.0], [-0.0322606452, 0.0, 0.0, 0.0], [0.0650752783, 0.0, 0.0, 0.0], [0.00472670281, 0.0, 0.0, 0.0], [-0.0401697867, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 13, 14, 15, 16, 17, 18, 12])
    branch_indices = np.array([0, 1, 3, 2, 5, 9, 10, 6, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_78(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [48.2021027, 3.0, 4.0, 8.0], [4045140480.0, 5.0, 6.0, 7.0], [3222181890.0, 7.0, 8.0, 2.0], [-0.101508491, 0.0, 0.0, 0.0], [7.76249981, 9.0, 10.0, 8.0], [0.0984081998, 0.0, 0.0, 0.0], [71805064.0, 11.0, 12.0, 7.0], [0.5, 13.0, 14.0, 5.0], [7.23960018, 15.0, 16.0, 8.0], [7.86460018, 17.0, 18.0, 8.0], [-0.0417062901, 0.0, 0.0, 0.0], [0.0376273394, 0.0, 0.0, 0.0], [-0.036994189, 0.0, 0.0, 0.0], [0.158150196, 0.0, 0.0, 0.0], [0.00956216361, 0.0, 0.0, 0.0], [0.110100627, 0.0, 0.0, 0.0], [-0.0796450898, 0.0, 0.0, 0.0], [0.00473497948, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 15, 16, 17, 18, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 9, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_79(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [48.2021027, 3.0, 4.0, 8.0], [4045140480.0, 5.0, 6.0, 7.0], [3222181890.0, 7.0, 8.0, 2.0], [0.101508491, 0.0, 0.0, 0.0], [7.76249981, 9.0, 10.0, 8.0], [-0.0984081924, 0.0, 0.0, 0.0], [71805064.0, 11.0, 12.0, 7.0], [0.5, 13.0, 14.0, 5.0], [7.23960018, 15.0, 16.0, 8.0], [7.86460018, 17.0, 18.0, 8.0], [0.0417062901, 0.0, 0.0, 0.0], [-0.0376273394, 0.0, 0.0, 0.0], [0.0369942002, 0.0, 0.0, 0.0], [-0.158150166, 0.0, 0.0, 0.0], [-0.00956217851, 0.0, 0.0, 0.0], [-0.110100627, 0.0, 0.0, 0.0], [0.0796450824, 0.0, 0.0, 0.0], [-0.00473497808, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 15, 16, 17, 18, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 9, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_80(xs):
    #Predicts Class 0
    function_dict = np.array([[914675200.0, 1.0, 2.0, 2.0], [14.9771004, 3.0, 4.0, 8.0], [982098304.0, 5.0, 6.0, 2.0], [31.0, 7.0, 8.0, 4.0], [391.5, 9.0, 10.0, 0.0], [-0.137266979, 0.0, 0.0, 0.0], [20.5, 11.0, 12.0, 4.0], [7.76249981, 13.0, 14.0, 8.0], [8.35210037, 15.0, 16.0, 8.0], [360137.0, 17.0, 18.0, 7.0], [292438.5, 19.0, 20.0, 7.0], [1746678.5, 21.0, 22.0, 7.0], [479872640.0, 23.0, 24.0, 9.0], [0.00929907802, 0.0, 0.0, 0.0], [0.140421614, 0.0, 0.0, 0.0], [-0.125894338, 0.0, 0.0, 0.0], [0.0745893195, 0.0, 0.0, 0.0], [0.0842380598, 0.0, 0.0, 0.0], [-0.0666040704, 0.0, 0.0, 0.0], [-0.105897382, 0.0, 0.0, 0.0], [0.0178578831, 0.0, 0.0, 0.0], [-0.00292675919, 0.0, 0.0, 0.0], [-0.0907042027, 0.0, 0.0, 0.0], [0.0373935662, 0.0, 0.0, 0.0], [-0.0311907008, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 5, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_81(xs):
    #Predicts Class 1
    function_dict = np.array([[914675200.0, 1.0, 2.0, 2.0], [14.9771004, 3.0, 4.0, 8.0], [982098304.0, 5.0, 6.0, 2.0], [31.0, 7.0, 8.0, 4.0], [391.5, 9.0, 10.0, 0.0], [0.137267023, 0.0, 0.0, 0.0], [20.5, 11.0, 12.0, 4.0], [7.76249981, 13.0, 14.0, 8.0], [8.35210037, 15.0, 16.0, 8.0], [360137.0, 17.0, 18.0, 7.0], [292438.5, 19.0, 20.0, 7.0], [1746678.5, 21.0, 22.0, 7.0], [479872640.0, 23.0, 24.0, 9.0], [-0.00929905009, 0.0, 0.0, 0.0], [-0.140421614, 0.0, 0.0, 0.0], [0.125894353, 0.0, 0.0, 0.0], [-0.0745893195, 0.0, 0.0, 0.0], [-0.0842380449, 0.0, 0.0, 0.0], [0.0666040778, 0.0, 0.0, 0.0], [0.105897374, 0.0, 0.0, 0.0], [-0.0178578757, 0.0, 0.0, 0.0], [0.00292676152, 0.0, 0.0, 0.0], [0.0907041952, 0.0, 0.0, 0.0], [-0.0373935625, 0.0, 0.0, 0.0], [0.0311907027, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 5, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_82(xs):
    #Predicts Class 0
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [372928.0, 3.0, 4.0, 7.0], [9.5, 5.0, 6.0, 4.0], [7.68335009, 7.0, 8.0, 8.0], [0.121038243, 0.0, 0.0, 0.0], [2.5, 9.0, 10.0, 5.0], [9.70624924, 11.0, 12.0, 8.0], [0.0738114789, 0.0, 0.0, 0.0], [7.88749981, 13.0, 14.0, 8.0], [15.5728998, 15.0, 16.0, 8.0], [0.0151982177, 0.0, 0.0, 0.0], [2399681020.0, 17.0, 18.0, 2.0], [1.5, 19.0, 20.0, 1.0], [-0.0835223794, 0.0, 0.0, 0.0], [0.0091903694, 0.0, 0.0, 0.0], [-0.0194289163, 0.0, 0.0, 0.0], [-0.13469784, 0.0, 0.0, 0.0], [0.0127757182, 0.0, 0.0, 0.0], [-0.0717692822, 0.0, 0.0, 0.0], [-0.0199687015, 0.0, 0.0, 0.0], [0.0495019518, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 13, 14, 4, 15, 16, 10, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 8, 2, 5, 9, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_83(xs):
    #Predicts Class 1
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [372928.0, 3.0, 4.0, 7.0], [9.5, 5.0, 6.0, 4.0], [7.68335009, 7.0, 8.0, 8.0], [-0.121038258, 0.0, 0.0, 0.0], [2.5, 9.0, 10.0, 5.0], [9.70624924, 11.0, 12.0, 8.0], [-0.0738115013, 0.0, 0.0, 0.0], [7.88749981, 13.0, 14.0, 8.0], [15.5728998, 15.0, 16.0, 8.0], [-0.0151981972, 0.0, 0.0, 0.0], [2399681020.0, 17.0, 18.0, 2.0], [1.5, 19.0, 20.0, 1.0], [0.0835223794, 0.0, 0.0, 0.0], [-0.00919037405, 0.0, 0.0, 0.0], [0.0194289219, 0.0, 0.0, 0.0], [0.13469784, 0.0, 0.0, 0.0], [-0.0127757145, 0.0, 0.0, 0.0], [0.0717692748, 0.0, 0.0, 0.0], [0.0199687015, 0.0, 0.0, 0.0], [-0.0495019518, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 13, 14, 4, 15, 16, 10, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 8, 2, 5, 9, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_84(xs):
    #Predicts Class 0
    function_dict = np.array([[2661.5, 1.0, 2.0, 7.0], [373.5, 3.0, 4.0, 0.0], [0.375, 5.0, 6.0, 4.0], [-0.0959162638, 0.0, 0.0, 0.0], [622.0, 7.0, 8.0, 0.0], [372928.0, 9.0, 10.0, 7.0], [9.5, 11.0, 12.0, 4.0], [0.0411001518, 0.0, 0.0, 0.0], [-0.047784958, 0.0, 0.0, 0.0], [7953.5, 13.0, 14.0, 7.0], [0.104124792, 0.0, 0.0, 0.0], [2854893570.0, 15.0, 16.0, 2.0], [196179056.0, 17.0, 18.0, 2.0], [0.0987261087, 0.0, 0.0, 0.0], [-0.0197997149, 0.0, 0.0, 0.0], [-0.0971480906, 0.0, 0.0, 0.0], [0.0107988967, 0.0, 0.0, 0.0], [0.0948994085, 0.0, 0.0, 0.0], [-0.00215104595, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 13, 14, 10, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 4, 2, 5, 9, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_85(xs):
    #Predicts Class 1
    function_dict = np.array([[2661.5, 1.0, 2.0, 7.0], [373.5, 3.0, 4.0, 0.0], [0.375, 5.0, 6.0, 4.0], [0.0959162787, 0.0, 0.0, 0.0], [622.0, 7.0, 8.0, 0.0], [372928.0, 9.0, 10.0, 7.0], [9.5, 11.0, 12.0, 4.0], [-0.0411001518, 0.0, 0.0, 0.0], [0.0477849469, 0.0, 0.0, 0.0], [7953.5, 13.0, 14.0, 7.0], [-0.104124777, 0.0, 0.0, 0.0], [2854893570.0, 15.0, 16.0, 2.0], [196179056.0, 17.0, 18.0, 2.0], [-0.0987261161, 0.0, 0.0, 0.0], [0.0197997373, 0.0, 0.0, 0.0], [0.0971480832, 0.0, 0.0, 0.0], [-0.0107988995, 0.0, 0.0, 0.0], [-0.0948994383, 0.0, 0.0, 0.0], [0.00215104525, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 13, 14, 10, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 4, 2, 5, 9, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_86(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [4045140480.0, 5.0, 6.0, 7.0], [315.0, 7.0, 8.0, 0.0], [371.5, 9.0, 10.0, 0.0], [3979916030.0, 11.0, 12.0, 7.0], [0.0870309025, 0.0, 0.0, 0.0], [0.0142903104, 0.0, 0.0, 0.0], [1600627580.0, 13.0, 14.0, 9.0], [141.5, 15.0, 16.0, 0.0], [13.9354496, 17.0, 18.0, 8.0], [3236726780.0, 19.0, 20.0, 2.0], [-0.0995128155, 0.0, 0.0, 0.0], [-0.121839486, 0.0, 0.0, 0.0], [-0.0097704269, 0.0, 0.0, 0.0], [0.0195770562, 0.0, 0.0, 0.0], [-0.0636776835, 0.0, 0.0, 0.0], [-0.00240050489, 0.0, 0.0, 0.0], [0.0810982361, 0.0, 0.0, 0.0], [0.022773426, 0.0, 0.0, 0.0], [-0.0193348955, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 13, 14, 15, 16, 17, 18, 19, 20, 12, 6])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_87(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [4045140480.0, 5.0, 6.0, 7.0], [315.0, 7.0, 8.0, 0.0], [371.5, 9.0, 10.0, 0.0], [3979916030.0, 11.0, 12.0, 7.0], [-0.0870309025, 0.0, 0.0, 0.0], [-0.0142902806, 0.0, 0.0, 0.0], [1600627580.0, 13.0, 14.0, 9.0], [141.5, 15.0, 16.0, 0.0], [13.9354496, 17.0, 18.0, 8.0], [3236726780.0, 19.0, 20.0, 2.0], [0.0995128304, 0.0, 0.0, 0.0], [0.121839494, 0.0, 0.0, 0.0], [0.00977044553, 0.0, 0.0, 0.0], [-0.0195770543, 0.0, 0.0, 0.0], [0.063677676, 0.0, 0.0, 0.0], [0.00240049907, 0.0, 0.0, 0.0], [-0.0810982361, 0.0, 0.0, 0.0], [-0.0227734242, 0.0, 0.0, 0.0], [0.0193349011, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 13, 14, 15, 16, 17, 18, 19, 20, 12, 6])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_88(xs):
    #Predicts Class 0
    function_dict = np.array([[2668.5, 1.0, 2.0, 7.0], [550.0, 3.0, 4.0, 0.0], [7575.5, 5.0, 6.0, 7.0], [1832209540.0, 7.0, 8.0, 2.0], [-0.0858880505, 0.0, 0.0, 0.0], [9.70624924, 9.0, 10.0, 8.0], [7.76249981, 11.0, 12.0, 8.0], [-0.0662220344, 0.0, 0.0, 0.0], [0.0480675027, 0.0, 0.0, 0.0], [0.0166047644, 0.0, 0.0, 0.0], [0.0910669416, 0.0, 0.0, 0.0], [168.5, 13.0, 14.0, 0.0], [350411.5, 15.0, 16.0, 7.0], [-0.0656659305, 0.0, 0.0, 0.0], [0.0775011629, 0.0, 0.0, 0.0], [0.00961981341, 0.0, 0.0, 0.0], [-0.0332436077, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 2, 5, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_89(xs):
    #Predicts Class 1
    function_dict = np.array([[2668.5, 1.0, 2.0, 7.0], [550.0, 3.0, 4.0, 0.0], [7575.5, 5.0, 6.0, 7.0], [1832209540.0, 7.0, 8.0, 2.0], [0.0858880356, 0.0, 0.0, 0.0], [9.70624924, 9.0, 10.0, 8.0], [7.76249981, 11.0, 12.0, 8.0], [0.0662220269, 0.0, 0.0, 0.0], [-0.0480674915, 0.0, 0.0, 0.0], [-0.0166047607, 0.0, 0.0, 0.0], [-0.0910669491, 0.0, 0.0, 0.0], [168.5, 13.0, 14.0, 0.0], [350411.5, 15.0, 16.0, 7.0], [0.065665938, 0.0, 0.0, 0.0], [-0.0775011703, 0.0, 0.0, 0.0], [-0.00961981341, 0.0, 0.0, 0.0], [0.0332436077, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 2, 5, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_90(xs):
    #Predicts Class 0
    function_dict = np.array([[1323098110.0, 1.0, 2.0, 2.0], [56.0, 3.0, 4.0, 0.0], [347742.5, 5.0, 6.0, 7.0], [0.10249044, 0.0, 0.0, 0.0], [349972.0, 7.0, 8.0, 7.0], [8.03960037, 9.0, 10.0, 8.0], [7.91040039, 11.0, 12.0, 8.0], [243256.0, 13.0, 14.0, 7.0], [254174624.0, 15.0, 16.0, 7.0], [764.0, 17.0, 18.0, 0.0], [26.5, 19.0, 20.0, 4.0], [3101264.5, 21.0, 22.0, 7.0], [18.5, 23.0, 24.0, 4.0], [-0.040950615, 0.0, 0.0, 0.0], [0.085889101, 0.0, 0.0, 0.0], [-0.179335266, 0.0, 0.0, 0.0], [0.00915155094, 0.0, 0.0, 0.0], [-0.107709497, 0.0, 0.0, 0.0], [0.0618317425, 0.0, 0.0, 0.0], [0.047187347, 0.0, 0.0, 0.0], [-0.0345172286, 0.0, 0.0, 0.0], [0.128320307, 0.0, 0.0, 0.0], [-0.00512348488, 0.0, 0.0, 0.0], [-0.0535145216, 0.0, 0.0, 0.0], [0.0388813391, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 10, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_91(xs):
    #Predicts Class 1
    function_dict = np.array([[1323098110.0, 1.0, 2.0, 2.0], [56.0, 3.0, 4.0, 0.0], [347742.5, 5.0, 6.0, 7.0], [-0.102490447, 0.0, 0.0, 0.0], [349972.0, 7.0, 8.0, 7.0], [8.03960037, 9.0, 10.0, 8.0], [7.91040039, 11.0, 12.0, 8.0], [243256.0, 13.0, 14.0, 7.0], [254174624.0, 15.0, 16.0, 7.0], [764.0, 17.0, 18.0, 0.0], [26.5, 19.0, 20.0, 4.0], [3101264.5, 21.0, 22.0, 7.0], [18.5, 23.0, 24.0, 4.0], [0.040950641, 0.0, 0.0, 0.0], [-0.085889101, 0.0, 0.0, 0.0], [0.179335296, 0.0, 0.0, 0.0], [-0.00915155932, 0.0, 0.0, 0.0], [0.10770952, 0.0, 0.0, 0.0], [-0.0618317463, 0.0, 0.0, 0.0], [-0.0471873432, 0.0, 0.0, 0.0], [0.0345172361, 0.0, 0.0, 0.0], [-0.128320321, 0.0, 0.0, 0.0], [0.00512347324, 0.0, 0.0, 0.0], [0.0535145104, 0.0, 0.0, 0.0], [-0.0388813354, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 10, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_92(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [39.5, 5.0, 6.0, 4.0], [315.0, 7.0, 8.0, 0.0], [4013449220.0, 9.0, 10.0, 2.0], [302.5, 11.0, 12.0, 0.0], [2172281860.0, 13.0, 14.0, 2.0], [0.0132667394, 0.0, 0.0, 0.0], [1600627580.0, 15.0, 16.0, 9.0], [235.0, 17.0, 18.0, 0.0], [0.0610395037, 0.0, 0.0, 0.0], [258.0, 19.0, 20.0, 0.0], [389.0, 21.0, 22.0, 0.0], [20696.0, 23.0, 24.0, 7.0], [3989815810.0, 25.0, 26.0, 2.0], [-0.113593288, 0.0, 0.0, 0.0], [-0.00895894784, 0.0, 0.0, 0.0], [-0.0598072447, 0.0, 0.0, 0.0], [0.00714637991, 0.0, 0.0, 0.0], [0.00977407768, 0.0, 0.0, 0.0], [-0.119592518, 0.0, 0.0, 0.0], [0.132835761, 0.0, 0.0, 0.0], [-0.00733980536, 0.0, 0.0, 0.0], [-0.0623806454, 0.0, 0.0, 0.0], [0.0464102365, 0.0, 0.0, 0.0], [0.121063799, 0.0, 0.0, 0.0], [-0.00442011142, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 10, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_93(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [39.5, 5.0, 6.0, 4.0], [315.0, 7.0, 8.0, 0.0], [4013449220.0, 9.0, 10.0, 2.0], [302.5, 11.0, 12.0, 0.0], [2172281860.0, 13.0, 14.0, 2.0], [-0.0132667506, 0.0, 0.0, 0.0], [1600627580.0, 15.0, 16.0, 9.0], [235.0, 17.0, 18.0, 0.0], [-0.0610394999, 0.0, 0.0, 0.0], [258.0, 19.0, 20.0, 0.0], [389.0, 21.0, 22.0, 0.0], [20696.0, 23.0, 24.0, 7.0], [3989815810.0, 25.0, 26.0, 2.0], [0.113593295, 0.0, 0.0, 0.0], [0.00895894971, 0.0, 0.0, 0.0], [0.0598072335, 0.0, 0.0, 0.0], [-0.00714637898, 0.0, 0.0, 0.0], [-0.00977408141, 0.0, 0.0, 0.0], [0.119592525, 0.0, 0.0, 0.0], [-0.132835746, 0.0, 0.0, 0.0], [0.00733979419, 0.0, 0.0, 0.0], [0.0623806342, 0.0, 0.0, 0.0], [-0.0464102402, 0.0, 0.0, 0.0], [-0.121063799, 0.0, 0.0, 0.0], [0.00442009047, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 10, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_94(xs):
    #Predicts Class 0
    function_dict = np.array([[820.5, 1.0, 2.0, 0.0], [764.0, 3.0, 4.0, 0.0], [25.5, 5.0, 6.0, 4.0], [744.5, 7.0, 8.0, 0.0], [7.76249981, 9.0, 10.0, 8.0], [0.0319156162, 0.0, 0.0, 0.0], [-0.103071168, 0.0, 0.0, 0.0], [7.69165039, 11.0, 12.0, 8.0], [-0.107910126, 0.0, 0.0, 0.0], [-0.0408742949, 0.0, 0.0, 0.0], [17.0, 13.0, 14.0, 4.0], [0.0615497716, 0.0, 0.0, 0.0], [-0.00381890801, 0.0, 0.0, 0.0], [0.0272802971, 0.0, 0.0, 0.0], [0.127660513, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 9, 13, 14, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_95(xs):
    #Predicts Class 1
    function_dict = np.array([[820.5, 1.0, 2.0, 0.0], [764.0, 3.0, 4.0, 0.0], [25.5, 5.0, 6.0, 4.0], [744.5, 7.0, 8.0, 0.0], [7.76249981, 9.0, 10.0, 8.0], [-0.0319155827, 0.0, 0.0, 0.0], [0.10307119, 0.0, 0.0, 0.0], [7.69165039, 11.0, 12.0, 8.0], [0.107910119, 0.0, 0.0, 0.0], [0.0408742912, 0.0, 0.0, 0.0], [17.0, 13.0, 14.0, 4.0], [-0.0615497753, 0.0, 0.0, 0.0], [0.0038189122, 0.0, 0.0, 0.0], [-0.0272803009, 0.0, 0.0, 0.0], [-0.127660513, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 9, 13, 14, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_96(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [48.2021027, 3.0, 4.0, 8.0], [2250458880.0, 5.0, 6.0, 10.0], [21.0375004, 7.0, 8.0, 8.0], [-0.085603185, 0.0, 0.0, 0.0], [1.5, 9.0, 10.0, 5.0], [1.5, 11.0, 12.0, 5.0], [10.8249998, 13.0, 14.0, 8.0], [249689.0, 15.0, 16.0, 7.0], [100.0, 17.0, 18.0, 0.0], [0.0798571631, 0.0, 0.0, 0.0], [0.0950863957, 0.0, 0.0, 0.0], [0.00113438442, 0.0, 0.0, 0.0], [0.00247640349, 0.0, 0.0, 0.0], [-0.0616026297, 0.0, 0.0, 0.0], [-0.0515367761, 0.0, 0.0, 0.0], [0.0711589158, 0.0, 0.0, 0.0], [-0.0536660738, 0.0, 0.0, 0.0], [0.0063041281, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 17, 18, 10, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 9, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_97(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [48.2021027, 3.0, 4.0, 8.0], [2250458880.0, 5.0, 6.0, 10.0], [21.0375004, 7.0, 8.0, 8.0], [0.0856031775, 0.0, 0.0, 0.0], [1.5, 9.0, 10.0, 5.0], [1.5, 11.0, 12.0, 5.0], [10.8249998, 13.0, 14.0, 8.0], [249689.0, 15.0, 16.0, 7.0], [100.0, 17.0, 18.0, 0.0], [-0.0798571631, 0.0, 0.0, 0.0], [-0.0950863883, 0.0, 0.0, 0.0], [-0.00113439455, 0.0, 0.0, 0.0], [-0.00247639371, 0.0, 0.0, 0.0], [0.0616026297, 0.0, 0.0, 0.0], [0.0515367463, 0.0, 0.0, 0.0], [-0.0711589009, 0.0, 0.0, 0.0], [0.0536660738, 0.0, 0.0, 0.0], [-0.00630413322, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 17, 18, 10, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 9, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_98(xs):
    #Predicts Class 0
    function_dict = np.array([[36.5, 1.0, 2.0, 4.0], [2140000510.0, 3.0, 4.0, 9.0], [179.5, 5.0, 6.0, 0.0], [2656.0, 7.0, 8.0, 7.0], [70.3228989, 9.0, 10.0, 8.0], [0.0989447311, 0.0, 0.0, 0.0], [462.0, 11.0, 12.0, 0.0], [2614067710.0, 13.0, 14.0, 2.0], [32.75, 15.0, 16.0, 8.0], [-0.0955505148, 0.0, 0.0, 0.0], [0.00396633195, 0.0, 0.0, 0.0], [49.5, 17.0, 18.0, 4.0], [61.5, 19.0, 20.0, 4.0], [-0.0210430492, 0.0, 0.0, 0.0], [-0.0852894932, 0.0, 0.0, 0.0], [-0.0028173856, 0.0, 0.0, 0.0], [0.0678558871, 0.0, 0.0, 0.0], [-0.069092229, 0.0, 0.0, 0.0], [0.0453938507, 0.0, 0.0, 0.0], [0.0788749307, 0.0, 0.0, 0.0], [-0.0771218315, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_99(xs):
    #Predicts Class 1
    function_dict = np.array([[36.5, 1.0, 2.0, 4.0], [2140000510.0, 3.0, 4.0, 9.0], [179.5, 5.0, 6.0, 0.0], [2656.0, 7.0, 8.0, 7.0], [70.3228989, 9.0, 10.0, 8.0], [-0.0989447236, 0.0, 0.0, 0.0], [462.0, 11.0, 12.0, 0.0], [2614067710.0, 13.0, 14.0, 2.0], [32.75, 15.0, 16.0, 8.0], [0.0955505222, 0.0, 0.0, 0.0], [-0.0039662933, 0.0, 0.0, 0.0], [49.5, 17.0, 18.0, 4.0], [61.5, 19.0, 20.0, 4.0], [0.0210430827, 0.0, 0.0, 0.0], [0.0852894932, 0.0, 0.0, 0.0], [0.00281738187, 0.0, 0.0, 0.0], [-0.0678558871, 0.0, 0.0, 0.0], [0.0690922365, 0.0, 0.0, 0.0], [-0.0453938693, 0.0, 0.0, 0.0], [-0.0788749158, 0.0, 0.0, 0.0], [0.0771218538, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_100(xs):
    #Predicts Class 0
    function_dict = np.array([[2256621820.0, 1.0, 2.0, 7.0], [3235088380.0, 3.0, 4.0, 2.0], [3033046020.0, 5.0, 6.0, 2.0], [2717574140.0, 7.0, 8.0, 2.0], [3307532030.0, 9.0, 10.0, 2.0], [2864528900.0, 11.0, 12.0, 7.0], [14.7749996, 13.0, 14.0, 8.0], [36.5, 15.0, 16.0, 4.0], [225.5, 17.0, 18.0, 0.0], [-0.141526923, 0.0, 0.0, 0.0], [3776773630.0, 19.0, 20.0, 2.0], [-0.115466364, 0.0, 0.0, 0.0], [2651748350.0, 21.0, 22.0, 2.0], [0.0645214915, 0.0, 0.0, 0.0], [0.0163563676, 0.0, 0.0, 0.0], [-0.00879694335, 0.0, 0.0, 0.0], [0.0448123105, 0.0, 0.0, 0.0], [-0.0507836081, 0.0, 0.0, 0.0], [0.100562483, 0.0, 0.0, 0.0], [0.0707592741, 0.0, 0.0, 0.0], [-0.0326683708, 0.0, 0.0, 0.0], [0.0251726601, 0.0, 0.0, 0.0], [-0.115161806, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 11, 21, 22, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_101(xs):
    #Predicts Class 1
    function_dict = np.array([[2256621820.0, 1.0, 2.0, 7.0], [3235088380.0, 3.0, 4.0, 2.0], [3033046020.0, 5.0, 6.0, 2.0], [2717574140.0, 7.0, 8.0, 2.0], [3307532030.0, 9.0, 10.0, 2.0], [2864528900.0, 11.0, 12.0, 7.0], [14.7749996, 13.0, 14.0, 8.0], [36.5, 15.0, 16.0, 4.0], [225.5, 17.0, 18.0, 0.0], [0.141526923, 0.0, 0.0, 0.0], [3776773630.0, 19.0, 20.0, 2.0], [0.115466356, 0.0, 0.0, 0.0], [2651748350.0, 21.0, 22.0, 2.0], [-0.0645214766, 0.0, 0.0, 0.0], [-0.0163563713, 0.0, 0.0, 0.0], [0.00879693497, 0.0, 0.0, 0.0], [-0.044812303, 0.0, 0.0, 0.0], [0.0507836379, 0.0, 0.0, 0.0], [-0.100562483, 0.0, 0.0, 0.0], [-0.0707592517, 0.0, 0.0, 0.0], [0.0326683819, 0.0, 0.0, 0.0], [-0.0251726769, 0.0, 0.0, 0.0], [0.115161799, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 11, 21, 22, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_102(xs):
    #Predicts Class 0
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [370370.5, 3.0, 4.0, 7.0], [2256621820.0, 5.0, 6.0, 7.0], [366038.5, 7.0, 8.0, 7.0], [23.7999992, 9.0, 10.0, 8.0], [820.5, 11.0, 12.0, 0.0], [3545664510.0, 13.0, 14.0, 2.0], [7.88749981, 15.0, 16.0, 8.0], [-0.0638174564, 0.0, 0.0, 0.0], [0.109410785, 0.0, 0.0, 0.0], [0.0183480904, 0.0, 0.0, 0.0], [13.5, 17.0, 18.0, 4.0], [24.5, 19.0, 20.0, 4.0], [2864528900.0, 21.0, 22.0, 7.0], [0.0484760702, 0.0, 0.0, 0.0], [-0.0468583815, 0.0, 0.0, 0.0], [0.0490198173, 0.0, 0.0, 0.0], [-0.0455258414, 0.0, 0.0, 0.0], [0.0139921606, 0.0, 0.0, 0.0], [0.0119048841, 0.0, 0.0, 0.0], [-0.0867210105, 0.0, 0.0, 0.0], [-0.134993732, 0.0, 0.0, 0.0], [-0.0250393413, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 9, 10, 17, 18, 19, 20, 21, 22, 14])
    branch_indices = np.array([0, 1, 3, 7, 4, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_103(xs):
    #Predicts Class 1
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [370370.5, 3.0, 4.0, 7.0], [2256621820.0, 5.0, 6.0, 7.0], [366038.5, 7.0, 8.0, 7.0], [23.7999992, 9.0, 10.0, 8.0], [820.5, 11.0, 12.0, 0.0], [3545664510.0, 13.0, 14.0, 2.0], [7.88749981, 15.0, 16.0, 8.0], [0.0638174564, 0.0, 0.0, 0.0], [-0.109410785, 0.0, 0.0, 0.0], [-0.0183481108, 0.0, 0.0, 0.0], [13.5, 17.0, 18.0, 4.0], [24.5, 19.0, 20.0, 4.0], [2864528900.0, 21.0, 22.0, 7.0], [-0.0484760739, 0.0, 0.0, 0.0], [0.0468583964, 0.0, 0.0, 0.0], [-0.0490198247, 0.0, 0.0, 0.0], [0.0455258377, 0.0, 0.0, 0.0], [-0.0139921578, 0.0, 0.0, 0.0], [-0.0119048813, 0.0, 0.0, 0.0], [0.0867210105, 0.0, 0.0, 0.0], [0.134993717, 0.0, 0.0, 0.0], [0.0250393376, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 9, 10, 17, 18, 19, 20, 21, 22, 14])
    branch_indices = np.array([0, 1, 3, 7, 4, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_104(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [2250458880.0, 5.0, 6.0, 10.0], [315.0, 7.0, 8.0, 0.0], [7.88749981, 9.0, 10.0, 8.0], [1.5, 11.0, 12.0, 5.0], [290.5, 13.0, 14.0, 0.0], [0.00379937538, 0.0, 0.0, 0.0], [549540736.0, 15.0, 16.0, 9.0], [348938.0, 17.0, 18.0, 7.0], [357212.5, 19.0, 20.0, 7.0], [20718.0, 21.0, 22.0, 7.0], [0.0727439299, 0.0, 0.0, 0.0], [0.0866111144, 0.0, 0.0, 0.0], [0.0101594906, 0.0, 0.0, 0.0], [-0.101595119, 0.0, 0.0, 0.0], [-0.0247457139, 0.0, 0.0, 0.0], [-0.0865428373, 0.0, 0.0, 0.0], [0.0322317742, 0.0, 0.0, 0.0], [0.0508783981, 0.0, 0.0, 0.0], [-0.0349210277, 0.0, 0.0, 0.0], [-0.0323921628, 0.0, 0.0, 0.0], [0.00777740963, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_105(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [2250458880.0, 5.0, 6.0, 10.0], [315.0, 7.0, 8.0, 0.0], [7.88749981, 9.0, 10.0, 8.0], [1.5, 11.0, 12.0, 5.0], [290.5, 13.0, 14.0, 0.0], [-0.00379935652, 0.0, 0.0, 0.0], [549540736.0, 15.0, 16.0, 9.0], [348938.0, 17.0, 18.0, 7.0], [357212.5, 19.0, 20.0, 7.0], [20718.0, 21.0, 22.0, 7.0], [-0.0727439225, 0.0, 0.0, 0.0], [-0.0866111144, 0.0, 0.0, 0.0], [-0.0101595027, 0.0, 0.0, 0.0], [0.101595111, 0.0, 0.0, 0.0], [0.0247457437, 0.0, 0.0, 0.0], [0.0865428299, 0.0, 0.0, 0.0], [-0.0322317742, 0.0, 0.0, 0.0], [-0.0508784018, 0.0, 0.0, 0.0], [0.0349210352, 0.0, 0.0, 0.0], [0.0323921666, 0.0, 0.0, 0.0], [-0.00777741196, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 21, 22, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_106(xs):
    #Predicts Class 0
    function_dict = np.array([[56.1978989, 1.0, 2.0, 8.0], [39.5, 3.0, 4.0, 8.0], [0.5, 5.0, 6.0, 6.0], [0.5, 7.0, 8.0, 5.0], [2318315780.0, 9.0, 10.0, 9.0], [247108656.0, 11.0, 12.0, 7.0], [113634.5, 13.0, 14.0, 7.0], [658073216.0, 15.0, 16.0, 2.0], [25.5, 17.0, 18.0, 4.0], [0.114434555, 0.0, 0.0, 0.0], [-0.00175077363, 0.0, 0.0, 0.0], [-0.0948065966, 0.0, 0.0, 0.0], [-0.00311169145, 0.0, 0.0, 0.0], [0.0689009428, 0.0, 0.0, 0.0], [-0.0367713086, 0.0, 0.0, 0.0], [0.0521570258, 0.0, 0.0, 0.0], [-0.0194607843, 0.0, 0.0, 0.0], [-0.0164863467, 0.0, 0.0, 0.0], [0.102513611, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_107(xs):
    #Predicts Class 1
    function_dict = np.array([[56.1978989, 1.0, 2.0, 8.0], [39.5, 3.0, 4.0, 8.0], [0.5, 5.0, 6.0, 6.0], [0.5, 7.0, 8.0, 5.0], [2318315780.0, 9.0, 10.0, 9.0], [247108656.0, 11.0, 12.0, 7.0], [113634.5, 13.0, 14.0, 7.0], [658073216.0, 15.0, 16.0, 2.0], [25.5, 17.0, 18.0, 4.0], [-0.11443454, 0.0, 0.0, 0.0], [0.00175079121, 0.0, 0.0, 0.0], [0.0948066041, 0.0, 0.0, 0.0], [0.00311169517, 0.0, 0.0, 0.0], [-0.0689009354, 0.0, 0.0, 0.0], [0.0367713198, 0.0, 0.0, 0.0], [-0.0521570295, 0.0, 0.0, 0.0], [0.0194607824, 0.0, 0.0, 0.0], [0.0164863542, 0.0, 0.0, 0.0], [-0.102513626, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_108(xs):
    #Predicts Class 0
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [118.5, 3.0, 4.0, 0.0], [387371.5, 5.0, 6.0, 7.0], [790394816.0, 7.0, 8.0, 10.0], [15.3729, 9.0, 10.0, 8.0], [347742.5, 11.0, 12.0, 7.0], [19.5, 13.0, 14.0, 4.0], [-0.0724201724, 0.0, 0.0, 0.0], [0.0109313447, 0.0, 0.0, 0.0], [330933.0, 15.0, 16.0, 7.0], [23.3500004, 17.0, 18.0, 8.0], [347075.5, 19.0, 20.0, 7.0], [980025344.0, 21.0, 22.0, 2.0], [31.5, 23.0, 24.0, 8.0], [2256621820.0, 25.0, 26.0, 7.0], [0.0252641048, 0.0, 0.0, 0.0], [0.109327361, 0.0, 0.0, 0.0], [-0.0623647124, 0.0, 0.0, 0.0], [0.0354929641, 0.0, 0.0, 0.0], [0.00559556996, 0.0, 0.0, 0.0], [-0.079627566, 0.0, 0.0, 0.0], [-0.0326558612, 0.0, 0.0, 0.0], [0.087341249, 0.0, 0.0, 0.0], [-0.155427143, 0.0, 0.0, 0.0], [0.0382021554, 0.0, 0.0, 0.0], [0.0450637266, 0.0, 0.0, 0.0], [-0.0248485133, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_109(xs):
    #Predicts Class 1
    function_dict = np.array([[0.209999993, 1.0, 2.0, 4.0], [118.5, 3.0, 4.0, 0.0], [387371.5, 5.0, 6.0, 7.0], [790394816.0, 7.0, 8.0, 10.0], [15.3729, 9.0, 10.0, 8.0], [347742.5, 11.0, 12.0, 7.0], [19.5, 13.0, 14.0, 4.0], [0.0724201798, 0.0, 0.0, 0.0], [-0.01093134, 0.0, 0.0, 0.0], [330933.0, 15.0, 16.0, 7.0], [23.3500004, 17.0, 18.0, 8.0], [347075.5, 19.0, 20.0, 7.0], [980025344.0, 21.0, 22.0, 2.0], [31.5, 23.0, 24.0, 8.0], [2256621820.0, 25.0, 26.0, 7.0], [-0.0252640732, 0.0, 0.0, 0.0], [-0.109327361, 0.0, 0.0, 0.0], [0.0623646975, 0.0, 0.0, 0.0], [-0.0354929455, 0.0, 0.0, 0.0], [-0.00559556996, 0.0, 0.0, 0.0], [0.079627566, 0.0, 0.0, 0.0], [0.0326558575, 0.0, 0.0, 0.0], [-0.087341249, 0.0, 0.0, 0.0], [0.155427158, 0.0, 0.0, 0.0], [-0.0382021628, 0.0, 0.0, 0.0], [-0.0450637266, 0.0, 0.0, 0.0], [0.0248485319, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_110(xs):
    #Predicts Class 0
    function_dict = np.array([[1.5, 1.0, 2.0, 1.0], [354.5, 3.0, 4.0, 0.0], [59.5, 5.0, 6.0, 4.0], [2731107070.0, 7.0, 8.0, 9.0], [2068766210.0, 9.0, 10.0, 2.0], [341.0, 11.0, 12.0, 0.0], [-0.0823112428, 0.0, 0.0, 0.0], [238.5, 13.0, 14.0, 0.0], [0.0821277127, 0.0, 0.0, 0.0], [37.0, 15.0, 16.0, 4.0], [565.0, 17.0, 18.0, 0.0], [255.0, 19.0, 20.0, 0.0], [206900272.0, 21.0, 22.0, 2.0], [-0.0591230057, 0.0, 0.0, 0.0], [0.038706895, 0.0, 0.0, 0.0], [-0.121535599, 0.0, 0.0, 0.0], [-0.035596557, 0.0, 0.0, 0.0], [0.0710195303, 0.0, 0.0, 0.0], [-0.0343973935, 0.0, 0.0, 0.0], [0.00737397, 0.0, 0.0, 0.0], [-0.0578999296, 0.0, 0.0, 0.0], [-0.0821726993, 0.0, 0.0, 0.0], [0.0400790386, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 15, 16, 17, 18, 19, 20, 21, 22, 6])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_111(xs):
    #Predicts Class 1
    function_dict = np.array([[1.5, 1.0, 2.0, 1.0], [354.5, 3.0, 4.0, 0.0], [59.5, 5.0, 6.0, 4.0], [2731107070.0, 7.0, 8.0, 9.0], [2068766210.0, 9.0, 10.0, 2.0], [341.0, 11.0, 12.0, 0.0], [0.0823112503, 0.0, 0.0, 0.0], [238.5, 13.0, 14.0, 0.0], [-0.0821276978, 0.0, 0.0, 0.0], [37.0, 15.0, 16.0, 4.0], [565.0, 17.0, 18.0, 0.0], [255.0, 19.0, 20.0, 0.0], [206900272.0, 21.0, 22.0, 2.0], [0.0591229983, 0.0, 0.0, 0.0], [-0.0387069173, 0.0, 0.0, 0.0], [0.121535599, 0.0, 0.0, 0.0], [0.0355965719, 0.0, 0.0, 0.0], [-0.0710195303, 0.0, 0.0, 0.0], [0.0343973711, 0.0, 0.0, 0.0], [-0.00737395603, 0.0, 0.0, 0.0], [0.0578999296, 0.0, 0.0, 0.0], [0.082172662, 0.0, 0.0, 0.0], [-0.0400790349, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 15, 16, 17, 18, 19, 20, 21, 22, 6])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_112(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [31.3312492, 3.0, 4.0, 8.0], [549.5, 5.0, 6.0, 0.0], [366226.0, 7.0, 8.0, 7.0], [1724186.0, 9.0, 10.0, 7.0], [462.0, 11.0, 12.0, 0.0], [578.5, 13.0, 14.0, 0.0], [0.5, 15.0, 16.0, 5.0], [294153440.0, 17.0, 18.0, 7.0], [-0.0892219022, 0.0, 0.0, 0.0], [-0.0200974923, 0.0, 0.0, 0.0], [389.0, 19.0, 20.0, 0.0], [0.0971544608, 0.0, 0.0, 0.0], [-0.140387461, 0.0, 0.0, 0.0], [26.1437492, 21.0, 22.0, 8.0], [-0.0263236929, 0.0, 0.0, 0.0], [0.0947115049, 0.0, 0.0, 0.0], [-0.0882944465, 0.0, 0.0, 0.0], [0.00636198418, 0.0, 0.0, 0.0], [0.020319283, 0.0, 0.0, 0.0], [-0.0486328714, 0.0, 0.0, 0.0], [0.0510926358, 0.0, 0.0, 0.0], [-0.0383777544, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 19, 20, 12, 13, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 11, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_113(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [31.3312492, 3.0, 4.0, 8.0], [549.5, 5.0, 6.0, 0.0], [366226.0, 7.0, 8.0, 7.0], [1724186.0, 9.0, 10.0, 7.0], [462.0, 11.0, 12.0, 0.0], [578.5, 13.0, 14.0, 0.0], [0.5, 15.0, 16.0, 5.0], [294153440.0, 17.0, 18.0, 7.0], [0.0892219022, 0.0, 0.0, 0.0], [0.020097509, 0.0, 0.0, 0.0], [389.0, 19.0, 20.0, 0.0], [-0.0971544459, 0.0, 0.0, 0.0], [0.140387446, 0.0, 0.0, 0.0], [26.1437492, 21.0, 22.0, 8.0], [0.026323704, 0.0, 0.0, 0.0], [-0.0947114751, 0.0, 0.0, 0.0], [0.0882944688, 0.0, 0.0, 0.0], [-0.00636200747, 0.0, 0.0, 0.0], [-0.0203192718, 0.0, 0.0, 0.0], [0.0486328602, 0.0, 0.0, 0.0], [-0.0510926284, 0.0, 0.0, 0.0], [0.038377773, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 19, 20, 12, 13, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 11, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_114(xs):
    #Predicts Class 0
    function_dict = np.array([[2668.5, 1.0, 2.0, 7.0], [0.5, 3.0, 4.0, 6.0], [7575.5, 5.0, 6.0, 7.0], [550.0, 7.0, 8.0, 0.0], [-0.0871468782, 0.0, 0.0, 0.0], [9.70624924, 9.0, 10.0, 8.0], [7.69165039, 11.0, 12.0, 8.0], [0.0381855145, 0.0, 0.0, 0.0], [-0.0480134971, 0.0, 0.0, 0.0], [0.0107233832, 0.0, 0.0, 0.0], [0.072635904, 0.0, 0.0, 0.0], [2668476420.0, 13.0, 14.0, 2.0], [27.1353989, 15.0, 16.0, 8.0], [0.0733148083, 0.0, 0.0, 0.0], [-0.0189581048, 0.0, 0.0, 0.0], [-0.0139500033, 0.0, 0.0, 0.0], [0.0170058589, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 2, 5, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_115(xs):
    #Predicts Class 1
    function_dict = np.array([[2668.5, 1.0, 2.0, 7.0], [0.5, 3.0, 4.0, 6.0], [7575.5, 5.0, 6.0, 7.0], [550.0, 7.0, 8.0, 0.0], [0.0871468782, 0.0, 0.0, 0.0], [9.70624924, 9.0, 10.0, 8.0], [7.69165039, 11.0, 12.0, 8.0], [-0.0381854996, 0.0, 0.0, 0.0], [0.0480135046, 0.0, 0.0, 0.0], [-0.0107234027, 0.0, 0.0, 0.0], [-0.0726359263, 0.0, 0.0, 0.0], [2668476420.0, 13.0, 14.0, 2.0], [27.1353989, 15.0, 16.0, 8.0], [-0.0733148009, 0.0, 0.0, 0.0], [0.0189581085, 0.0, 0.0, 0.0], [0.0139499977, 0.0, 0.0, 0.0], [-0.0170058571, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 2, 5, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_116(xs):
    #Predicts Class 0
    function_dict = np.array([[840381056.0, 1.0, 2.0, 9.0], [0.5, 3.0, 4.0, 5.0], [51.5, 5.0, 6.0, 4.0], [3.0, 7.0, 8.0, 4.0], [14.5, 9.0, 10.0, 4.0], [1041301120.0, 11.0, 12.0, 2.0], [0.0311781913, 0.0, 0.0, 0.0], [330933.0, 13.0, 14.0, 7.0], [349254.0, 15.0, 16.0, 7.0], [3316199420.0, 17.0, 18.0, 2.0], [15.9750004, 19.0, 20.0, 8.0], [0.0476969667, 0.0, 0.0, 0.0], [37.0, 21.0, 22.0, 8.0], [-0.0384696834, 0.0, 0.0, 0.0], [0.106681406, 0.0, 0.0, 0.0], [0.00438528415, 0.0, 0.0, 0.0], [-0.0409016795, 0.0, 0.0, 0.0], [-0.0581636764, 0.0, 0.0, 0.0], [0.0382128991, 0.0, 0.0, 0.0], [0.0152460672, 0.0, 0.0, 0.0], [0.0927014649, 0.0, 0.0, 0.0], [-0.104002818, 0.0, 0.0, 0.0], [-0.0242636185, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 11, 21, 22, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_117(xs):
    #Predicts Class 1
    function_dict = np.array([[840381056.0, 1.0, 2.0, 9.0], [0.5, 3.0, 4.0, 5.0], [51.5, 5.0, 6.0, 4.0], [3.0, 7.0, 8.0, 4.0], [14.5, 9.0, 10.0, 4.0], [1041301120.0, 11.0, 12.0, 2.0], [-0.0311781969, 0.0, 0.0, 0.0], [330933.0, 13.0, 14.0, 7.0], [349254.0, 15.0, 16.0, 7.0], [3316199420.0, 17.0, 18.0, 2.0], [15.9750004, 19.0, 20.0, 8.0], [-0.0476969704, 0.0, 0.0, 0.0], [37.0, 21.0, 22.0, 8.0], [0.0384696946, 0.0, 0.0, 0.0], [-0.106681399, 0.0, 0.0, 0.0], [-0.00438529346, 0.0, 0.0, 0.0], [0.0409016795, 0.0, 0.0, 0.0], [0.0581636839, 0.0, 0.0, 0.0], [-0.038212873, 0.0, 0.0, 0.0], [-0.0152460616, 0.0, 0.0, 0.0], [-0.0927014798, 0.0, 0.0, 0.0], [0.104002818, 0.0, 0.0, 0.0], [0.0242636353, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 11, 21, 22, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_118(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [250650.0, 3.0, 4.0, 7.0], [479872640.0, 5.0, 6.0, 9.0], [9660.0, 7.0, 8.0, 7.0], [366226.0, 9.0, 10.0, 7.0], [830.5, 11.0, 12.0, 0.0], [37.0, 13.0, 14.0, 8.0], [12.84795, 15.0, 16.0, 8.0], [-0.104074411, 0.0, 0.0, 0.0], [3222181890.0, 17.0, 18.0, 2.0], [2305741310.0, 19.0, 20.0, 2.0], [11.3708496, 21.0, 22.0, 8.0], [0.0873899683, 0.0, 0.0, 0.0], [533.0, 23.0, 24.0, 0.0], [0.5, 25.0, 26.0, 5.0], [-0.0367379412, 0.0, 0.0, 0.0], [0.0273149386, 0.0, 0.0, 0.0], [0.00350064342, 0.0, 0.0, 0.0], [0.075708665, 0.0, 0.0, 0.0], [-0.0791720226, 0.0, 0.0, 0.0], [-0.000505729986, 0.0, 0.0, 0.0], [-0.00815791357, 0.0, 0.0, 0.0], [0.0343436897, 0.0, 0.0, 0.0], [-0.0137247322, 0.0, 0.0, 0.0], [-0.0971782207, 0.0, 0.0, 0.0], [0.062892504, 0.0, 0.0, 0.0], [-0.0255195182, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 21, 22, 12, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_119(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [250650.0, 3.0, 4.0, 7.0], [479872640.0, 5.0, 6.0, 9.0], [9660.0, 7.0, 8.0, 7.0], [366226.0, 9.0, 10.0, 7.0], [830.5, 11.0, 12.0, 0.0], [37.0, 13.0, 14.0, 8.0], [12.84795, 15.0, 16.0, 8.0], [0.104074426, 0.0, 0.0, 0.0], [3222181890.0, 17.0, 18.0, 2.0], [2305741310.0, 19.0, 20.0, 2.0], [11.3708496, 21.0, 22.0, 8.0], [-0.0873899758, 0.0, 0.0, 0.0], [533.0, 23.0, 24.0, 0.0], [0.5, 25.0, 26.0, 5.0], [0.0367379189, 0.0, 0.0, 0.0], [-0.0273149312, 0.0, 0.0, 0.0], [-0.003500639, 0.0, 0.0, 0.0], [-0.075708665, 0.0, 0.0, 0.0], [0.0791720077, 0.0, 0.0, 0.0], [0.000505718461, 0.0, 0.0, 0.0], [0.00815792289, 0.0, 0.0, 0.0], [-0.0343436897, 0.0, 0.0, 0.0], [0.0137247462, 0.0, 0.0, 0.0], [0.0971782058, 0.0, 0.0, 0.0], [-0.0628924966, 0.0, 0.0, 0.0], [0.0255195219, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 21, 22, 12, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_120(xs):
    #Predicts Class 0
    function_dict = np.array([[56.1978989, 1.0, 2.0, 8.0], [39.5, 3.0, 4.0, 8.0], [2635741440.0, 5.0, 6.0, 2.0], [4062966020.0, 7.0, 8.0, 2.0], [654.5, 9.0, 10.0, 0.0], [790394816.0, 11.0, 12.0, 10.0], [3703931390.0, 13.0, 14.0, 2.0], [3842726400.0, 15.0, 16.0, 2.0], [4218452480.0, 17.0, 18.0, 2.0], [0.0965053737, 0.0, 0.0, 0.0], [0.00107386243, 0.0, 0.0, 0.0], [0.00286712684, 0.0, 0.0, 0.0], [-0.09553729, 0.0, 0.0, 0.0], [0.0528946482, 0.0, 0.0, 0.0], [-0.0364394374, 0.0, 0.0, 0.0], [-0.00408881763, 0.0, 0.0, 0.0], [0.0790862665, 0.0, 0.0, 0.0], [-0.086026147, 0.0, 0.0, 0.0], [0.0325708576, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_121(xs):
    #Predicts Class 1
    function_dict = np.array([[56.1978989, 1.0, 2.0, 8.0], [39.5, 3.0, 4.0, 8.0], [2635741440.0, 5.0, 6.0, 2.0], [4062966020.0, 7.0, 8.0, 2.0], [654.5, 9.0, 10.0, 0.0], [790394816.0, 11.0, 12.0, 10.0], [3703931390.0, 13.0, 14.0, 2.0], [3842726400.0, 15.0, 16.0, 2.0], [4218452480.0, 17.0, 18.0, 2.0], [-0.0965053588, 0.0, 0.0, 0.0], [-0.00107387116, 0.0, 0.0, 0.0], [-0.00286710612, 0.0, 0.0, 0.0], [0.0955372825, 0.0, 0.0, 0.0], [-0.0528946631, 0.0, 0.0, 0.0], [0.0364394486, 0.0, 0.0, 0.0], [0.00408881623, 0.0, 0.0, 0.0], [-0.0790862665, 0.0, 0.0, 0.0], [0.086026147, 0.0, 0.0, 0.0], [-0.0325708687, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_122(xs):
    #Predicts Class 0
    function_dict = np.array([[3236726780.0, 1.0, 2.0, 2.0], [3082723330.0, 3.0, 4.0, 2.0], [3275228670.0, 5.0, 6.0, 2.0], [314014.0, 7.0, 8.0, 7.0], [0.0862870738, 0.0, 0.0, 0.0], [-0.123107933, 0.0, 0.0, 0.0], [3776773630.0, 9.0, 10.0, 2.0], [30.75, 11.0, 12.0, 8.0], [350030.0, 13.0, 14.0, 7.0], [21.5, 15.0, 16.0, 4.0], [3842726400.0, 17.0, 18.0, 2.0], [-0.035272561, 0.0, 0.0, 0.0], [0.0253371634, 0.0, 0.0, 0.0], [0.0654908866, 0.0, 0.0, 0.0], [-0.0104655232, 0.0, 0.0, 0.0], [-0.0195840467, 0.0, 0.0, 0.0], [0.0895785317, 0.0, 0.0, 0.0], [-0.124205828, 0.0, 0.0, 0.0], [0.00575642521, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 9, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_123(xs):
    #Predicts Class 1
    function_dict = np.array([[3236726780.0, 1.0, 2.0, 2.0], [3082723330.0, 3.0, 4.0, 2.0], [3275228670.0, 5.0, 6.0, 2.0], [314014.0, 7.0, 8.0, 7.0], [-0.0862870887, 0.0, 0.0, 0.0], [0.123107933, 0.0, 0.0, 0.0], [3776773630.0, 9.0, 10.0, 2.0], [30.75, 11.0, 12.0, 8.0], [350030.0, 13.0, 14.0, 7.0], [21.5, 15.0, 16.0, 4.0], [3842726400.0, 17.0, 18.0, 2.0], [0.0352725536, 0.0, 0.0, 0.0], [-0.0253371336, 0.0, 0.0, 0.0], [-0.065490894, 0.0, 0.0, 0.0], [0.0104655223, 0.0, 0.0, 0.0], [0.0195840597, 0.0, 0.0, 0.0], [-0.0895785242, 0.0, 0.0, 0.0], [0.124205828, 0.0, 0.0, 0.0], [-0.00575640798, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 9, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_124(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [10.8249998, 3.0, 4.0, 8.0], [7.76249981, 5.0, 6.0, 8.0], [348353.0, 7.0, 8.0, 7.0], [14.15625, 9.0, 10.0, 8.0], [3854757890.0, 11.0, 12.0, 7.0], [3233670910.0, 13.0, 14.0, 2.0], [8.03960037, 15.0, 16.0, 8.0], [2812186880.0, 17.0, 18.0, 2.0], [-0.108221397, 0.0, 0.0, 0.0], [2.5, 19.0, 20.0, 1.0], [2666.0, 21.0, 22.0, 7.0], [-0.0088331718, 0.0, 0.0, 0.0], [2726228990.0, 23.0, 24.0, 2.0], [3437313790.0, 25.0, 26.0, 2.0], [-0.0740878358, 0.0, 0.0, 0.0], [0.0519033819, 0.0, 0.0, 0.0], [0.0794063956, 0.0, 0.0, 0.0], [-0.0141631728, 0.0, 0.0, 0.0], [-0.0663919747, 0.0, 0.0, 0.0], [0.024294043, 0.0, 0.0, 0.0], [0.00203605997, 0.0, 0.0, 0.0], [0.0963415354, 0.0, 0.0, 0.0], [0.00110510725, 0.0, 0.0, 0.0], [0.0799189135, 0.0, 0.0, 0.0], [-0.10428568, 0.0, 0.0, 0.0], [0.0100573907, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 21, 22, 12, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_125(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [10.8249998, 3.0, 4.0, 8.0], [7.76249981, 5.0, 6.0, 8.0], [348353.0, 7.0, 8.0, 7.0], [14.15625, 9.0, 10.0, 8.0], [3854757890.0, 11.0, 12.0, 7.0], [3233670910.0, 13.0, 14.0, 2.0], [8.03960037, 15.0, 16.0, 8.0], [2812186880.0, 17.0, 18.0, 2.0], [0.108221389, 0.0, 0.0, 0.0], [2.5, 19.0, 20.0, 1.0], [2666.0, 21.0, 22.0, 7.0], [0.00883317273, 0.0, 0.0, 0.0], [2726228990.0, 23.0, 24.0, 2.0], [3437313790.0, 25.0, 26.0, 2.0], [0.0740878209, 0.0, 0.0, 0.0], [-0.0519033819, 0.0, 0.0, 0.0], [-0.0794064254, 0.0, 0.0, 0.0], [0.01416317, 0.0, 0.0, 0.0], [0.0663919672, 0.0, 0.0, 0.0], [-0.0242940243, 0.0, 0.0, 0.0], [-0.00203607255, 0.0, 0.0, 0.0], [-0.0963415354, 0.0, 0.0, 0.0], [-0.00110510364, 0.0, 0.0, 0.0], [-0.0799189359, 0.0, 0.0, 0.0], [0.104285665, 0.0, 0.0, 0.0], [-0.0100574, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 21, 22, 12, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_126(xs):
    #Predicts Class 0
    function_dict = np.array([[56.1978989, 1.0, 2.0, 8.0], [39.5, 3.0, 4.0, 8.0], [21.5, 5.0, 6.0, 4.0], [4062966020.0, 7.0, 8.0, 2.0], [654.5, 9.0, 10.0, 0.0], [0.019360885, 0.0, 0.0, 0.0], [2542931970.0, 11.0, 12.0, 2.0], [3842726400.0, 13.0, 14.0, 2.0], [303.5, 15.0, 16.0, 0.0], [0.0843932554, 0.0, 0.0, 0.0], [-0.00184591801, 0.0, 0.0, 0.0], [-0.0842366368, 0.0, 0.0, 0.0], [0.0169495884, 0.0, 0.0, 0.0], [-0.00234830799, 0.0, 0.0, 0.0], [0.0675121173, 0.0, 0.0, 0.0], [-0.0782746822, 0.0, 0.0, 0.0], [0.00967136957, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_127(xs):
    #Predicts Class 1
    function_dict = np.array([[56.1978989, 1.0, 2.0, 8.0], [39.5, 3.0, 4.0, 8.0], [21.5, 5.0, 6.0, 4.0], [4062966020.0, 7.0, 8.0, 2.0], [654.5, 9.0, 10.0, 0.0], [-0.0193608701, 0.0, 0.0, 0.0], [2542931970.0, 11.0, 12.0, 2.0], [3842726400.0, 13.0, 14.0, 2.0], [303.5, 15.0, 16.0, 0.0], [-0.0843932852, 0.0, 0.0, 0.0], [0.00184594491, 0.0, 0.0, 0.0], [0.0842366368, 0.0, 0.0, 0.0], [-0.0169495903, 0.0, 0.0, 0.0], [0.00234830403, 0.0, 0.0, 0.0], [-0.0675121173, 0.0, 0.0, 0.0], [0.0782746896, 0.0, 0.0, 0.0], [-0.00967139751, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_128(xs):
    #Predicts Class 0
    function_dict = np.array([[3776773630.0, 1.0, 2.0, 2.0], [3437313790.0, 3.0, 4.0, 2.0], [3842726400.0, 5.0, 6.0, 2.0], [3236726780.0, 7.0, 8.0, 2.0], [0.0738880783, 0.0, 0.0, 0.0], [-0.108107373, 0.0, 0.0, 0.0], [10.8166504, 9.0, 10.0, 8.0], [3082723330.0, 11.0, 12.0, 2.0], [3292546050.0, 13.0, 14.0, 2.0], [1342256510.0, 15.0, 16.0, 3.0], [374939.0, 17.0, 18.0, 7.0], [0.0012920585, 0.0, 0.0, 0.0], [0.0765289739, 0.0, 0.0, 0.0], [-0.0920946151, 0.0, 0.0, 0.0], [-0.00374762085, 0.0, 0.0, 0.0], [0.0172720086, 0.0, 0.0, 0.0], [0.086546585, 0.0, 0.0, 0.0], [-0.062541239, 0.0, 0.0, 0.0], [0.0405386165, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 9, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_129(xs):
    #Predicts Class 1
    function_dict = np.array([[3776773630.0, 1.0, 2.0, 2.0], [3437313790.0, 3.0, 4.0, 2.0], [3842726400.0, 5.0, 6.0, 2.0], [3236726780.0, 7.0, 8.0, 2.0], [-0.0738881081, 0.0, 0.0, 0.0], [0.108107366, 0.0, 0.0, 0.0], [10.8166504, 9.0, 10.0, 8.0], [3082723330.0, 11.0, 12.0, 2.0], [3292546050.0, 13.0, 14.0, 2.0], [1342256510.0, 15.0, 16.0, 3.0], [374939.0, 17.0, 18.0, 7.0], [-0.00129206083, 0.0, 0.0, 0.0], [-0.076528959, 0.0, 0.0, 0.0], [0.09209463, 0.0, 0.0, 0.0], [0.00374759408, 0.0, 0.0, 0.0], [-0.0172719657, 0.0, 0.0, 0.0], [-0.0865465701, 0.0, 0.0, 0.0], [0.062541239, 0.0, 0.0, 0.0], [-0.040538609, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 9, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_130(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [485.0, 3.0, 4.0, 0.0], [549.5, 5.0, 6.0, 0.0], [314.5, 7.0, 8.0, 0.0], [289818.0, 9.0, 10.0, 7.0], [49.5, 11.0, 12.0, 4.0], [578.5, 13.0, 14.0, 0.0], [347074.0, 15.0, 16.0, 7.0], [14.1229496, 17.0, 18.0, 8.0], [-0.0340799168, 0.0, 0.0, 0.0], [8.74374962, 19.0, 20.0, 8.0], [1476925440.0, 21.0, 22.0, 2.0], [0.100464255, 0.0, 0.0, 0.0], [-0.109399013, 0.0, 0.0, 0.0], [1911294080.0, 23.0, 24.0, 2.0], [0.0501639731, 0.0, 0.0, 0.0], [-0.0452754386, 0.0, 0.0, 0.0], [-0.0912398845, 0.0, 0.0, 0.0], [-0.0243876483, 0.0, 0.0, 0.0], [0.064281106, 0.0, 0.0, 0.0], [0.0030003495, 0.0, 0.0, 0.0], [-0.0241412669, 0.0, 0.0, 0.0], [0.026207, 0.0, 0.0, 0.0], [0.0708951056, 0.0, 0.0, 0.0], [-0.018779492, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 21, 22, 12, 13, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 11, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_131(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [485.0, 3.0, 4.0, 0.0], [549.5, 5.0, 6.0, 0.0], [314.5, 7.0, 8.0, 0.0], [289818.0, 9.0, 10.0, 7.0], [49.5, 11.0, 12.0, 4.0], [578.5, 13.0, 14.0, 0.0], [347074.0, 15.0, 16.0, 7.0], [14.1229496, 17.0, 18.0, 8.0], [0.034079928, 0.0, 0.0, 0.0], [8.74374962, 19.0, 20.0, 8.0], [1476925440.0, 21.0, 22.0, 2.0], [-0.100464255, 0.0, 0.0, 0.0], [0.109399028, 0.0, 0.0, 0.0], [1911294080.0, 23.0, 24.0, 2.0], [-0.0501640216, 0.0, 0.0, 0.0], [0.0452754349, 0.0, 0.0, 0.0], [0.0912398845, 0.0, 0.0, 0.0], [0.0243876297, 0.0, 0.0, 0.0], [-0.064281106, 0.0, 0.0, 0.0], [-0.00300032762, 0.0, 0.0, 0.0], [0.0241412595, 0.0, 0.0, 0.0], [-0.0262070075, 0.0, 0.0, 0.0], [-0.0708951131, 0.0, 0.0, 0.0], [0.0187794883, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 21, 22, 12, 13, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 11, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_132(xs):
    #Predicts Class 0
    function_dict = np.array([[914675200.0, 1.0, 2.0, 2.0], [21.5, 3.0, 4.0, 4.0], [1323098110.0, 5.0, 6.0, 2.0], [18.375, 7.0, 8.0, 8.0], [597.0, 9.0, 10.0, 0.0], [235.0, 11.0, 12.0, 0.0], [86.0, 13.0, 14.0, 0.0], [0.106378809, 0.0, 0.0, 0.0], [-0.00567101641, 0.0, 0.0, 0.0], [336.0, 15.0, 16.0, 0.0], [0.0399231762, 0.0, 0.0, 0.0], [0.016666567, 0.0, 0.0, 0.0], [349623.0, 17.0, 18.0, 7.0], [19.1896, 19.0, 20.0, 8.0], [16.5, 21.0, 22.0, 4.0], [0.0107718064, 0.0, 0.0, 0.0], [-0.0850621685, 0.0, 0.0, 0.0], [0.0046906611, 0.0, 0.0, 0.0], [-0.124254912, 0.0, 0.0, 0.0], [-0.0959658474, 0.0, 0.0, 0.0], [0.0501174033, 0.0, 0.0, 0.0], [-0.0234725028, 0.0, 0.0, 0.0], [0.0238943603, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 10, 11, 17, 18, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 4, 9, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_133(xs):
    #Predicts Class 1
    function_dict = np.array([[914675200.0, 1.0, 2.0, 2.0], [21.5, 3.0, 4.0, 4.0], [1323098110.0, 5.0, 6.0, 2.0], [18.375, 7.0, 8.0, 8.0], [597.0, 9.0, 10.0, 0.0], [235.0, 11.0, 12.0, 0.0], [86.0, 13.0, 14.0, 0.0], [-0.106378824, 0.0, 0.0, 0.0], [0.00567105878, 0.0, 0.0, 0.0], [336.0, 15.0, 16.0, 0.0], [-0.0399231687, 0.0, 0.0, 0.0], [-0.0166665856, 0.0, 0.0, 0.0], [349623.0, 17.0, 18.0, 7.0], [19.1896, 19.0, 20.0, 8.0], [16.5, 21.0, 22.0, 4.0], [-0.0107718147, 0.0, 0.0, 0.0], [0.0850621909, 0.0, 0.0, 0.0], [-0.00469071185, 0.0, 0.0, 0.0], [0.12425492, 0.0, 0.0, 0.0], [0.0959658474, 0.0, 0.0, 0.0], [-0.050117366, 0.0, 0.0, 0.0], [0.0234724954, 0.0, 0.0, 0.0], [-0.023894364, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 10, 11, 17, 18, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 4, 9, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_134(xs):
    #Predicts Class 0
    function_dict = np.array([[52.5499992, 1.0, 2.0, 8.0], [39.5, 3.0, 4.0, 8.0], [21.5, 5.0, 6.0, 4.0], [3233494530.0, 7.0, 8.0, 9.0], [0.0609295107, 0.0, 0.0, 0.0], [0.0171401836, 0.0, 0.0, 0.0], [36.5, 9.0, 10.0, 4.0], [2417548800.0, 11.0, 12.0, 2.0], [-0.0593617558, 0.0, 0.0, 0.0], [-0.0896821171, 0.0, 0.0, 0.0], [0.00569516234, 0.0, 0.0, 0.0], [0.0177348703, 0.0, 0.0, 0.0], [-0.0132049955, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 10])
    branch_indices = np.array([0, 1, 3, 7, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_135(xs):
    #Predicts Class 1
    function_dict = np.array([[52.5499992, 1.0, 2.0, 8.0], [39.5, 3.0, 4.0, 8.0], [21.5, 5.0, 6.0, 4.0], [3233494530.0, 7.0, 8.0, 9.0], [-0.0609294847, 0.0, 0.0, 0.0], [-0.0171401761, 0.0, 0.0, 0.0], [36.5, 9.0, 10.0, 4.0], [2417548800.0, 11.0, 12.0, 2.0], [0.0593617335, 0.0, 0.0, 0.0], [0.0896820948, 0.0, 0.0, 0.0], [-0.00569517491, 0.0, 0.0, 0.0], [-0.0177348666, 0.0, 0.0, 0.0], [0.0132050049, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 10])
    branch_indices = np.array([0, 1, 3, 7, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_136(xs):
    #Predicts Class 0
    function_dict = np.array([[1.5, 1.0, 2.0, 1.0], [27.1353989, 3.0, 4.0, 8.0], [285.0, 5.0, 6.0, 0.0], [-0.078638427, 0.0, 0.0, 0.0], [35277.0, 7.0, 8.0, 7.0], [267.5, 9.0, 10.0, 0.0], [2669.5, 11.0, 12.0, 7.0], [19935.5, 13.0, 14.0, 7.0], [52.5499992, 15.0, 16.0, 8.0], [347932.5, 17.0, 18.0, 7.0], [-0.111983404, 0.0, 0.0, 0.0], [-0.0413118936, 0.0, 0.0, 0.0], [991314048.0, 19.0, 20.0, 2.0], [-0.00898268539, 0.0, 0.0, 0.0], [-0.0827053934, 0.0, 0.0, 0.0], [0.0761260763, 0.0, 0.0, 0.0], [-0.0177370589, 0.0, 0.0, 0.0], [-0.028330937, 0.0, 0.0, 0.0], [0.0355282091, 0.0, 0.0, 0.0], [-0.0216972046, 0.0, 0.0, 0.0], [0.0407666601, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 10, 11, 19, 20])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 6, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_137(xs):
    #Predicts Class 1
    function_dict = np.array([[1.5, 1.0, 2.0, 1.0], [27.1353989, 3.0, 4.0, 8.0], [285.0, 5.0, 6.0, 0.0], [0.0786384046, 0.0, 0.0, 0.0], [35277.0, 7.0, 8.0, 7.0], [267.5, 9.0, 10.0, 0.0], [2669.5, 11.0, 12.0, 7.0], [19935.5, 13.0, 14.0, 7.0], [52.5499992, 15.0, 16.0, 8.0], [347932.5, 17.0, 18.0, 7.0], [0.111983396, 0.0, 0.0, 0.0], [0.0413118899, 0.0, 0.0, 0.0], [991314048.0, 19.0, 20.0, 2.0], [0.0089826975, 0.0, 0.0, 0.0], [0.0827054009, 0.0, 0.0, 0.0], [-0.0761260688, 0.0, 0.0, 0.0], [0.017737044, 0.0, 0.0, 0.0], [0.0283309221, 0.0, 0.0, 0.0], [-0.0355282202, 0.0, 0.0, 0.0], [0.0216972046, 0.0, 0.0, 0.0], [-0.0407666601, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 10, 11, 19, 20])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 6, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_138(xs):
    #Predicts Class 0
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [549.5, 5.0, 6.0, 0.0], [315.0, 7.0, 8.0, 0.0], [361.0, 9.0, 10.0, 0.0], [462.0, 11.0, 12.0, 0.0], [25.5, 13.0, 14.0, 4.0], [-0.00663822331, 0.0, 0.0, 0.0], [-0.0776924565, 0.0, 0.0, 0.0], [141.5, 15.0, 16.0, 0.0], [13.9354496, 17.0, 18.0, 8.0], [389.0, 19.0, 20.0, 0.0], [0.079026185, 0.0, 0.0, 0.0], [2667.0, 21.0, 22.0, 7.0], [13213.5, 23.0, 24.0, 7.0], [0.0196462441, 0.0, 0.0, 0.0], [-0.0538338944, 0.0, 0.0, 0.0], [-0.0128134741, 0.0, 0.0, 0.0], [0.0709479377, 0.0, 0.0, 0.0], [0.0188360196, 0.0, 0.0, 0.0], [-0.0389902145, 0.0, 0.0, 0.0], [-0.0816711038, 0.0, 0.0, 0.0], [0.0819563195, 0.0, 0.0, 0.0], [0.058308918, 0.0, 0.0, 0.0], [-0.0684684366, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 12, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_139(xs):
    #Predicts Class 1
    function_dict = np.array([[1342256510.0, 1.0, 2.0, 3.0], [2.5, 3.0, 4.0, 1.0], [549.5, 5.0, 6.0, 0.0], [315.0, 7.0, 8.0, 0.0], [361.0, 9.0, 10.0, 0.0], [462.0, 11.0, 12.0, 0.0], [25.5, 13.0, 14.0, 4.0], [0.0066382247, 0.0, 0.0, 0.0], [0.0776924491, 0.0, 0.0, 0.0], [141.5, 15.0, 16.0, 0.0], [13.9354496, 17.0, 18.0, 8.0], [389.0, 19.0, 20.0, 0.0], [-0.079026185, 0.0, 0.0, 0.0], [2667.0, 21.0, 22.0, 7.0], [13213.5, 23.0, 24.0, 7.0], [-0.019646259, 0.0, 0.0, 0.0], [0.0538338907, 0.0, 0.0, 0.0], [0.0128134806, 0.0, 0.0, 0.0], [-0.0709479228, 0.0, 0.0, 0.0], [-0.0188360102, 0.0, 0.0, 0.0], [0.0389902331, 0.0, 0.0, 0.0], [0.0816711038, 0.0, 0.0, 0.0], [-0.0819563195, 0.0, 0.0, 0.0], [-0.0583089404, 0.0, 0.0, 0.0], [0.0684684142, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 12, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_140(xs):
    #Predicts Class 0
    function_dict = np.array([[35.5, 1.0, 2.0, 4.0], [30.5, 3.0, 4.0, 4.0], [3616746500.0, 5.0, 6.0, 2.0], [27.5, 7.0, 8.0, 4.0], [0.5, 9.0, 10.0, 5.0], [61.5, 11.0, 12.0, 4.0], [-0.0449974164, 0.0, 0.0, 0.0], [914675200.0, 13.0, 14.0, 2.0], [3547426300.0, 15.0, 16.0, 2.0], [444.0, 17.0, 18.0, 0.0], [0.0328969918, 0.0, 0.0, 0.0], [2029829250.0, 19.0, 20.0, 2.0], [-0.0439916737, 0.0, 0.0, 0.0], [0.0359980799, 0.0, 0.0, 0.0], [-0.0194509216, 0.0, 0.0, 0.0], [0.0919436961, 0.0, 0.0, 0.0], [-0.0383643694, 0.0, 0.0, 0.0], [0.0213920847, 0.0, 0.0, 0.0], [-0.106114142, 0.0, 0.0, 0.0], [0.011723469, 0.0, 0.0, 0.0], [0.102026351, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 10, 19, 20, 12, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_141(xs):
    #Predicts Class 1
    function_dict = np.array([[35.5, 1.0, 2.0, 4.0], [30.5, 3.0, 4.0, 4.0], [3616746500.0, 5.0, 6.0, 2.0], [27.5, 7.0, 8.0, 4.0], [0.5, 9.0, 10.0, 5.0], [61.5, 11.0, 12.0, 4.0], [0.0449974015, 0.0, 0.0, 0.0], [914675200.0, 13.0, 14.0, 2.0], [3547426300.0, 15.0, 16.0, 2.0], [444.0, 17.0, 18.0, 0.0], [-0.0328969806, 0.0, 0.0, 0.0], [2029829250.0, 19.0, 20.0, 2.0], [0.0439916924, 0.0, 0.0, 0.0], [-0.0359980613, 0.0, 0.0, 0.0], [0.0194509234, 0.0, 0.0, 0.0], [-0.091943711, 0.0, 0.0, 0.0], [0.038364362, 0.0, 0.0, 0.0], [-0.021392066, 0.0, 0.0, 0.0], [0.106114164, 0.0, 0.0, 0.0], [-0.0117234867, 0.0, 0.0, 0.0], [-0.102026343, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 10, 19, 20, 12, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_142(xs):
    #Predicts Class 0
    function_dict = np.array([[1860468100.0, 1.0, 2.0, 7.0], [3776773630.0, 3.0, 4.0, 2.0], [3443318270.0, 5.0, 6.0, 2.0], [1342256510.0, 7.0, 8.0, 3.0], [790394816.0, 9.0, 10.0, 10.0], [2672107520.0, 11.0, 12.0, 2.0], [0.0336962789, 0.0, 0.0, 0.0], [3222181890.0, 13.0, 14.0, 2.0], [319524.5, 15.0, 16.0, 7.0], [4021744640.0, 17.0, 18.0, 2.0], [279.0, 19.0, 20.0, 0.0], [27.5, 21.0, 22.0, 4.0], [-0.0847583786, 0.0, 0.0, 0.0], [-0.0264568385, 0.0, 0.0, 0.0], [0.0444593281, 0.0, 0.0, 0.0], [0.00272022816, 0.0, 0.0, 0.0], [0.0620478392, 0.0, 0.0, 0.0], [-0.102567188, 0.0, 0.0, 0.0], [0.0158969201, 0.0, 0.0, 0.0], [0.0488915183, 0.0, 0.0, 0.0], [-0.0125546511, 0.0, 0.0, 0.0], [-0.0417787321, 0.0, 0.0, 0.0], [0.0337249115, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_143(xs):
    #Predicts Class 1
    function_dict = np.array([[1860468100.0, 1.0, 2.0, 7.0], [3776773630.0, 3.0, 4.0, 2.0], [3443318270.0, 5.0, 6.0, 2.0], [1342256510.0, 7.0, 8.0, 3.0], [790394816.0, 9.0, 10.0, 10.0], [2672107520.0, 11.0, 12.0, 2.0], [-0.0336963274, 0.0, 0.0, 0.0], [3222181890.0, 13.0, 14.0, 2.0], [319524.5, 15.0, 16.0, 7.0], [4021744640.0, 17.0, 18.0, 2.0], [279.0, 19.0, 20.0, 0.0], [27.5, 21.0, 22.0, 4.0], [0.084758386, 0.0, 0.0, 0.0], [0.0264568347, 0.0, 0.0, 0.0], [-0.044459302, 0.0, 0.0, 0.0], [-0.00272021256, 0.0, 0.0, 0.0], [-0.062047828, 0.0, 0.0, 0.0], [0.102567166, 0.0, 0.0, 0.0], [-0.0158969164, 0.0, 0.0, 0.0], [-0.0488915369, 0.0, 0.0, 0.0], [0.0125546455, 0.0, 0.0, 0.0], [0.0417787321, 0.0, 0.0, 0.0], [-0.0337249041, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def logit_class_0(xs):
    sum_of_leaf_values = np.zeros(xs.shape[0])
    if pool != -1:
        sum_of_leaf_values = np.sum(list(pool.starmap(apply,[(eval('booster_' + str(booster_index)), xs) for booster_index in range(0,144,2)])), axis=0)
    else:
        for booster_index in range(0,144,2):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def logit_class_1(xs):
    sum_of_leaf_values = np.zeros(xs.shape[0])
    if pool != -1:
        sum_of_leaf_values = np.sum(list(pool.starmap(apply,[(eval('booster_' + str(booster_index)), xs) for booster_index in range(1,144,2)])), axis=0)
    else:
        for booster_index in range(1,144,2):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def classify(rows, return_probabilities=False):
    rows=np.array(rows)
    logits = []
    logits.append(logit_class_0)
    logits.append(logit_class_1)
    o = np.array([logits[class_index](rows) for class_index in range(2)]).T
    if not return_probabilities:
        return np.argmax(o,axis=1)
    else:
        exps = np.exp(o)
        Z = np.sum(exps, axis=1).reshape(-1, 1)
        return exps/Z





def Validate(cleanarr):
    #note that classification is a single line of code 
    outputs = classify(cleanarr[:, :-1]) 
    #metrics 
    count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0 
    correct_count = int(np.sum(outputs.reshape(-1) == cleanarr[:, -1].reshape(-1))) 
    count = outputs.shape[0] 
    num_TP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, cleanarr[:, -1].reshape(-1) == 1))) 
    num_TN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, cleanarr[:, -1].reshape(-1) == 0))) 
    num_FN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, cleanarr[:, -1].reshape(-1) == 1))) 
    num_FP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, cleanarr[:, -1].reshape(-1) == 0))) 
    num_class_0 = int(np.sum(cleanarr[:, -1].reshape(-1) == 0)) 
    num_class_1 = int(np.sum(cleanarr[:, -1].reshape(-1) == 1)) 
    return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, outputs


def Predict(arr,headerless,csvfile, get_key, classmapping):
    with open(csvfile, 'r') as csvinput:
        #readers and writers
        reader = csv.reader(csvinput)

        #print original header
        if (not headerless):
            print(','.join(next(reader, None) + ["Prediction"]))
        
        outputs = classify(arr)
        for i, row in enumerate(reader):
            #use the transformed array as input to predictor
            pred = str(get_key(int(outputs[i]), classmapping))
            #use original untransformed line to write out
            row.append(pred)
            print(','.join(['"' + field + '"' if ',' in field else field for field in row]))



# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile', action='store_true', help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    parser.add_argument('-json', action="store_true", default=False, help="report measurements as json")
    parser.add_argument('-trim', action="store_true", help="If true, the prediction will not output ignored columns.")
    args = parser.parse_args()
    faulthandler.enable()
    if args.validate:
        args.trim = True
    #clean if not already clean
    if not args.cleanfile:
        tempdir = tempfile.gettempdir()
        cleanfile = tempfile.NamedTemporaryFile().name
        preprocessedfile = tempfile.NamedTemporaryFile().name
        output = preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate), trim=args.trim)
        get_key, classmapping = clean(preprocessedfile if output!=-1 else args.csvfile, cleanfile, -1, args.headerless, (not args.validate), trim=args.trim)
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x, y: x
        classmapping = {}
        output=None


    #load file
    cleanarr = np.loadtxt(cleanfile, delimiter=',', dtype='float64')
    if not args.trim and ignorecolumns != []:
        cleanarr = cleanarr[:, important_idxs]


    #Predict
    if not args.validate:
        Predict(cleanarr, args.headerless, preprocessedfile if output!=-1 else args.csvfile, get_key, classmapping)


    #Validate
    else:
        classifier_type = 'RF'
        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds = Validate(cleanarr)
        #Correct Labels
        true_labels = cleanarr[:, -1]

        #Report Metrics
        model_cap=12
        if args.json:
            import json
        if n_classes == 2:
            #Base metrics
            FN = float(num_FN) * 100.0 / float(count)
            FP = float(num_FP) * 100.0 / float(count)
            TN = float(num_TN) * 100.0 / float(count)
            TP = float(num_TP) * 100.0 / float(count)
            num_correct = correct_count

            #Calculated Metrics
            if int(num_TP + num_FN) != 0:
                TPR = num_TP / (num_TP + num_FN) # Sensitivity, Recall
            if int(num_TN + num_FP) != 0:
                TNR = num_TN / (num_TN + num_FP) # Specificity
            if int(num_TP + num_FP) != 0:
                PPV = num_TP / (num_TP + num_FP) # Recall
            if int(num_FN + num_TP) != 0:
                FNR = num_FN / (num_FN + num_TP) # Miss rate
            if int(2 * num_TP + num_FP + num_FN) != 0:
                FONE = 2 * num_TP / (2 * num_TP + num_FP + num_FN) # F1 Score
            if int(num_TP + num_FN + num_FP) != 0:
                TS = num_TP / (num_TP + num_FN + num_FP) # Critical Success Index
            #Best Guess Accuracy
            randguess = int(float(10000.0 * max(num_class_1, num_class_0)) / count) / 100.0
            #Model Accuracy
            classbalance = [float(num_class_0)/count, float(num_class_1)/count]
            H = float(-1.0 * sum([classbalance[i] * math.log(classbalance[i]) / math.log(2) for i in range(len(classbalance))]))

            modelacc = int(float(num_correct * 10000) / count) / 100.0
            #Report
            json_dict = {'instance_count':                        count ,
                         'classifier_type':                        classifier_type,
                         'classes':                            2 ,
                         'false_negative_instances':    num_FN ,
                         'false_positive_instances':    num_FP ,
                         'true_positive_instances':    num_TP ,
                         'true_negative_instances':    num_TN,
                         'false_negatives':                        FN ,
                         'false_positives':                        FP ,
                         'true_negatives':                        TN ,
                         'true_positives':                        TP ,
                         'number_correct':                        num_correct ,
                         'accuracy': {
                             'best_guess': randguess,
                             'improvement': modelacc-randguess,
                             'model_accuracy': modelacc,
                         },
                         'model_capacity':                        model_cap ,
                         'generalization_ratio':                int(float(num_correct * 100) / model_cap) * H/ 100.0,
                         'model_efficiency':                    int(100 * (modelacc - randguess) / model_cap) / 100.0,
                        'shannon_entropy_of_labels':           H,
                        'classbalance':                        classbalance}
            if classifier_type == 'NN':
                json_dict['capacity_utilized_by_nn'] = cap_utilized # noqa
            if args.json:
                pass
            else:
                if classifier_type == 'NN':
                    print("Classifier Type:                    Neural Network")
                elif classifier_type == 'RF':
                    print("Classifier Type:                    Random Forest")
                else:
                    print("Classifier Type:                    Decision Tree")
                print("System Type:                        Binary classifier")
                print("Best-guess accuracy:                {:.2f}%".format(randguess))
                print("Model accuracy:                     {:.2f}%".format(modelacc) + " (" + str(int(num_correct)) + "/" + str(count) + " correct)")
                print("Improvement over best guess:        {:.2f}%".format(modelacc - randguess) + " (of possible " + str(round(100 - randguess, 2)) + "%)")
                print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
                if classifier_type == 'NN':
                    print("Model Capacity Utilized:            {:.0f} bits".format(cap_utilized)) # noqa
                print("Generalization ratio:               {:.2f}".format(int(float(num_correct * 100) / model_cap) / 100.0 * H) + " bits/bit")
                print("Model efficiency:                   {:.2f}%/parameter".format(int(100 * (modelacc - randguess) / model_cap) / 100.0))
                print("System behavior")
                print("True Negatives:                     {:.2f}%".format(TN) + " (" + str(int(num_TN)) + "/" + str(count) + ")")
                print("True Positives:                     {:.2f}%".format(TP) + " (" + str(int(num_TP)) + "/" + str(count) + ")")
                print("False Negatives:                    {:.2f}%".format(FN) + " (" + str(int(num_FN)) + "/" + str(count) + ")")
                print("False Positives:                    {:.2f}%".format(FP) + " (" + str(int(num_FP)) + "/" + str(count) + ")")
                if int(num_TP + num_FN) != 0:
                    print("True Pos. Rate/Sensitivity/Recall:  {:.2f}".format(TPR))
                if int(num_TN + num_FP) != 0:
                    print("True Neg. Rate/Specificity:         {:.2f}".format(TNR))
                if int(num_TP + num_FP) != 0:
                    print("Precision:                          {:.2f}".format(PPV))
                if int(2 * num_TP + num_FP + num_FN) != 0:
                    print("F-1 Measure:                        {:.2f}".format(FONE))
                if int(num_TP + num_FN) != 0:
                    print("False Negative Rate/Miss Rate:      {:.2f}".format(FNR))
                if int(num_TP + num_FN + num_FP) != 0:
                    print("Critical Success Index:             {:.2f}".format(TS))
        #Multiclass
        else:
            num_correct = correct_count
            modelacc = int(float(num_correct * 10000) / count) / 100.0
            randguess = round(max(numeachclass.values()) / sum(numeachclass.values()) * 100, 2)
            classbalance = [float(numofcertainclass) / count for numofcertainclass in numeachclass.values()]
            H = float(-1.0 * sum([classbalance[i] * math.log(classbalance[i]) / math.log(2) for i in range(len(classbalance))]))

            if args.json:
                json_dict = {'instance_count':                        count,
                            'classifier_type':                        classifier_type,
                            'classes':                            n_classes,
                             'number_correct': num_correct,
                             'accuracy': {
                                 'best_guess': randguess,
                                 'improvement': modelacc - randguess,
                                 'model_accuracy': modelacc,
                             },
                             'model_capacity': model_cap,
                            'generalization_ratio':                int(float(num_correct * 100) / model_cap) / 100.0 * H,
                            'model_efficiency':                    int(100 * (modelacc - randguess) / model_cap) / 100.0,
                        'shannon_entropy_of_labels':           H,
                        'classbalance':                        classbalance}
                if classifier_type == 'NN':
                    json_dict['capacity_utilized_by_nn'] = cap_utilized # noqa
            else:
                if classifier_type == 'NN':
                    print("Classifier Type:                    Neural Network")
                elif classifier_type == 'RF':
                    print("Classifier Type:                    Random Forest")
                else:
                    print("Classifier Type:                    Decision Tree")
                print("System Type:                        " + str(n_classes) + "-way classifier")
                print("Best-guess accuracy:                {:.2f}%".format(randguess))
                print("Model accuracy:                     {:.2f}%".format(modelacc) + " (" + str(int(num_correct)) + "/" + str(count) + " correct)")
                print("Improvement over best guess:        {:.2f}%".format(modelacc - randguess) + " (of possible " + str(round(100 - randguess, 2)) + "%)")
                print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
                if classifier_type == 'NN':
                    print("Model Capacity Utilized:            {:.0f} bits".format(cap_utilized)) # noqa              
                print("Generalization ratio:               {:.2f}".format(int(float(num_correct * 100) / model_cap) / 100.0 * H) + " bits/bit")
                print("Model efficiency:                   {:.2f}%/parameter".format(int(100 * (modelacc - randguess) / model_cap) / 100.0))

        try:
            import numpy as np # For numpy see: http://numpy.org
            from numpy import array
        except:
            print("Note: If you install numpy (https://www.numpy.org) and scipy (https://www.scipy.org) this predictor generates a confusion matrix")

        def confusion_matrix(y_true, y_pred, json, labels=None, sample_weight=None, normalize=None):
            stats = {}
            if labels is None:
                labels = np.array(list(set(list(y_true.astype('int')))))
            else:
                labels = np.asarray(labels)
                if np.all([l not in y_true for l in labels]):
                    raise ValueError("At least one label specified must be in y_true")
            n_labels = labels.size

            for class_i in range(n_labels):
                stats[class_i] = {'TP':{},'FP':{},'FN':{},'TN':{}}
                class_i_indices = np.argwhere(y_true==class_i) #indices with bus(call class_i=bus in this example)
                not_class_i_indices = np.argwhere(y_true!=class_i) #indices with not bus
                stats[int(class_i)]['TP'] = int(np.sum(y_pred[class_i_indices] == class_i)) #indices where bus, and we predict == bus
                stats[int(class_i)]['FN'] = int(np.sum(y_pred[class_i_indices] != class_i)) #indices where bus, and we predict != bus
                stats[int(class_i)]['TN'] = int(np.sum(y_pred[not_class_i_indices] != class_i)) #indices with not bus, where we predict != bus
                stats[int(class_i)]['FP'] = int(np.sum(y_pred[not_class_i_indices] == class_i)) #indices where not bus, we predict as bus
            #check for numpy/scipy is imported
            try:
                from scipy.sparse import coo_matrix #required for multiclass metrics
            except:
                if not json:
                    print("Note: If you install scipy (https://www.scipy.org) this predictor generates a confusion matrix")
                    sys.exit()
                else:
                    return np.array([]), stats
                

            # Compute confusion matrix to evaluate the accuracy of a classification.
            # By definition a confusion matrix :math:C is such that :math:C_{i, j}
            # is equal to the number of observations known to be in group :math:i and
            # predicted to be in group :math:j.
            # Thus in binary classification, the count of true negatives is
            # :math:C_{0,0}, false negatives is :math:C_{1,0}, true positives is
            # :math:C_{1,1} and false positives is :math:C_{0,1}.
            # Read more in the :ref:User Guide <confusion_matrix>.
            # Parameters
            # ----------
            # y_true : array-like of shape (n_samples,)
            # Ground truth (correct) target values.
            # y_pred : array-like of shape (n_samples,)
            # Estimated targets as returned by a classifier.
            # labels : array-like of shape (n_classes), default=None
            # List of labels to index the matrix. This may be used to reorder
            # or select a subset of labels.
            # If None is given, those that appear at least once
            # in y_true or y_pred are used in sorted order.
            # sample_weight : array-like of shape (n_samples,), default=None
            # Sample weights.
            # normalize : {'true', 'pred', 'all'}, default=None
            # Normalizes confusion matrix over the true (rows), predicted (columns)
            # conditions or all the population. If None, confusion matrix will not be
            # normalized.
            # Returns
            # -------
            # C : ndarray of shape (n_classes, n_classes)
            # Confusion matrix.
            # References
            # ----------



            if sample_weight is None:
                sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
            else:
                sample_weight = np.asarray(sample_weight)
            if y_true.shape[0]!=y_pred.shape[0]:
                raise ValueError("y_true and y_pred must be of the same length")

            if normalize not in ['true', 'pred', 'all', None]:
                raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")


            label_to_ind = {y: x for x, y in enumerate(labels)}
            # convert yt, yp into index
            y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
            y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
            # intersect y_pred, y_true with labels, eliminate items not in labels
            ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
            y_pred = y_pred[ind]
            y_true = y_true[ind]

            # also eliminate weights of eliminated items
            sample_weight = sample_weight[ind]
            # Choose the accumulator dtype to always have high precision
            if sample_weight.dtype.kind in {'i', 'u', 'b'}:
                dtype = np.int64
            else:
                dtype = np.float64
            cm = coo_matrix((sample_weight, (y_true, y_pred)), shape=(n_labels, n_labels), dtype=dtype,).toarray()


            with np.errstate(all='ignore'):
                if normalize == 'true':
                    cm = cm / cm.sum(axis=1, keepdims=True)
                elif normalize == 'pred':
                    cm = cm / cm.sum(axis=0, keepdims=True)
                elif normalize == 'all':
                    cm = cm / cm.sum()
                cm = np.nan_to_num(cm)
            return cm, stats
        mtrx, stats = confusion_matrix(np.array(true_labels).reshape(-1), np.array(preds).reshape(-1), args.json)
        if args.json:
            json_dict['confusion_matrix'] = mtrx.tolist()
            json_dict['multiclass_stats'] = stats
            print(json.dumps(json_dict))
        else:
            mtrx = mtrx / np.sum(mtrx) * 100.0
            print("Confusion Matrix:")
            print(' ' + np.array2string(mtrx, formatter={'float': (lambda x: '{:.2f}%'.format(round(float(x), 2)))})[1:-1])

    #Clean Up
    if not args.cleanfile:
        os.remove(cleanfile)
        if output!=-1:
            os.remove(preprocessedfile)
