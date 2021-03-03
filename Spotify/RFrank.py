#!/usr/bin/env python3
#
# This code has been produced by a free evaluation version of Daimensions(tm).
# Portions of this code copyright (c) 2019, 2020 by Brainome, Inc. All Rights Reserved.
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
# Output of Brainome Daimensions(tm) 0.991 Table Compiler v0.99.
# Invocation: btc train.csv -headerless -f RF -rank -o RFrank.py -riskoverfit --yes
# Total compiler execution time: 0:00:23.55. Finished on: Mar-03-2021 20:01:35.
# This source code requires Python 3.
#
"""
Classifier Type:                    Random Forest
System Type:                         Binary classifier
Best-guess accuracy:                 56.62%
Overall Model accuracy:              88.37% (540/611 correct)
Overall Improvement over best guess: 31.75% (of possible 43.38%)
Model capacity (MEC):                10 bits
Generalization ratio:                53.31 bits/bit
Model efficiency:                    3.17%/parameter
System behavior
True Negatives:                      37.15% (227/611)
True Positives:                      51.23% (313/611)
False Negatives:                     5.40% (33/611)
False Positives:                     6.22% (38/611)
True Pos. Rate/Sensitivity/Recall:   0.90
True Neg. Rate/Specificity:          0.86
Precision:                           0.89
F-1 Measure:                         0.90
False Negative Rate/Miss Rate:       0.10
Critical Success Index:              0.82
Confusion Matrix:
 [37.15% 6.22%]
 [5.40% 51.23%]
Generalization index:                26.53
Percent of Data Memorized:           3.77%
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
{"to_select_idxs":[5, 9], "to_ignore_idxs":[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16], "overfit_risk":1.509903313490213e-14, "risk_progression":[48.843761847633594, 53.21595555581067], "test_accuracy_progression":[[5, 0.8936170212765957], [9, 0.9328968903436988]]}

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
TRAINFILE = "train.csv"

try:
    import numpy as np # For numpy see: http://numpy.org
    from numpy import array
except:
    print("This predictor requires the Numpy library. For installation instructions please refer to: http://numpy.org")

#Number of attributes
num_attr = 2
n_classes = 2
transform_true = False

# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=["0","1","2","3","4","6","7","8","10","11","12","13","14","15","16",]
target=""
important_idxs=[5,9]

def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[], trim=False):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=["0","1","2","3","4","6","7","8","10","11","12","13","14","15","16",]
    target=""
    important_idxs=[5,9]
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
    clean.mapping = {}
    

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


# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)
# Classifier
def apply(f, x):
    return f(x)

def booster_0(xs):
    #Predicts Class 0
    function_dict = np.array([[0.601999998, 1.0, 2.0, 0.0], [0.568499982, 3.0, 4.0, 0.0], [0.730499983, 5.0, 6.0, 0.0], [0.302999973, 7.0, 8.0, 0.0], [0.595000029, 9.0, 10.0, 0.0], [0.5, 11.0, 12.0, 1.0], [0.740499973, 13.0, 14.0, 0.0], [-0.449908137, 0.0, 0.0, 0.0], [0.309499979, 15.0, 16.0, 0.0], [0.578500032, 17.0, 18.0, 0.0], [0.5, 19.0, 20.0, 1.0], [0.63499999, 21.0, 22.0, 0.0], [0.709499955, 23.0, 24.0, 0.0], [0.435466677, 0.0, 0.0, 0.0], [0.863499999, 25.0, 26.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [-0.326600015, 0.0, 0.0, 0.0], [-0.163300008, 0.0, 0.0, 0.0], [0.0699857175, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [-0.381033331, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [0.244949996, 0.0, 0.0, 0.0], [-0.0518717654, 0.0, 0.0, 0.0], [0.139971435, 0.0, 0.0, 0.0], [0.251570255, 0.0, 0.0, 0.0], [0.377922863, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 25, 26])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11, 12, 6, 14])
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
    function_dict = np.array([[0.601999998, 1.0, 2.0, 0.0], [0.568499982, 3.0, 4.0, 0.0], [0.730499983, 5.0, 6.0, 0.0], [0.302999973, 7.0, 8.0, 0.0], [0.595000029, 9.0, 10.0, 0.0], [0.5, 11.0, 12.0, 1.0], [0.740499973, 13.0, 14.0, 0.0], [0.449908137, 0.0, 0.0, 0.0], [0.309499979, 15.0, 16.0, 0.0], [0.578500032, 17.0, 18.0, 0.0], [0.5, 19.0, 20.0, 1.0], [0.63499999, 21.0, 22.0, 0.0], [0.709499955, 23.0, 24.0, 0.0], [-0.435466677, 0.0, 0.0, 0.0], [0.863499999, 25.0, 26.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [0.326600015, 0.0, 0.0, 0.0], [0.163300008, 0.0, 0.0, 0.0], [-0.0699857175, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [0.381033331, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [-0.244949996, 0.0, 0.0, 0.0], [0.0518717654, 0.0, 0.0, 0.0], [-0.139971435, 0.0, 0.0, 0.0], [-0.251570255, 0.0, 0.0, 0.0], [-0.377922863, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 25, 26])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11, 12, 6, 14])
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
    function_dict = np.array([[0.601999998, 1.0, 2.0, 0.0], [0.323000014, 3.0, 4.0, 0.0], [0.730499983, 5.0, 6.0, 0.0], [-0.298605978, 0.0, 0.0, 0.0], [0.380999982, 7.0, 8.0, 0.0], [0.605499983, 9.0, 10.0, 0.0], [0.754000008, 11.0, 12.0, 0.0], [0.5, 13.0, 14.0, 1.0], [0.5, 15.0, 16.0, 1.0], [0.357456952, 0.0, 0.0, 0.0], [0.725000024, 17.0, 18.0, 0.0], [0.266000599, 0.0, 0.0, 0.0], [0.774500012, 19.0, 20.0, 0.0], [-0.254345715, 0.0, 0.0, 0.0], [0.0149948094, 0.0, 0.0, 0.0], [-0.0936168656, 0.0, 0.0, 0.0], [-0.24734664, 0.0, 0.0, 0.0], [0.0279258601, 0.0, 0.0, 0.0], [-0.27969864, 0.0, 0.0, 0.0], [-0.111545369, 0.0, 0.0, 0.0], [0.191849291, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 9, 17, 18, 11, 19, 20])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 10, 6, 12])
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
    function_dict = np.array([[0.601999998, 1.0, 2.0, 0.0], [0.323000014, 3.0, 4.0, 0.0], [0.730499983, 5.0, 6.0, 0.0], [0.298605978, 0.0, 0.0, 0.0], [0.380999982, 7.0, 8.0, 0.0], [0.605499983, 9.0, 10.0, 0.0], [0.754000008, 11.0, 12.0, 0.0], [0.5, 13.0, 14.0, 1.0], [0.5, 15.0, 16.0, 1.0], [-0.357456952, 0.0, 0.0, 0.0], [0.725000024, 17.0, 18.0, 0.0], [-0.266000599, 0.0, 0.0, 0.0], [0.774500012, 19.0, 20.0, 0.0], [0.254345715, 0.0, 0.0, 0.0], [-0.0149948094, 0.0, 0.0, 0.0], [0.093616873, 0.0, 0.0, 0.0], [0.247346669, 0.0, 0.0, 0.0], [-0.0279259067, 0.0, 0.0, 0.0], [0.27969861, 0.0, 0.0, 0.0], [0.111545347, 0.0, 0.0, 0.0], [-0.191849276, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 9, 17, 18, 11, 19, 20])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 10, 6, 12])
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
    function_dict = np.array([[0.568499982, 1.0, 2.0, 0.0], [0.271499991, 3.0, 4.0, 0.0], [0.701499999, 5.0, 6.0, 0.0], [-0.272910088, 0.0, 0.0, 0.0], [0.514500022, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.708500028, 11.0, 12.0, 0.0], [0.510499954, 13.0, 14.0, 0.0], [0.5, 15.0, 16.0, 1.0], [0.63499999, 17.0, 18.0, 0.0], [0.572000027, 19.0, 20.0, 0.0], [0.300703019, 0.0, 0.0, 0.0], [0.718999982, 21.0, 22.0, 0.0], [-0.128276572, 0.0, 0.0, 0.0], [0.504520953, 0.0, 0.0, 0.0], [-0.0847043246, 0.0, 0.0, 0.0], [-0.293965608, 0.0, 0.0, 0.0], [0.000239010304, 0.0, 0.0, 0.0], [0.164177671, 0.0, 0.0, 0.0], [0.233534575, 0.0, 0.0, 0.0], [-0.0591271669, 0.0, 0.0, 0.0], [-0.0747445822, 0.0, 0.0, 0.0], [0.114435412, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 19, 20, 11, 21, 22])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 10, 6, 12])
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
    function_dict = np.array([[0.568499982, 1.0, 2.0, 0.0], [0.271499991, 3.0, 4.0, 0.0], [0.701499999, 5.0, 6.0, 0.0], [0.272910088, 0.0, 0.0, 0.0], [0.514500022, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.708500028, 11.0, 12.0, 0.0], [0.510499954, 13.0, 14.0, 0.0], [0.5, 15.0, 16.0, 1.0], [0.63499999, 17.0, 18.0, 0.0], [0.572000027, 19.0, 20.0, 0.0], [-0.300703019, 0.0, 0.0, 0.0], [0.718999982, 21.0, 22.0, 0.0], [0.128276572, 0.0, 0.0, 0.0], [-0.504520953, 0.0, 0.0, 0.0], [0.084704347, 0.0, 0.0, 0.0], [0.293965608, 0.0, 0.0, 0.0], [-0.000239024361, 0.0, 0.0, 0.0], [-0.164177701, 0.0, 0.0, 0.0], [-0.233534589, 0.0, 0.0, 0.0], [0.059127178, 0.0, 0.0, 0.0], [0.0747445598, 0.0, 0.0, 0.0], [-0.114435434, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 19, 20, 11, 21, 22])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 10, 6, 12])
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
    function_dict = np.array([[0.553499997, 1.0, 2.0, 0.0], [0.271499991, 3.0, 4.0, 0.0], [0.5, 5.0, 6.0, 1.0], [-0.241723284, 0.0, 0.0, 0.0], [0.514500022, 7.0, 8.0, 0.0], [0.790500045, 9.0, 10.0, 0.0], [0.644500017, 11.0, 12.0, 0.0], [0.484499991, 13.0, 14.0, 0.0], [0.537, 15.0, 16.0, 0.0], [0.613999963, 17.0, 18.0, 0.0], [0.82249999, 19.0, 20.0, 0.0], [0.628000021, 21.0, 22.0, 0.0], [0.792999983, 23.0, 24.0, 0.0], [-0.0966622829, 0.0, 0.0, 0.0], [0.120990083, 0.0, 0.0, 0.0], [-0.269655973, 0.0, 0.0, 0.0], [-0.101648189, 0.0, 0.0, 0.0], [0.157338411, 0.0, 0.0, 0.0], [0.00212204247, 0.0, 0.0, 0.0], [0.275767028, 0.0, 0.0, 0.0], [0.0802615732, 0.0, 0.0, 0.0], [-0.0160172284, 0.0, 0.0, 0.0], [-0.210687801, 0.0, 0.0, 0.0], [0.070643045, 0.0, 0.0, 0.0], [-0.108985536, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_7(xs):
    #Predicts Class 1
    function_dict = np.array([[0.553499997, 1.0, 2.0, 0.0], [0.271499991, 3.0, 4.0, 0.0], [0.5, 5.0, 6.0, 1.0], [0.241723284, 0.0, 0.0, 0.0], [0.514500022, 7.0, 8.0, 0.0], [0.790500045, 9.0, 10.0, 0.0], [0.644500017, 11.0, 12.0, 0.0], [0.484499991, 13.0, 14.0, 0.0], [0.537, 15.0, 16.0, 0.0], [0.613999963, 17.0, 18.0, 0.0], [0.82249999, 19.0, 20.0, 0.0], [0.628000021, 21.0, 22.0, 0.0], [0.792999983, 23.0, 24.0, 0.0], [0.0966622978, 0.0, 0.0, 0.0], [-0.120990053, 0.0, 0.0, 0.0], [0.269655973, 0.0, 0.0, 0.0], [0.101648189, 0.0, 0.0, 0.0], [-0.157338411, 0.0, 0.0, 0.0], [-0.00212201383, 0.0, 0.0, 0.0], [-0.275767028, 0.0, 0.0, 0.0], [-0.080261603, 0.0, 0.0, 0.0], [0.0160172191, 0.0, 0.0, 0.0], [0.210687816, 0.0, 0.0, 0.0], [-0.0706430301, 0.0, 0.0, 0.0], [0.108985528, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_8(xs):
    #Predicts Class 0
    function_dict = np.array([[0.637500048, 1.0, 2.0, 0.0], [0.271499991, 3.0, 4.0, 0.0], [0.881000042, 5.0, 6.0, 0.0], [-0.217811093, 0.0, 0.0, 0.0], [0.61500001, 7.0, 8.0, 0.0], [0.657500029, 9.0, 10.0, 0.0], [0.944499969, 11.0, 12.0, 0.0], [0.601999998, 13.0, 14.0, 0.0], [0.622500002, 15.0, 16.0, 0.0], [0.5, 17.0, 18.0, 1.0], [0.670000017, 19.0, 20.0, 0.0], [0.205328867, 0.0, 0.0, 0.0], [-0.0217213128, 0.0, 0.0, 0.0], [-0.0523778535, 0.0, 0.0, 0.0], [0.138399065, 0.0, 0.0, 0.0], [-0.226877213, 0.0, 0.0, 0.0], [-0.0697707459, 0.0, 0.0, 0.0], [0.247326523, 0.0, 0.0, 0.0], [0.0818437859, 0.0, 0.0, 0.0], [-0.112236641, 0.0, 0.0, 0.0], [0.0239512883, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 10, 6])
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
    function_dict = np.array([[0.637500048, 1.0, 2.0, 0.0], [0.271499991, 3.0, 4.0, 0.0], [0.881000042, 5.0, 6.0, 0.0], [0.217811123, 0.0, 0.0, 0.0], [0.61500001, 7.0, 8.0, 0.0], [0.657500029, 9.0, 10.0, 0.0], [0.944499969, 11.0, 12.0, 0.0], [0.601999998, 13.0, 14.0, 0.0], [0.622500002, 15.0, 16.0, 0.0], [0.5, 17.0, 18.0, 1.0], [0.670000017, 19.0, 20.0, 0.0], [-0.205328882, 0.0, 0.0, 0.0], [0.0217212997, 0.0, 0.0, 0.0], [0.0523778535, 0.0, 0.0, 0.0], [-0.138399035, 0.0, 0.0, 0.0], [0.226877257, 0.0, 0.0, 0.0], [0.0697707683, 0.0, 0.0, 0.0], [-0.247326553, 0.0, 0.0, 0.0], [-0.0818437785, 0.0, 0.0, 0.0], [0.112236664, 0.0, 0.0, 0.0], [-0.0239512883, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 9, 10, 6])
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
    function_dict = np.array([[0.578500032, 1.0, 2.0, 0.0], [0.271499991, 3.0, 4.0, 0.0], [0.858500004, 5.0, 6.0, 0.0], [-0.195980698, 0.0, 0.0, 0.0], [0.380999982, 7.0, 8.0, 0.0], [0.851999998, 9.0, 10.0, 0.0], [0.862499952, 11.0, 12.0, 0.0], [0.5, 13.0, 14.0, 1.0], [0.402999997, 15.0, 16.0, 0.0], [0.834499955, 17.0, 18.0, 0.0], [-0.308123142, 0.0, 0.0, 0.0], [0.209888935, 0.0, 0.0, 0.0], [0.881000042, 19.0, 20.0, 0.0], [-0.109632105, 0.0, 0.0, 0.0], [0.112023778, 0.0, 0.0, 0.0], [-0.251238018, 0.0, 0.0, 0.0], [-0.0338854119, 0.0, 0.0, 0.0], [0.00655693188, 0.0, 0.0, 0.0], [0.232680395, 0.0, 0.0, 0.0], [-0.075371854, 0.0, 0.0, 0.0], [0.124440983, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_11(xs):
    #Predicts Class 1
    function_dict = np.array([[0.578500032, 1.0, 2.0, 0.0], [0.271499991, 3.0, 4.0, 0.0], [0.858500004, 5.0, 6.0, 0.0], [0.195980698, 0.0, 0.0, 0.0], [0.380999982, 7.0, 8.0, 0.0], [0.851999998, 9.0, 10.0, 0.0], [0.862499952, 11.0, 12.0, 0.0], [0.5, 13.0, 14.0, 1.0], [0.402999997, 15.0, 16.0, 0.0], [0.834499955, 17.0, 18.0, 0.0], [0.308123112, 0.0, 0.0, 0.0], [-0.20988895, 0.0, 0.0, 0.0], [0.881000042, 19.0, 20.0, 0.0], [0.109632097, 0.0, 0.0, 0.0], [-0.112023741, 0.0, 0.0, 0.0], [0.251238018, 0.0, 0.0, 0.0], [0.0338853896, 0.0, 0.0, 0.0], [-0.00655693188, 0.0, 0.0, 0.0], [-0.23268041, 0.0, 0.0, 0.0], [0.0753718093, 0.0, 0.0, 0.0], [-0.124440953, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_12(xs):
    #Predicts Class 0
    function_dict = np.array([[0.342999995, 1.0, 2.0, 0.0], [0.5, 3.0, 4.0, 1.0], [0.364499986, 5.0, 6.0, 0.0], [0.301999986, 7.0, 8.0, 0.0], [0.277999997, 9.0, 10.0, 0.0], [0.5, 11.0, 12.0, 1.0], [0.5, 13.0, 14.0, 1.0], [-0.173081011, 0.0, 0.0, 0.0], [0.131512389, 0.0, 0.0, 0.0], [0.0151651194, 0.0, 0.0, 0.0], [0.327000022, 15.0, 16.0, 0.0], [-0.159609675, 0.0, 0.0, 0.0], [0.358500004, 17.0, 18.0, 0.0], [0.404500008, 19.0, 20.0, 0.0], [0.875499964, 21.0, 22.0, 0.0], [-0.0974744633, 0.0, 0.0, 0.0], [-0.239792824, 0.0, 0.0, 0.0], [0.0901365876, 0.0, 0.0, 0.0], [0.481236547, 0.0, 0.0, 0.0], [-0.194690526, 0.0, 0.0, 0.0], [0.0394250639, 0.0, 0.0, 0.0], [-0.0351009592, 0.0, 0.0, 0.0], [0.187880188, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 15, 16, 11, 17, 18, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 4, 10, 2, 5, 12, 6, 13, 14])
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
    function_dict = np.array([[0.342999995, 1.0, 2.0, 0.0], [0.5, 3.0, 4.0, 1.0], [0.364499986, 5.0, 6.0, 0.0], [0.301999986, 7.0, 8.0, 0.0], [0.277999997, 9.0, 10.0, 0.0], [0.5, 11.0, 12.0, 1.0], [0.5, 13.0, 14.0, 1.0], [0.173081025, 0.0, 0.0, 0.0], [-0.131512389, 0.0, 0.0, 0.0], [-0.0151649583, 0.0, 0.0, 0.0], [0.327000022, 15.0, 16.0, 0.0], [0.15960969, 0.0, 0.0, 0.0], [0.358500004, 17.0, 18.0, 0.0], [0.404500008, 19.0, 20.0, 0.0], [0.875499964, 21.0, 22.0, 0.0], [0.0974744782, 0.0, 0.0, 0.0], [0.239792854, 0.0, 0.0, 0.0], [-0.0901365578, 0.0, 0.0, 0.0], [-0.481236488, 0.0, 0.0, 0.0], [0.194690555, 0.0, 0.0, 0.0], [-0.0394250564, 0.0, 0.0, 0.0], [0.0351009779, 0.0, 0.0, 0.0], [-0.187880144, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 15, 16, 11, 17, 18, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 4, 10, 2, 5, 12, 6, 13, 14])
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
    function_dict = np.array([[0.334500015, 1.0, 2.0, 0.0], [0.271499991, 3.0, 4.0, 0.0], [0.411000013, 5.0, 6.0, 0.0], [-0.170423701, 0.0, 0.0, 0.0], [0.305999994, 7.0, 8.0, 0.0], [0.402999997, 9.0, 10.0, 0.0], [0.436500013, 11.0, 12.0, 0.0], [0.284500003, 13.0, 14.0, 0.0], [0.323000014, 15.0, 16.0, 0.0], [0.380999982, 17.0, 18.0, 0.0], [0.505719125, 0.0, 0.0, 0.0], [0.5, 19.0, 20.0, 1.0], [0.453000009, 21.0, 22.0, 0.0], [0.100991659, 0.0, 0.0, 0.0], [0.0252857786, 0.0, 0.0, 0.0], [-0.191530451, 0.0, 0.0, 0.0], [-0.0414262228, 0.0, 0.0, 0.0], [0.0593776107, 0.0, 0.0, 0.0], [-0.208221808, 0.0, 0.0, 0.0], [-0.0828914717, 0.0, 0.0, 0.0], [-0.232157052, 0.0, 0.0, 0.0], [0.182818875, 0.0, 0.0, 0.0], [-0.00126316922, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_15(xs):
    #Predicts Class 1
    function_dict = np.array([[0.334500015, 1.0, 2.0, 0.0], [0.271499991, 3.0, 4.0, 0.0], [0.411000013, 5.0, 6.0, 0.0], [0.170423806, 0.0, 0.0, 0.0], [0.305999994, 7.0, 8.0, 0.0], [0.402999997, 9.0, 10.0, 0.0], [0.436500013, 11.0, 12.0, 0.0], [0.284500003, 13.0, 14.0, 0.0], [0.323000014, 15.0, 16.0, 0.0], [0.380999982, 17.0, 18.0, 0.0], [-0.505719185, 0.0, 0.0, 0.0], [0.5, 19.0, 20.0, 1.0], [0.453000009, 21.0, 22.0, 0.0], [-0.100991622, 0.0, 0.0, 0.0], [-0.0252857637, 0.0, 0.0, 0.0], [0.191530451, 0.0, 0.0, 0.0], [0.0414262563, 0.0, 0.0, 0.0], [-0.059377566, 0.0, 0.0, 0.0], [0.208221793, 0.0, 0.0, 0.0], [0.082891427, 0.0, 0.0, 0.0], [0.232157037, 0.0, 0.0, 0.0], [-0.182818964, 0.0, 0.0, 0.0], [0.00126316294, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_16(xs):
    #Predicts Class 0
    function_dict = np.array([[0.637500048, 1.0, 2.0, 0.0], [0.628000021, 3.0, 4.0, 0.0], [0.666499972, 5.0, 6.0, 0.0], [0.622500002, 7.0, 8.0, 0.0], [0.636500001, 9.0, 10.0, 0.0], [0.5, 11.0, 12.0, 1.0], [0.670000017, 13.0, 14.0, 0.0], [0.618999958, 15.0, 16.0, 0.0], [0.5, 17.0, 18.0, 1.0], [0.631000042, 19.0, 20.0, 0.0], [-0.00321014668, 0.0, 0.0, 0.0], [0.234376013, 0.0, 0.0, 0.0], [0.664000034, 21.0, 22.0, 0.0], [-0.280183673, 0.0, 0.0, 0.0], [0.817499995, 23.0, 24.0, 0.0], [-0.0144970967, 0.0, 0.0, 0.0], [-0.245916501, 0.0, 0.0, 0.0], [-0.0995204002, 0.0, 0.0, 0.0], [0.200301945, 0.0, 0.0, 0.0], [-0.064135462, 0.0, 0.0, 0.0], [-0.279387772, 0.0, 0.0, 0.0], [0.00761712855, 0.0, 0.0, 0.0], [0.216747612, 0.0, 0.0, 0.0], [0.0274162516, 0.0, 0.0, 0.0], [-0.0361330956, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_17(xs):
    #Predicts Class 1
    function_dict = np.array([[0.637500048, 1.0, 2.0, 0.0], [0.628000021, 3.0, 4.0, 0.0], [0.666499972, 5.0, 6.0, 0.0], [0.622500002, 7.0, 8.0, 0.0], [0.636500001, 9.0, 10.0, 0.0], [0.5, 11.0, 12.0, 1.0], [0.670000017, 13.0, 14.0, 0.0], [0.618999958, 15.0, 16.0, 0.0], [0.5, 17.0, 18.0, 1.0], [0.631000042, 19.0, 20.0, 0.0], [0.00321011432, 0.0, 0.0, 0.0], [-0.234375998, 0.0, 0.0, 0.0], [0.664000034, 21.0, 22.0, 0.0], [0.280183703, 0.0, 0.0, 0.0], [0.817499995, 23.0, 24.0, 0.0], [0.0144970901, 0.0, 0.0, 0.0], [0.245916501, 0.0, 0.0, 0.0], [0.0995204002, 0.0, 0.0, 0.0], [-0.200301886, 0.0, 0.0, 0.0], [0.0641354546, 0.0, 0.0, 0.0], [0.279387772, 0.0, 0.0, 0.0], [-0.00761714345, 0.0, 0.0, 0.0], [-0.216747627, 0.0, 0.0, 0.0], [-0.0274162423, 0.0, 0.0, 0.0], [0.0361331031, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_18(xs):
    #Predicts Class 0
    function_dict = np.array([[0.834499955, 1.0, 2.0, 0.0], [0.830500007, 3.0, 4.0, 0.0], [0.851999998, 5.0, 6.0, 0.0], [0.302999973, 7.0, 8.0, 0.0], [-0.304591358, 0.0, 0.0, 0.0], [0.200403109, 0.0, 0.0, 0.0], [0.858500004, 9.0, 10.0, 0.0], [0.27700001, 11.0, 12.0, 0.0], [0.730499983, 13.0, 14.0, 0.0], [-0.182784274, 0.0, 0.0, 0.0], [0.921499968, 15.0, 16.0, 0.0], [0.0165485702, 0.0, 0.0, 0.0], [-0.183722079, 0.0, 0.0, 0.0], [-0.00765329506, 0.0, 0.0, 0.0], [0.029737385, 0.0, 0.0, 0.0], [0.124029152, 0.0, 0.0, 0.0], [-0.0384407565, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 9, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 10])
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
    function_dict = np.array([[0.834499955, 1.0, 2.0, 0.0], [0.830500007, 3.0, 4.0, 0.0], [0.851999998, 5.0, 6.0, 0.0], [0.302999973, 7.0, 8.0, 0.0], [0.304591358, 0.0, 0.0, 0.0], [-0.200403109, 0.0, 0.0, 0.0], [0.858500004, 9.0, 10.0, 0.0], [0.27700001, 11.0, 12.0, 0.0], [0.730499983, 13.0, 14.0, 0.0], [0.182784259, 0.0, 0.0, 0.0], [0.921499968, 15.0, 16.0, 0.0], [-0.0165487882, 0.0, 0.0, 0.0], [0.183722049, 0.0, 0.0, 0.0], [0.00765329646, 0.0, 0.0, 0.0], [-0.0297373645, 0.0, 0.0, 0.0], [-0.124029145, 0.0, 0.0, 0.0], [0.0384407453, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 9, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 10])
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
    function_dict = np.array([[0.271499991, 1.0, 2.0, 0.0], [-0.15044491, 0.0, 0.0, 0.0], [0.949000001, 3.0, 4.0, 0.0], [0.921499968, 5.0, 6.0, 0.0], [0.15084219, 0.0, 0.0, 0.0], [0.881000042, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [-0.0019589162, 0.0, 0.0, 0.0], [0.171495616, 0.0, 0.0, 0.0], [-0.339252889, 0.0, 0.0, 0.0], [0.14006412, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 7, 8, 9, 10, 4])
    branch_indices = np.array([0, 2, 3, 5, 6])
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
    function_dict = np.array([[0.271499991, 1.0, 2.0, 0.0], [0.150445059, 0.0, 0.0, 0.0], [0.949000001, 3.0, 4.0, 0.0], [0.921499968, 5.0, 6.0, 0.0], [-0.150842175, 0.0, 0.0, 0.0], [0.881000042, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.00195891596, 0.0, 0.0, 0.0], [-0.171495616, 0.0, 0.0, 0.0], [0.339253038, 0.0, 0.0, 0.0], [-0.14006409, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 7, 8, 9, 10, 4])
    branch_indices = np.array([0, 2, 3, 5, 6])
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
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.817499995, 3.0, 4.0, 0.0], [0.944499969, 5.0, 6.0, 0.0], [0.813500047, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.18276985, 0.0, 0.0, 0.0], [-0.0509946458, 0.0, 0.0, 0.0], [0.792999983, 11.0, 12.0, 0.0], [0.211231858, 0.0, 0.0, 0.0], [0.878499985, 13.0, 14.0, 0.0], [0.834499955, 15.0, 16.0, 0.0], [0.00272408896, 0.0, 0.0, 0.0], [-0.147374958, 0.0, 0.0, 0.0], [0.0942609683, 0.0, 0.0, 0.0], [-0.167695299, 0.0, 0.0, 0.0], [-0.435513288, 0.0, 0.0, 0.0], [0.0290595852, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 13, 14, 15, 16, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2])
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
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.817499995, 3.0, 4.0, 0.0], [0.944499969, 5.0, 6.0, 0.0], [0.813500047, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [-0.18276985, 0.0, 0.0, 0.0], [0.0509946458, 0.0, 0.0, 0.0], [0.792999983, 11.0, 12.0, 0.0], [-0.211231843, 0.0, 0.0, 0.0], [0.878499985, 13.0, 14.0, 0.0], [0.834499955, 15.0, 16.0, 0.0], [-0.00272409641, 0.0, 0.0, 0.0], [0.147375017, 0.0, 0.0, 0.0], [-0.0942609534, 0.0, 0.0, 0.0], [0.167695209, 0.0, 0.0, 0.0], [0.435513288, 0.0, 0.0, 0.0], [-0.0290595219, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 13, 14, 15, 16, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2])
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
    function_dict = np.array([[0.271499991, 1.0, 2.0, 0.0], [-0.132214099, 0.0, 0.0, 0.0], [0.305999994, 3.0, 4.0, 0.0], [0.117170021, 0.0, 0.0, 0.0], [0.323000014, 5.0, 6.0, 0.0], [-0.160467893, 0.0, 0.0, 0.0], [0.408500016, 7.0, 8.0, 0.0], [0.0375759043, 0.0, 0.0, 0.0], [-0.00302904402, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 5, 7, 8])
    branch_indices = np.array([0, 2, 4, 6])
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
    function_dict = np.array([[0.271499991, 1.0, 2.0, 0.0], [0.132214263, 0.0, 0.0, 0.0], [0.305999994, 3.0, 4.0, 0.0], [-0.117170036, 0.0, 0.0, 0.0], [0.323000014, 5.0, 6.0, 0.0], [0.160467908, 0.0, 0.0, 0.0], [0.408500016, 7.0, 8.0, 0.0], [-0.0375758857, 0.0, 0.0, 0.0], [0.00302904239, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 5, 7, 8])
    branch_indices = np.array([0, 2, 4, 6])
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
    function_dict = np.array([[0.484499991, 1.0, 2.0, 0.0], [0.475000024, 3.0, 4.0, 0.0], [0.495499998, 5.0, 6.0, 0.0], [0.440500021, 7.0, 8.0, 0.0], [-0.219646215, 0.0, 0.0, 0.0], [0.225387976, 0.0, 0.0, 0.0], [0.506000042, 9.0, 10.0, 0.0], [0.411000013, 11.0, 12.0, 0.0], [0.5, 13.0, 14.0, 1.0], [-0.20001848, 0.0, 0.0, 0.0], [0.514500022, 15.0, 16.0, 0.0], [0.00471154833, 0.0, 0.0, 0.0], [-0.13295123, 0.0, 0.0, 0.0], [0.327504694, 0.0, 0.0, 0.0], [-0.0746534094, 0.0, 0.0, 0.0], [0.166601494, 0.0, 0.0, 0.0], [-3.69921436e-05, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 9, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 10])
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
    function_dict = np.array([[0.484499991, 1.0, 2.0, 0.0], [0.475000024, 3.0, 4.0, 0.0], [0.495499998, 5.0, 6.0, 0.0], [0.440500021, 7.0, 8.0, 0.0], [0.219646186, 0.0, 0.0, 0.0], [-0.22538802, 0.0, 0.0, 0.0], [0.506000042, 9.0, 10.0, 0.0], [0.411000013, 11.0, 12.0, 0.0], [0.5, 13.0, 14.0, 1.0], [0.20001848, 0.0, 0.0, 0.0], [0.514500022, 15.0, 16.0, 0.0], [-0.00471155858, 0.0, 0.0, 0.0], [0.13295123, 0.0, 0.0, 0.0], [-0.327504665, 0.0, 0.0, 0.0], [0.074653402, 0.0, 0.0, 0.0], [-0.166601509, 0.0, 0.0, 0.0], [3.69834306e-05, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 9, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 10])
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
    function_dict = np.array([[0.578500032, 1.0, 2.0, 0.0], [0.514500022, 3.0, 4.0, 0.0], [0.595000029, 5.0, 6.0, 0.0], [0.510499954, 7.0, 8.0, 0.0], [0.537, 9.0, 10.0, 0.0], [0.590499997, 11.0, 12.0, 0.0], [0.5995, 13.0, 14.0, 0.0], [0.507500052, 15.0, 16.0, 0.0], [0.256390482, 0.0, 0.0, 0.0], [-0.219363406, 0.0, 0.0, 0.0], [0.5, 17.0, 18.0, 1.0], [0.582000017, 19.0, 20.0, 0.0], [0.254951388, 0.0, 0.0, 0.0], [-0.153673619, 0.0, 0.0, 0.0], [0.605499983, 21.0, 22.0, 0.0], [-0.00680126855, 0.0, 0.0, 0.0], [-0.217189103, 0.0, 0.0, 0.0], [0.0507153869, 0.0, 0.0, 0.0], [-0.115548782, 0.0, 0.0, 0.0], [0.163639858, 0.0, 0.0, 0.0], [-0.0760248005, 0.0, 0.0, 0.0], [0.12407542, 0.0, 0.0, 0.0], [0.00359945302, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 9, 17, 18, 19, 20, 12, 13, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2, 5, 11, 6, 14])
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
    function_dict = np.array([[0.578500032, 1.0, 2.0, 0.0], [0.514500022, 3.0, 4.0, 0.0], [0.595000029, 5.0, 6.0, 0.0], [0.510499954, 7.0, 8.0, 0.0], [0.537, 9.0, 10.0, 0.0], [0.590499997, 11.0, 12.0, 0.0], [0.5995, 13.0, 14.0, 0.0], [0.507500052, 15.0, 16.0, 0.0], [-0.256390482, 0.0, 0.0, 0.0], [0.219363421, 0.0, 0.0, 0.0], [0.5, 17.0, 18.0, 1.0], [0.582000017, 19.0, 20.0, 0.0], [-0.254951388, 0.0, 0.0, 0.0], [0.153673574, 0.0, 0.0, 0.0], [0.605499983, 21.0, 22.0, 0.0], [0.00680126669, 0.0, 0.0, 0.0], [0.217189118, 0.0, 0.0, 0.0], [-0.0507153869, 0.0, 0.0, 0.0], [0.115548782, 0.0, 0.0, 0.0], [-0.163639888, 0.0, 0.0, 0.0], [0.0760247782, 0.0, 0.0, 0.0], [-0.12407545, 0.0, 0.0, 0.0], [-0.00359946047, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 9, 17, 18, 19, 20, 12, 13, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2, 5, 11, 6, 14])
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
    function_dict = np.array([[0.302999973, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.327000022, 5.0, 6.0, 0.0], [0.0791082382, 0.0, 0.0, 0.0], [-0.17473343, 0.0, 0.0, 0.0], [0.117939539, 0.0, 0.0, 0.0], [0.334500015, 7.0, 8.0, 0.0], [-0.169748828, 0.0, 0.0, 0.0], [0.345499992, 9.0, 10.0, 0.0], [0.0958376527, 0.0, 0.0, 0.0], [-0.00078664138, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 7, 9, 10])
    branch_indices = np.array([0, 1, 2, 6, 8])
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
    function_dict = np.array([[0.302999973, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.327000022, 5.0, 6.0, 0.0], [-0.0791082755, 0.0, 0.0, 0.0], [0.174733415, 0.0, 0.0, 0.0], [-0.117939547, 0.0, 0.0, 0.0], [0.334500015, 7.0, 8.0, 0.0], [0.169748843, 0.0, 0.0, 0.0], [0.345499992, 9.0, 10.0, 0.0], [-0.0958376899, 0.0, 0.0, 0.0], [0.000786646269, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 7, 9, 10])
    branch_indices = np.array([0, 1, 2, 6, 8])
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
    function_dict = np.array([[0.271499991, 1.0, 2.0, 0.0], [-0.125342667, 0.0, 0.0, 0.0], [0.305999994, 3.0, 4.0, 0.0], [0.105865777, 0.0, 0.0, 0.0], [0.351999998, 5.0, 6.0, 0.0], [0.345499992, 7.0, 8.0, 0.0], [0.355499983, 9.0, 10.0, 0.0], [-0.0304206628, 0.0, 0.0, 0.0], [-0.194859803, 0.0, 0.0, 0.0], [0.198387668, 0.0, 0.0, 0.0], [-0.000531003461, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 7, 8, 9, 10])
    branch_indices = np.array([0, 2, 4, 5, 6])
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
    function_dict = np.array([[0.271499991, 1.0, 2.0, 0.0], [0.125342831, 0.0, 0.0, 0.0], [0.305999994, 3.0, 4.0, 0.0], [-0.105865836, 0.0, 0.0, 0.0], [0.351999998, 5.0, 6.0, 0.0], [0.345499992, 7.0, 8.0, 0.0], [0.355499983, 9.0, 10.0, 0.0], [0.0304206703, 0.0, 0.0, 0.0], [0.194859773, 0.0, 0.0, 0.0], [-0.198387712, 0.0, 0.0, 0.0], [0.00053099182, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 7, 8, 9, 10])
    branch_indices = np.array([0, 2, 4, 5, 6])
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
    function_dict = np.array([[0.37349999, 1.0, 2.0, 0.0], [0.364499986, 3.0, 4.0, 0.0], [0.379500002, 5.0, 6.0, 0.0], [0.358500004, 7.0, 8.0, 0.0], [-0.225823522, 0.0, 0.0, 0.0], [0.244624332, 0.0, 0.0, 0.0], [0.402999997, 9.0, 10.0, 0.0], [0.355499983, 11.0, 12.0, 0.0], [0.199132755, 0.0, 0.0, 0.0], [0.380999982, 13.0, 14.0, 0.0], [0.408500016, 15.0, 16.0, 0.0], [-0.00850299094, 0.0, 0.0, 0.0], [-0.1947653, 0.0, 0.0, 0.0], [0.0262615103, 0.0, 0.0, 0.0], [-0.185385302, 0.0, 0.0, 0.0], [0.293559462, 0.0, 0.0, 0.0], [-0.00142975268, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 9, 10])
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
    function_dict = np.array([[0.37349999, 1.0, 2.0, 0.0], [0.364499986, 3.0, 4.0, 0.0], [0.379500002, 5.0, 6.0, 0.0], [0.358500004, 7.0, 8.0, 0.0], [0.225823551, 0.0, 0.0, 0.0], [-0.244624272, 0.0, 0.0, 0.0], [0.402999997, 9.0, 10.0, 0.0], [0.355499983, 11.0, 12.0, 0.0], [-0.199132726, 0.0, 0.0, 0.0], [0.380999982, 13.0, 14.0, 0.0], [0.408500016, 15.0, 16.0, 0.0], [0.00850300677, 0.0, 0.0, 0.0], [0.194765314, 0.0, 0.0, 0.0], [-0.0262614693, 0.0, 0.0, 0.0], [0.185385257, 0.0, 0.0, 0.0], [-0.293559462, 0.0, 0.0, 0.0], [0.00142975256, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 9, 10])
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
    function_dict = np.array([[0.553499997, 1.0, 2.0, 0.0], [0.548500001, 3.0, 4.0, 0.0], [0.56400001, 5.0, 6.0, 0.0], [0.54400003, 7.0, 8.0, 0.0], [-0.187312797, 0.0, 0.0, 0.0], [0.555500031, 9.0, 10.0, 0.0], [0.568499982, 11.0, 12.0, 0.0], [0.514500022, 13.0, 14.0, 0.0], [0.103108212, 0.0, 0.0, 0.0], [0.0169796012, 0.0, 0.0, 0.0], [0.157883897, 0.0, 0.0, 0.0], [-0.183649242, 0.0, 0.0, 0.0], [0.572000027, 15.0, 16.0, 0.0], [-0.00468675559, 0.0, 0.0, 0.0], [-0.128893554, 0.0, 0.0, 0.0], [0.127130523, 0.0, 0.0, 0.0], [0.00428536721, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 4, 9, 10, 11, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 5, 6, 12])
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
    function_dict = np.array([[0.553499997, 1.0, 2.0, 0.0], [0.548500001, 3.0, 4.0, 0.0], [0.56400001, 5.0, 6.0, 0.0], [0.54400003, 7.0, 8.0, 0.0], [0.187312797, 0.0, 0.0, 0.0], [0.555500031, 9.0, 10.0, 0.0], [0.568499982, 11.0, 12.0, 0.0], [0.514500022, 13.0, 14.0, 0.0], [-0.103108242, 0.0, 0.0, 0.0], [-0.0169796068, 0.0, 0.0, 0.0], [-0.157883912, 0.0, 0.0, 0.0], [0.183649242, 0.0, 0.0, 0.0], [0.572000027, 15.0, 16.0, 0.0], [0.00468675466, 0.0, 0.0, 0.0], [0.128893569, 0.0, 0.0, 0.0], [-0.127130553, 0.0, 0.0, 0.0], [-0.0042853686, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 4, 9, 10, 11, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 5, 6, 12])
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
    function_dict = np.array([[0.5, 1.0, 2.0, 1.0], [0.423500001, 3.0, 4.0, 0.0], [0.792999983, 5.0, 6.0, 0.0], [0.340499997, 7.0, 8.0, 0.0], [0.475499988, 9.0, 10.0, 0.0], [0.774500012, 11.0, 12.0, 0.0], [0.834499955, 13.0, 14.0, 0.0], [0.323000014, 15.0, 16.0, 0.0], [-0.158144891, 0.0, 0.0, 0.0], [0.230419427, 0.0, 0.0, 0.0], [0.537, 17.0, 18.0, 0.0], [0.754000008, 19.0, 20.0, 0.0], [0.786499977, 21.0, 22.0, 0.0], [0.813500047, 23.0, 24.0, 0.0], [0.862499952, 25.0, 26.0, 0.0], [-0.0230925996, 0.0, 0.0, 0.0], [0.0733489469, 0.0, 0.0, 0.0], [-0.12757884, 0.0, 0.0, 0.0], [0.0209571831, 0.0, 0.0, 0.0], [-0.00290548941, 0.0, 0.0, 0.0], [-0.308502316, 0.0, 0.0, 0.0], [0.228975028, 0.0, 0.0, 0.0], [0.063510932, 0.0, 0.0, 0.0], [-0.270304024, 0.0, 0.0, 0.0], [-0.106651559, 0.0, 0.0, 0.0], [0.0993909538, 0.0, 0.0, 0.0], [-0.047195632, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 9, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2, 5, 11, 12, 6, 13, 14])
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
    function_dict = np.array([[0.5, 1.0, 2.0, 1.0], [0.423500001, 3.0, 4.0, 0.0], [0.792999983, 5.0, 6.0, 0.0], [0.340499997, 7.0, 8.0, 0.0], [0.475499988, 9.0, 10.0, 0.0], [0.774500012, 11.0, 12.0, 0.0], [0.834499955, 13.0, 14.0, 0.0], [0.323000014, 15.0, 16.0, 0.0], [0.158144861, 0.0, 0.0, 0.0], [-0.230419442, 0.0, 0.0, 0.0], [0.537, 17.0, 18.0, 0.0], [0.754000008, 19.0, 20.0, 0.0], [0.786499977, 21.0, 22.0, 0.0], [0.813500047, 23.0, 24.0, 0.0], [0.862499952, 25.0, 26.0, 0.0], [0.0230926126, 0.0, 0.0, 0.0], [-0.0733489841, 0.0, 0.0, 0.0], [0.127578825, 0.0, 0.0, 0.0], [-0.0209571663, 0.0, 0.0, 0.0], [0.00290548755, 0.0, 0.0, 0.0], [0.308502287, 0.0, 0.0, 0.0], [-0.228975043, 0.0, 0.0, 0.0], [-0.0635109469, 0.0, 0.0, 0.0], [0.270304054, 0.0, 0.0, 0.0], [0.106651552, 0.0, 0.0, 0.0], [-0.0993909463, 0.0, 0.0, 0.0], [0.0471956879, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 9, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2, 5, 11, 12, 6, 13, 14])
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
    function_dict = np.array([[0.730499983, 1.0, 2.0, 0.0], [0.725000024, 3.0, 4.0, 0.0], [0.740499973, 5.0, 6.0, 0.0], [0.718999982, 7.0, 8.0, 0.0], [-0.252217799, 0.0, 0.0, 0.0], [0.220352426, 0.0, 0.0, 0.0], [0.744500041, 9.0, 10.0, 0.0], [0.713, 11.0, 12.0, 0.0], [0.238575816, 0.0, 0.0, 0.0], [0.5, 13.0, 14.0, 1.0], [0.754000008, 15.0, 16.0, 0.0], [-0.00201701047, 0.0, 0.0, 0.0], [-0.265318513, 0.0, 0.0, 0.0], [-0.299496651, 0.0, 0.0, 0.0], [-0.0184674244, 0.0, 0.0, 0.0], [0.227032498, 0.0, 0.0, 0.0], [-0.00988405664, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 9, 10])
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
    function_dict = np.array([[0.730499983, 1.0, 2.0, 0.0], [0.725000024, 3.0, 4.0, 0.0], [0.740499973, 5.0, 6.0, 0.0], [0.718999982, 7.0, 8.0, 0.0], [0.252217799, 0.0, 0.0, 0.0], [-0.220352456, 0.0, 0.0, 0.0], [0.744500041, 9.0, 10.0, 0.0], [0.713, 11.0, 12.0, 0.0], [-0.238575831, 0.0, 0.0, 0.0], [0.5, 13.0, 14.0, 1.0], [0.754000008, 15.0, 16.0, 0.0], [0.00201700162, 0.0, 0.0, 0.0], [0.265318513, 0.0, 0.0, 0.0], [0.299496651, 0.0, 0.0, 0.0], [0.0184674244, 0.0, 0.0, 0.0], [-0.227032498, 0.0, 0.0, 0.0], [0.00988405198, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 9, 10])
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
    function_dict = np.array([[0.701499999, 1.0, 2.0, 0.0], [0.688500047, 3.0, 4.0, 0.0], [0.710500002, 5.0, 6.0, 0.0], [0.644500017, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.708500028, 11.0, 12.0, 0.0], [0.715499997, 13.0, 14.0, 0.0], [0.642500043, 15.0, 16.0, 0.0], [0.6505, 17.0, 18.0, 0.0], [0.694000006, 19.0, 20.0, 0.0], [0.696500003, 21.0, 22.0, 0.0], [0.214812219, 0.0, 0.0, 0.0], [0.0663102865, 0.0, 0.0, 0.0], [-0.105170853, 0.0, 0.0, 0.0], [0.725000024, 23.0, 24.0, 0.0], [-0.00421478553, 0.0, 0.0, 0.0], [-0.231597215, 0.0, 0.0, 0.0], [0.243691459, 0.0, 0.0, 0.0], [0.0276330095, 0.0, 0.0, 0.0], [-2.32679158e-05, 0.0, 0.0, 0.0], [-0.0576582365, 0.0, 0.0, 0.0], [-0.326401204, 0.0, 0.0, 0.0], [-0.0697726458, 0.0, 0.0, 0.0], [0.135386005, 0.0, 0.0, 0.0], [0.00224916195, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 11, 12, 13, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 6, 14])
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
    function_dict = np.array([[0.701499999, 1.0, 2.0, 0.0], [0.688500047, 3.0, 4.0, 0.0], [0.710500002, 5.0, 6.0, 0.0], [0.644500017, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.708500028, 11.0, 12.0, 0.0], [0.715499997, 13.0, 14.0, 0.0], [0.642500043, 15.0, 16.0, 0.0], [0.6505, 17.0, 18.0, 0.0], [0.694000006, 19.0, 20.0, 0.0], [0.696500003, 21.0, 22.0, 0.0], [-0.214812204, 0.0, 0.0, 0.0], [-0.0663103014, 0.0, 0.0, 0.0], [0.105170824, 0.0, 0.0, 0.0], [0.725000024, 23.0, 24.0, 0.0], [0.00421477808, 0.0, 0.0, 0.0], [0.23159723, 0.0, 0.0, 0.0], [-0.243691429, 0.0, 0.0, 0.0], [-0.0276329927, 0.0, 0.0, 0.0], [2.32854381e-05, 0.0, 0.0, 0.0], [0.0576582551, 0.0, 0.0, 0.0], [0.326401204, 0.0, 0.0, 0.0], [0.0697726458, 0.0, 0.0, 0.0], [-0.135386005, 0.0, 0.0, 0.0], [-0.00224916078, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 11, 12, 13, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 6, 14])
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
    function_dict = np.array([[0.774500012, 1.0, 2.0, 0.0], [0.756500006, 3.0, 4.0, 0.0], [0.781000018, 5.0, 6.0, 0.0], [0.744500041, 7.0, 8.0, 0.0], [0.765499949, 9.0, 10.0, 0.0], [0.189306229, 0.0, 0.0, 0.0], [0.787500024, 11.0, 12.0, 0.0], [0.740499973, 13.0, 14.0, 0.0], [0.754000008, 15.0, 16.0, 0.0], [-0.0696778297, 0.0, 0.0, 0.0], [-0.215541735, 0.0, 0.0, 0.0], [-0.102400564, 0.0, 0.0, 0.0], [0.817499995, 17.0, 18.0, 0.0], [0.000185445722, 0.0, 0.0, 0.0], [-0.11595802, 0.0, 0.0, 0.0], [0.194556713, 0.0, 0.0, 0.0], [0.018677989, 0.0, 0.0, 0.0], [0.0865108594, 0.0, 0.0, 0.0], [-0.0220920704, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 12])
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
    function_dict = np.array([[0.774500012, 1.0, 2.0, 0.0], [0.756500006, 3.0, 4.0, 0.0], [0.781000018, 5.0, 6.0, 0.0], [0.744500041, 7.0, 8.0, 0.0], [0.765499949, 9.0, 10.0, 0.0], [-0.189306214, 0.0, 0.0, 0.0], [0.787500024, 11.0, 12.0, 0.0], [0.740499973, 13.0, 14.0, 0.0], [0.754000008, 15.0, 16.0, 0.0], [0.0696778297, 0.0, 0.0, 0.0], [0.21554175, 0.0, 0.0, 0.0], [0.102400579, 0.0, 0.0, 0.0], [0.817499995, 17.0, 18.0, 0.0], [-0.000185445417, 0.0, 0.0, 0.0], [0.11595802, 0.0, 0.0, 0.0], [-0.194556713, 0.0, 0.0, 0.0], [-0.018677976, 0.0, 0.0, 0.0], [-0.0865108594, 0.0, 0.0, 0.0], [0.022092063, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 12])
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
    function_dict = np.array([[0.881000042, 1.0, 2.0, 0.0], [0.817499995, 3.0, 4.0, 0.0], [0.921499968, 5.0, 6.0, 0.0], [0.796000004, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.167831689, 0.0, 0.0, 0.0], [0.92750001, 11.0, 12.0, 0.0], [0.791499972, 13.0, 14.0, 0.0], [0.804499984, 15.0, 16.0, 0.0], [0.833000004, 17.0, 18.0, 0.0], [0.834499955, 19.0, 20.0, 0.0], [-0.0948327854, 0.0, 0.0, 0.0], [0.0433253944, 0.0, 0.0, 0.0], [-0.000330465671, 0.0, 0.0, 0.0], [-0.179476991, 0.0, 0.0, 0.0], [0.228690848, 0.0, 0.0, 0.0], [0.0315467827, 0.0, 0.0, 0.0], [0.106742889, 0.0, 0.0, 0.0], [-0.0531372651, 0.0, 0.0, 0.0], [-0.25069499, 0.0, 0.0, 0.0], [-0.020733837, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 6])
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
    function_dict = np.array([[0.881000042, 1.0, 2.0, 0.0], [0.817499995, 3.0, 4.0, 0.0], [0.921499968, 5.0, 6.0, 0.0], [0.796000004, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [-0.167831734, 0.0, 0.0, 0.0], [0.92750001, 11.0, 12.0, 0.0], [0.791499972, 13.0, 14.0, 0.0], [0.804499984, 15.0, 16.0, 0.0], [0.833000004, 17.0, 18.0, 0.0], [0.834499955, 19.0, 20.0, 0.0], [0.094832778, 0.0, 0.0, 0.0], [-0.0433254205, 0.0, 0.0, 0.0], [0.00033047708, 0.0, 0.0, 0.0], [0.179476991, 0.0, 0.0, 0.0], [-0.228690848, 0.0, 0.0, 0.0], [-0.031546779, 0.0, 0.0, 0.0], [-0.106742889, 0.0, 0.0, 0.0], [0.0531373024, 0.0, 0.0, 0.0], [0.25069496, 0.0, 0.0, 0.0], [0.020733837, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 6])
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
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.878499985, 3.0, 4.0, 0.0], [0.944499969, 5.0, 6.0, 0.0], [0.863499999, 7.0, 8.0, 0.0], [-0.082787998, 0.0, 0.0, 0.0], [0.165279672, 0.0, 0.0, 0.0], [-0.0465392135, 0.0, 0.0, 0.0], [0.851999998, 9.0, 10.0, 0.0], [0.160874471, 0.0, 0.0, 0.0], [0.000220035377, 0.0, 0.0, 0.0], [-0.0883305371, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 8, 4, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 2])
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
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.878499985, 3.0, 4.0, 0.0], [0.944499969, 5.0, 6.0, 0.0], [0.863499999, 7.0, 8.0, 0.0], [0.0827879012, 0.0, 0.0, 0.0], [-0.165279642, 0.0, 0.0, 0.0], [0.0465392284, 0.0, 0.0, 0.0], [0.851999998, 9.0, 10.0, 0.0], [-0.160874471, 0.0, 0.0, 0.0], [-0.000220036163, 0.0, 0.0, 0.0], [0.0883305222, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 8, 4, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 2])
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
    function_dict = np.array([[0.484499991, 1.0, 2.0, 0.0], [0.453000009, 3.0, 4.0, 0.0], [0.488499999, 5.0, 6.0, 0.0], [0.440500021, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.278720081, 0.0, 0.0, 0.0], [0.49150002, 11.0, 12.0, 0.0], [0.411000013, 13.0, 14.0, 0.0], [0.446500003, 15.0, 16.0, 0.0], [-0.0250906982, 0.0, 0.0, 0.0], [-0.194070548, 0.0, 0.0, 0.0], [-0.17682077, 0.0, 0.0, 0.0], [0.495499998, 17.0, 18.0, 0.0], [0.0151886092, 0.0, 0.0, 0.0], [-0.105596401, 0.0, 0.0, 0.0], [0.134817362, 0.0, 0.0, 0.0], [0.009107586, 0.0, 0.0, 0.0], [0.165604398, 0.0, 0.0, 0.0], [0.0017498394, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 12])
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
    function_dict = np.array([[0.484499991, 1.0, 2.0, 0.0], [0.453000009, 3.0, 4.0, 0.0], [0.488499999, 5.0, 6.0, 0.0], [0.440500021, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [-0.278720081, 0.0, 0.0, 0.0], [0.49150002, 11.0, 12.0, 0.0], [0.411000013, 13.0, 14.0, 0.0], [0.446500003, 15.0, 16.0, 0.0], [0.0250907429, 0.0, 0.0, 0.0], [0.194070563, 0.0, 0.0, 0.0], [0.17682077, 0.0, 0.0, 0.0], [0.495499998, 17.0, 18.0, 0.0], [-0.0151886242, 0.0, 0.0, 0.0], [0.10559646, 0.0, 0.0, 0.0], [-0.134817377, 0.0, 0.0, 0.0], [-0.00910760928, 0.0, 0.0, 0.0], [-0.165604383, 0.0, 0.0, 0.0], [-0.00174983684, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 12])
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
    function_dict = np.array([[0.834499955, 1.0, 2.0, 0.0], [0.82249999, 3.0, 4.0, 0.0], [0.851999998, 5.0, 6.0, 0.0], [0.796000004, 7.0, 8.0, 0.0], [-0.131207868, 0.0, 0.0, 0.0], [0.161865905, 0.0, 0.0, 0.0], [0.858500004, 9.0, 10.0, 0.0], [0.791499972, 11.0, 12.0, 0.0], [0.5, 13.0, 14.0, 1.0], [-0.112380289, 0.0, 0.0, 0.0], [0.862499952, 15.0, 16.0, 0.0], [-0.000855398597, 0.0, 0.0, 0.0], [-0.118078053, 0.0, 0.0, 0.0], [0.19539614, 0.0, 0.0, 0.0], [-0.042715501, 0.0, 0.0, 0.0], [0.166753292, 0.0, 0.0, 0.0], [0.000430014625, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 9, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 10])
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
    function_dict = np.array([[0.834499955, 1.0, 2.0, 0.0], [0.82249999, 3.0, 4.0, 0.0], [0.851999998, 5.0, 6.0, 0.0], [0.796000004, 7.0, 8.0, 0.0], [0.131207898, 0.0, 0.0, 0.0], [-0.161865905, 0.0, 0.0, 0.0], [0.858500004, 9.0, 10.0, 0.0], [0.791499972, 11.0, 12.0, 0.0], [0.5, 13.0, 14.0, 1.0], [0.112380289, 0.0, 0.0, 0.0], [0.862499952, 15.0, 16.0, 0.0], [0.00085539819, 0.0, 0.0, 0.0], [0.118078053, 0.0, 0.0, 0.0], [-0.19539614, 0.0, 0.0, 0.0], [0.0427155085, 0.0, 0.0, 0.0], [-0.166753307, 0.0, 0.0, 0.0], [-0.000430030632, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 5, 9, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 6, 10])
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
    function_dict = np.array([[0.701499999, 1.0, 2.0, 0.0], [0.688500047, 3.0, 4.0, 0.0], [0.713, 5.0, 6.0, 0.0], [0.674000025, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.140112609, 0.0, 0.0, 0.0], [0.718999982, 11.0, 12.0, 0.0], [0.666499972, 13.0, 14.0, 0.0], [0.5, 15.0, 16.0, 1.0], [0.694000006, 17.0, 18.0, 0.0], [0.696500003, 19.0, 20.0, 0.0], [-0.175885737, 0.0, 0.0, 0.0], [0.725000024, 21.0, 22.0, 0.0], [0.00172024185, 0.0, 0.0, 0.0], [-0.151437283, 0.0, 0.0, 0.0], [0.14712508, 0.0, 0.0, 0.0], [-0.0114328144, 0.0, 0.0, 0.0], [-0.000301439315, 0.0, 0.0, 0.0], [-0.0413244106, 0.0, 0.0, 0.0], [-0.239578471, 0.0, 0.0, 0.0], [-0.0458927974, 0.0, 0.0, 0.0], [0.172953025, 0.0, 0.0, 0.0], [0.0053872643, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_55(xs):
    #Predicts Class 1
    function_dict = np.array([[0.701499999, 1.0, 2.0, 0.0], [0.688500047, 3.0, 4.0, 0.0], [0.713, 5.0, 6.0, 0.0], [0.674000025, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [-0.140112579, 0.0, 0.0, 0.0], [0.718999982, 11.0, 12.0, 0.0], [0.666499972, 13.0, 14.0, 0.0], [0.5, 15.0, 16.0, 1.0], [0.694000006, 17.0, 18.0, 0.0], [0.696500003, 19.0, 20.0, 0.0], [0.175885767, 0.0, 0.0, 0.0], [0.725000024, 21.0, 22.0, 0.0], [-0.00172023359, 0.0, 0.0, 0.0], [0.151437283, 0.0, 0.0, 0.0], [-0.147125065, 0.0, 0.0, 0.0], [0.0114328638, 0.0, 0.0, 0.0], [0.000301456865, 0.0, 0.0, 0.0], [0.0413244106, 0.0, 0.0, 0.0], [0.239578456, 0.0, 0.0, 0.0], [0.0458927862, 0.0, 0.0, 0.0], [-0.172952995, 0.0, 0.0, 0.0], [-0.00538726337, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_56(xs):
    #Predicts Class 0
    function_dict = np.array([[0.342999995, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.377499998, 5.0, 6.0, 0.0], [0.0761682019, 0.0, 0.0, 0.0], [0.5, 7.0, 8.0, 1.0], [0.5, 9.0, 10.0, 1.0], [0.402999997, 11.0, 12.0, 0.0], [0.310499996, 13.0, 14.0, 0.0], [0.327000022, 15.0, 16.0, 0.0], [-0.140852273, 0.0, 0.0, 0.0], [0.369499981, 17.0, 18.0, 0.0], [-0.129977167, 0.0, 0.0, 0.0], [0.408500016, 19.0, 20.0, 0.0], [0.0525850095, 0.0, 0.0, 0.0], [-0.00552148977, 0.0, 0.0, 0.0], [-0.0261396244, 0.0, 0.0, 0.0], [-0.204221576, 0.0, 0.0, 0.0], [0.0506600626, 0.0, 0.0, 0.0], [0.209362537, 0.0, 0.0, 0.0], [0.192930415, 0.0, 0.0, 0.0], [-0.000894203433, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 9, 17, 18, 11, 19, 20])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 10, 6, 12])
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
    function_dict = np.array([[0.342999995, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.377499998, 5.0, 6.0, 0.0], [-0.0761683509, 0.0, 0.0, 0.0], [0.5, 7.0, 8.0, 1.0], [0.5, 9.0, 10.0, 1.0], [0.402999997, 11.0, 12.0, 0.0], [0.310499996, 13.0, 14.0, 0.0], [0.327000022, 15.0, 16.0, 0.0], [0.140852258, 0.0, 0.0, 0.0], [0.369499981, 17.0, 18.0, 0.0], [0.129977182, 0.0, 0.0, 0.0], [0.408500016, 19.0, 20.0, 0.0], [-0.0525849797, 0.0, 0.0, 0.0], [0.00552148325, 0.0, 0.0, 0.0], [0.0261396393, 0.0, 0.0, 0.0], [0.204221576, 0.0, 0.0, 0.0], [-0.0506600626, 0.0, 0.0, 0.0], [-0.209362522, 0.0, 0.0, 0.0], [-0.19293043, 0.0, 0.0, 0.0], [0.000894203957, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 9, 17, 18, 11, 19, 20])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 10, 6, 12])
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
    function_dict = np.array([[0.302999973, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.327000022, 5.0, 6.0, 0.0], [0.0511958934, 0.0, 0.0, 0.0], [-0.164810762, 0.0, 0.0, 0.0], [0.0742640272, 0.0, 0.0, 0.0], [0.637500048, 7.0, 8.0, 0.0], [0.628000021, 9.0, 10.0, 0.0], [0.638499975, 11.0, 12.0, 0.0], [-0.00241235457, 0.0, 0.0, 0.0], [-0.120412409, 0.0, 0.0, 0.0], [0.191024706, 0.0, 0.0, 0.0], [0.00446546637, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 2, 6, 7, 8])
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
    function_dict = np.array([[0.302999973, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.327000022, 5.0, 6.0, 0.0], [-0.0511960015, 0.0, 0.0, 0.0], [0.164810792, 0.0, 0.0, 0.0], [-0.0742640048, 0.0, 0.0, 0.0], [0.637500048, 7.0, 8.0, 0.0], [0.628000021, 9.0, 10.0, 0.0], [0.638499975, 11.0, 12.0, 0.0], [0.00241236668, 0.0, 0.0, 0.0], [0.120412387, 0.0, 0.0, 0.0], [-0.191024736, 0.0, 0.0, 0.0], [-0.00446546916, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 2, 6, 7, 8])
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
    function_dict = np.array([[0.323000014, 1.0, 2.0, 0.0], [0.305999994, 3.0, 4.0, 0.0], [0.345499992, 5.0, 6.0, 0.0], [0.286000013, 7.0, 8.0, 0.0], [-0.155850023, 0.0, 0.0, 0.0], [0.334500015, 9.0, 10.0, 0.0], [0.37349999, 11.0, 12.0, 0.0], [-0.0291937534, 0.0, 0.0, 0.0], [0.0621752515, 0.0, 0.0, 0.0], [-0.00632596808, 0.0, 0.0, 0.0], [0.116081759, 0.0, 0.0, 0.0], [0.364499986, 13.0, 14.0, 0.0], [0.405000001, 15.0, 16.0, 0.0], [0.0118814213, 0.0, 0.0, 0.0], [-0.197384775, 0.0, 0.0, 0.0], [0.0972087011, 0.0, 0.0, 0.0], [-0.000900096318, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_61(xs):
    #Predicts Class 1
    function_dict = np.array([[0.323000014, 1.0, 2.0, 0.0], [0.305999994, 3.0, 4.0, 0.0], [0.345499992, 5.0, 6.0, 0.0], [0.286000013, 7.0, 8.0, 0.0], [0.155850023, 0.0, 0.0, 0.0], [0.334500015, 9.0, 10.0, 0.0], [0.37349999, 11.0, 12.0, 0.0], [0.029193731, 0.0, 0.0, 0.0], [-0.0621752888, 0.0, 0.0, 0.0], [0.0063259881, 0.0, 0.0, 0.0], [-0.116081752, 0.0, 0.0, 0.0], [0.364499986, 13.0, 14.0, 0.0], [0.405000001, 15.0, 16.0, 0.0], [-0.0118814232, 0.0, 0.0, 0.0], [0.197384745, 0.0, 0.0, 0.0], [-0.0972087607, 0.0, 0.0, 0.0], [0.000900094397, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_62(xs):
    #Predicts Class 0
    function_dict = np.array([[0.484499991, 1.0, 2.0, 0.0], [0.475000024, 3.0, 4.0, 0.0], [0.488499999, 5.0, 6.0, 0.0], [0.5, 7.0, 8.0, 1.0], [-0.163685113, 0.0, 0.0, 0.0], [0.176589116, 0.0, 0.0, 0.0], [0.506000042, 9.0, 10.0, 0.0], [0.423500001, 11.0, 12.0, 0.0], [0.411000013, 13.0, 14.0, 0.0], [0.495499998, 15.0, 16.0, 0.0], [0.514500022, 17.0, 18.0, 0.0], [-0.0598787256, 0.0, 0.0, 0.0], [0.134415805, 0.0, 0.0, 0.0], [0.0115840416, 0.0, 0.0, 0.0], [-0.141890839, 0.0, 0.0, 0.0], [0.00206579361, 0.0, 0.0, 0.0], [-0.155376807, 0.0, 0.0, 0.0], [0.129622385, 0.0, 0.0, 0.0], [0.000387454726, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_63(xs):
    #Predicts Class 1
    function_dict = np.array([[0.484499991, 1.0, 2.0, 0.0], [0.475000024, 3.0, 4.0, 0.0], [0.488499999, 5.0, 6.0, 0.0], [0.5, 7.0, 8.0, 1.0], [0.163685158, 0.0, 0.0, 0.0], [-0.176589102, 0.0, 0.0, 0.0], [0.506000042, 9.0, 10.0, 0.0], [0.423500001, 11.0, 12.0, 0.0], [0.411000013, 13.0, 14.0, 0.0], [0.495499998, 15.0, 16.0, 0.0], [0.514500022, 17.0, 18.0, 0.0], [0.0598787628, 0.0, 0.0, 0.0], [-0.13441582, 0.0, 0.0, 0.0], [-0.0115840286, 0.0, 0.0, 0.0], [0.141890824, 0.0, 0.0, 0.0], [-0.00206578383, 0.0, 0.0, 0.0], [0.155376807, 0.0, 0.0, 0.0], [-0.12962237, 0.0, 0.0, 0.0], [-0.000387458538, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_64(xs):
    #Predicts Class 0
    function_dict = np.array([[0.579999983, 1.0, 2.0, 0.0], [0.572000027, 3.0, 4.0, 0.0], [0.595000029, 5.0, 6.0, 0.0], [0.568499982, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.590499997, 11.0, 12.0, 0.0], [0.601999998, 13.0, 14.0, 0.0], [0.5, 15.0, 16.0, 1.0], [0.083736524, 0.0, 0.0, 0.0], [0.00361472019, 0.0, 0.0, 0.0], [-0.194635257, 0.0, 0.0, 0.0], [0.0286306925, 0.0, 0.0, 0.0], [0.166361168, 0.0, 0.0, 0.0], [0.5, 17.0, 18.0, 1.0], [0.605499983, 19.0, 20.0, 0.0], [0.0131500317, 0.0, 0.0, 0.0], [-0.0323940367, 0.0, 0.0, 0.0], [0.012331144, 0.0, 0.0, 0.0], [-0.195875779, 0.0, 0.0, 0.0], [0.20036988, 0.0, 0.0, 0.0], [0.00168306788, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 9, 10, 11, 12, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 4, 2, 5, 6, 13, 14])
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
    function_dict = np.array([[0.579999983, 1.0, 2.0, 0.0], [0.572000027, 3.0, 4.0, 0.0], [0.595000029, 5.0, 6.0, 0.0], [0.568499982, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.590499997, 11.0, 12.0, 0.0], [0.601999998, 13.0, 14.0, 0.0], [0.5, 15.0, 16.0, 1.0], [-0.0837364867, 0.0, 0.0, 0.0], [-0.00361473998, 0.0, 0.0, 0.0], [0.194635257, 0.0, 0.0, 0.0], [-0.0286307205, 0.0, 0.0, 0.0], [-0.166361168, 0.0, 0.0, 0.0], [0.5, 17.0, 18.0, 1.0], [0.605499983, 19.0, 20.0, 0.0], [-0.0131500559, 0.0, 0.0, 0.0], [0.0323940255, 0.0, 0.0, 0.0], [-0.0123311598, 0.0, 0.0, 0.0], [0.195875779, 0.0, 0.0, 0.0], [-0.200369865, 0.0, 0.0, 0.0], [-0.00168306951, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 9, 10, 11, 12, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 4, 2, 5, 6, 13, 14])
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
    function_dict = np.array([[0.823500037, 1.0, 2.0, 0.0], [0.817499995, 3.0, 4.0, 0.0], [0.833000004, 5.0, 6.0, 0.0], [0.813500047, 7.0, 8.0, 0.0], [-0.190522388, 0.0, 0.0, 0.0], [0.158102542, 0.0, 0.0, 0.0], [0.835999966, 9.0, 10.0, 0.0], [0.804499984, 11.0, 12.0, 0.0], [0.176367432, 0.0, 0.0, 0.0], [-0.125145301, 0.0, 0.0, 0.0], [0.921499968, 13.0, 14.0, 0.0], [0.000141520213, 0.0, 0.0, 0.0], [-0.148188457, 0.0, 0.0, 0.0], [0.0429346785, 0.0, 0.0, 0.0], [-0.0315725431, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 10])
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
    function_dict = np.array([[0.823500037, 1.0, 2.0, 0.0], [0.817499995, 3.0, 4.0, 0.0], [0.833000004, 5.0, 6.0, 0.0], [0.813500047, 7.0, 8.0, 0.0], [0.190522477, 0.0, 0.0, 0.0], [-0.158102542, 0.0, 0.0, 0.0], [0.835999966, 9.0, 10.0, 0.0], [0.804499984, 11.0, 12.0, 0.0], [-0.176367417, 0.0, 0.0, 0.0], [0.125145286, 0.0, 0.0, 0.0], [0.921499968, 13.0, 14.0, 0.0], [-0.000141521101, 0.0, 0.0, 0.0], [0.148188427, 0.0, 0.0, 0.0], [-0.0429347083, 0.0, 0.0, 0.0], [0.0315725133, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 10])
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
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.862499952, 3.0, 4.0, 0.0], [0.944499969, 5.0, 6.0, 0.0], [0.834499955, 7.0, 8.0, 0.0], [0.881000042, 9.0, 10.0, 0.0], [0.145052716, 0.0, 0.0, 0.0], [-0.026339341, 0.0, 0.0, 0.0], [0.82249999, 11.0, 12.0, 0.0], [0.851999998, 13.0, 14.0, 0.0], [-0.108180344, 0.0, 0.0, 0.0], [0.00868193246, 0.0, 0.0, 0.0], [0.000493024127, 0.0, 0.0, 0.0], [-0.103184812, 0.0, 0.0, 0.0], [0.142629817, 0.0, 0.0, 0.0], [0.00624672929, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2])
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
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.862499952, 3.0, 4.0, 0.0], [0.944499969, 5.0, 6.0, 0.0], [0.834499955, 7.0, 8.0, 0.0], [0.881000042, 9.0, 10.0, 0.0], [-0.145052701, 0.0, 0.0, 0.0], [0.026339354, 0.0, 0.0, 0.0], [0.82249999, 11.0, 12.0, 0.0], [0.851999998, 13.0, 14.0, 0.0], [0.108180299, 0.0, 0.0, 0.0], [-0.00868199021, 0.0, 0.0, 0.0], [-0.00049302564, 0.0, 0.0, 0.0], [0.103184812, 0.0, 0.0, 0.0], [-0.142629802, 0.0, 0.0, 0.0], [-0.00624671718, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2])
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
    function_dict = np.array([[0.949000001, 1.0, 2.0, 0.0], [0.921499968, 3.0, 4.0, 0.0], [0.144388631, 0.0, 0.0, 0.0], [0.881000042, 5.0, 6.0, 0.0], [-0.15227598, 0.0, 0.0, 0.0], [0.862499952, 7.0, 8.0, 0.0], [0.148422331, 0.0, 0.0, 0.0], [0.000241702874, 0.0, 0.0, 0.0], [-0.0642653778, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 6, 4, 2])
    branch_indices = np.array([0, 1, 3, 5])
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
    function_dict = np.array([[0.949000001, 1.0, 2.0, 0.0], [0.921499968, 3.0, 4.0, 0.0], [-0.144388646, 0.0, 0.0, 0.0], [0.881000042, 5.0, 6.0, 0.0], [0.152276099, 0.0, 0.0, 0.0], [0.862499952, 7.0, 8.0, 0.0], [-0.148422331, 0.0, 0.0, 0.0], [-0.00024171012, 0.0, 0.0, 0.0], [0.0642653629, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 6, 4, 2])
    branch_indices = np.array([0, 1, 3, 5])
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
    function_dict = np.array([[0.730499983, 1.0, 2.0, 0.0], [0.725000024, 3.0, 4.0, 0.0], [0.740499973, 5.0, 6.0, 0.0], [0.718999982, 7.0, 8.0, 0.0], [-0.169022575, 0.0, 0.0, 0.0], [0.191109553, 0.0, 0.0, 0.0], [0.743499994, 9.0, 10.0, 0.0], [0.713, 11.0, 12.0, 0.0], [0.146652982, 0.0, 0.0, 0.0], [-0.119337231, 0.0, 0.0, 0.0], [0.754000008, 13.0, 14.0, 0.0], [-0.000823876297, 0.0, 0.0, 0.0], [-0.122358136, 0.0, 0.0, 0.0], [0.114271946, 0.0, 0.0, 0.0], [0.000597345701, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 10])
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
    function_dict = np.array([[0.730499983, 1.0, 2.0, 0.0], [0.725000024, 3.0, 4.0, 0.0], [0.740499973, 5.0, 6.0, 0.0], [0.718999982, 7.0, 8.0, 0.0], [0.16902259, 0.0, 0.0, 0.0], [-0.191109598, 0.0, 0.0, 0.0], [0.743499994, 9.0, 10.0, 0.0], [0.713, 11.0, 12.0, 0.0], [-0.146652982, 0.0, 0.0, 0.0], [0.119337194, 0.0, 0.0, 0.0], [0.754000008, 13.0, 14.0, 0.0], [0.000823874609, 0.0, 0.0, 0.0], [0.122358128, 0.0, 0.0, 0.0], [-0.114271991, 0.0, 0.0, 0.0], [-0.000597343664, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 10])
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
    function_dict = np.array([[0.774500012, 1.0, 2.0, 0.0], [0.756500006, 3.0, 4.0, 0.0], [0.781000018, 5.0, 6.0, 0.0], [0.744500041, 7.0, 8.0, 0.0], [0.75999999, 9.0, 10.0, 0.0], [0.160039231, 0.0, 0.0, 0.0], [0.790500045, 11.0, 12.0, 0.0], [0.740499973, 13.0, 14.0, 0.0], [0.5, 15.0, 16.0, 1.0], [-0.208053038, 0.0, 0.0, 0.0], [0.766499996, 17.0, 18.0, 0.0], [0.5, 19.0, 20.0, 1.0], [0.792999983, 21.0, 22.0, 0.0], [0.000166229715, 0.0, 0.0, 0.0], [-0.0611774921, 0.0, 0.0, 0.0], [0.18440944, 0.0, 0.0, 0.0], [-0.0292314142, 0.0, 0.0, 0.0], [0.0495505407, 0.0, 0.0, 0.0], [-0.0930214971, 0.0, 0.0, 0.0], [-0.151964873, 0.0, 0.0, 0.0], [0.0392864197, 0.0, 0.0, 0.0], [0.185795218, 0.0, 0.0, 0.0], [0.00541534415, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 17, 18, 5, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 6, 11, 12])
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
    function_dict = np.array([[0.774500012, 1.0, 2.0, 0.0], [0.756500006, 3.0, 4.0, 0.0], [0.781000018, 5.0, 6.0, 0.0], [0.744500041, 7.0, 8.0, 0.0], [0.75999999, 9.0, 10.0, 0.0], [-0.160039246, 0.0, 0.0, 0.0], [0.790500045, 11.0, 12.0, 0.0], [0.740499973, 13.0, 14.0, 0.0], [0.5, 15.0, 16.0, 1.0], [0.208053052, 0.0, 0.0, 0.0], [0.766499996, 17.0, 18.0, 0.0], [0.5, 19.0, 20.0, 1.0], [0.792999983, 21.0, 22.0, 0.0], [-0.000166234677, 0.0, 0.0, 0.0], [0.0611775257, 0.0, 0.0, 0.0], [-0.18440944, 0.0, 0.0, 0.0], [0.029231349, 0.0, 0.0, 0.0], [-0.0495505296, 0.0, 0.0, 0.0], [0.0930215195, 0.0, 0.0, 0.0], [0.151964903, 0.0, 0.0, 0.0], [-0.0392863825, 0.0, 0.0, 0.0], [-0.185795203, 0.0, 0.0, 0.0], [-0.00541531667, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 17, 18, 5, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 6, 11, 12])
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
    function_dict = np.array([[0.863499999, 1.0, 2.0, 0.0], [0.833000004, 3.0, 4.0, 0.0], [0.878499985, 5.0, 6.0, 0.0], [0.823500037, 7.0, 8.0, 0.0], [0.858500004, 9.0, 10.0, 0.0], [0.153623194, 0.0, 0.0, 0.0], [0.924000025, 11.0, 12.0, 0.0], [0.817499995, 13.0, 14.0, 0.0], [0.142975569, 0.0, 0.0, 0.0], [0.851999998, 15.0, 16.0, 0.0], [-0.0157255027, 0.0, 0.0, 0.0], [-0.0692773238, 0.0, 0.0, 0.0], [0.944499969, 17.0, 18.0, 0.0], [0.000821912603, 0.0, 0.0, 0.0], [-0.11471279, 0.0, 0.0, 0.0], [-0.0317688882, 0.0, 0.0, 0.0], [-0.0985413939, 0.0, 0.0, 0.0], [0.145217434, 0.0, 0.0, 0.0], [-0.034667328, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 15, 16, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 2, 6, 12])
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
    function_dict = np.array([[0.863499999, 1.0, 2.0, 0.0], [0.833000004, 3.0, 4.0, 0.0], [0.878499985, 5.0, 6.0, 0.0], [0.823500037, 7.0, 8.0, 0.0], [0.858500004, 9.0, 10.0, 0.0], [-0.153623194, 0.0, 0.0, 0.0], [0.924000025, 11.0, 12.0, 0.0], [0.817499995, 13.0, 14.0, 0.0], [-0.142975569, 0.0, 0.0, 0.0], [0.851999998, 15.0, 16.0, 0.0], [0.0157254767, 0.0, 0.0, 0.0], [0.0692772865, 0.0, 0.0, 0.0], [0.944499969, 17.0, 18.0, 0.0], [-0.000821908878, 0.0, 0.0, 0.0], [0.114712827, 0.0, 0.0, 0.0], [0.0317688994, 0.0, 0.0, 0.0], [0.0985414013, 0.0, 0.0, 0.0], [-0.145217434, 0.0, 0.0, 0.0], [0.0346672684, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 15, 16, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 2, 6, 12])
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
    function_dict = np.array([[0.796000004, 1.0, 2.0, 0.0], [0.789499998, 3.0, 4.0, 0.0], [0.804499984, 5.0, 6.0, 0.0], [0.787500024, 7.0, 8.0, 0.0], [-0.100524537, 0.0, 0.0, 0.0], [0.183756322, 0.0, 0.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.771000028, 11.0, 12.0, 0.0], [0.156657189, 0.0, 0.0, 0.0], [0.82249999, 13.0, 14.0, 0.0], [0.814999998, 15.0, 16.0, 0.0], [0.000475860463, 0.0, 0.0, 0.0], [-0.0683414862, 0.0, 0.0, 0.0], [0.182984039, 0.0, 0.0, 0.0], [0.00675017061, 0.0, 0.0, 0.0], [-0.1895466, 0.0, 0.0, 0.0], [-0.00787562225, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 9, 10])
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
    function_dict = np.array([[0.796000004, 1.0, 2.0, 0.0], [0.789499998, 3.0, 4.0, 0.0], [0.804499984, 5.0, 6.0, 0.0], [0.787500024, 7.0, 8.0, 0.0], [0.100524507, 0.0, 0.0, 0.0], [-0.183756322, 0.0, 0.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.771000028, 11.0, 12.0, 0.0], [-0.156657189, 0.0, 0.0, 0.0], [0.82249999, 13.0, 14.0, 0.0], [0.814999998, 15.0, 16.0, 0.0], [-0.000475866225, 0.0, 0.0, 0.0], [0.068341516, 0.0, 0.0, 0.0], [-0.182984024, 0.0, 0.0, 0.0], [-0.00675017247, 0.0, 0.0, 0.0], [0.1895466, 0.0, 0.0, 0.0], [0.00787560362, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 9, 10])
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
    function_dict = np.array([[0.65200001, 1.0, 2.0, 0.0], [0.648499966, 3.0, 4.0, 0.0], [0.657500029, 5.0, 6.0, 0.0], [0.644500017, 7.0, 8.0, 0.0], [-0.228016883, 0.0, 0.0, 0.0], [0.167718649, 0.0, 0.0, 0.0], [0.664000034, 9.0, 10.0, 0.0], [0.642500043, 11.0, 12.0, 0.0], [0.188890532, 0.0, 0.0, 0.0], [0.662, 13.0, 14.0, 0.0], [0.666499972, 15.0, 16.0, 0.0], [-0.00262384419, 0.0, 0.0, 0.0], [-0.187266022, 0.0, 0.0, 0.0], [-0.0189736187, 0.0, 0.0, 0.0], [-0.141211122, 0.0, 0.0, 0.0], [0.134982616, 0.0, 0.0, 0.0], [0.00160801515, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 9, 10])
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
    function_dict = np.array([[0.65200001, 1.0, 2.0, 0.0], [0.648499966, 3.0, 4.0, 0.0], [0.657500029, 5.0, 6.0, 0.0], [0.644500017, 7.0, 8.0, 0.0], [0.228016883, 0.0, 0.0, 0.0], [-0.167718649, 0.0, 0.0, 0.0], [0.664000034, 9.0, 10.0, 0.0], [0.642500043, 11.0, 12.0, 0.0], [-0.188890517, 0.0, 0.0, 0.0], [0.662, 13.0, 14.0, 0.0], [0.666499972, 15.0, 16.0, 0.0], [0.00262386375, 0.0, 0.0, 0.0], [0.187266022, 0.0, 0.0, 0.0], [0.0189735796, 0.0, 0.0, 0.0], [0.141211078, 0.0, 0.0, 0.0], [-0.134982631, 0.0, 0.0, 0.0], [-0.0016080162, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 9, 10])
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
    function_dict = np.array([[0.701499999, 1.0, 2.0, 0.0], [0.688500047, 3.0, 4.0, 0.0], [0.708500028, 5.0, 6.0, 0.0], [0.686499953, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.158164278, 0.0, 0.0, 0.0], [0.730499983, 11.0, 12.0, 0.0], [0.682500005, 13.0, 14.0, 0.0], [0.0905003324, 0.0, 0.0, 0.0], [0.694000006, 15.0, 16.0, 0.0], [0.696500003, 17.0, 18.0, 0.0], [0.5, 19.0, 20.0, 1.0], [0.740499973, 21.0, 22.0, 0.0], [0.000218744128, 0.0, 0.0, 0.0], [-0.121925898, 0.0, 0.0, 0.0], [-0.00252712891, 0.0, 0.0, 0.0], [-0.0317724347, 0.0, 0.0, 0.0], [-0.194475278, 0.0, 0.0, 0.0], [-0.0328021944, 0.0, 0.0, 0.0], [-0.113545664, 0.0, 0.0, 0.0], [0.0195894409, 0.0, 0.0, 0.0], [0.166993767, 0.0, 0.0, 0.0], [0.00687531009, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 15, 16, 17, 18, 5, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 6, 11, 12])
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
    function_dict = np.array([[0.701499999, 1.0, 2.0, 0.0], [0.688500047, 3.0, 4.0, 0.0], [0.708500028, 5.0, 6.0, 0.0], [0.686499953, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [-0.158164263, 0.0, 0.0, 0.0], [0.730499983, 11.0, 12.0, 0.0], [0.682500005, 13.0, 14.0, 0.0], [-0.0905003622, 0.0, 0.0, 0.0], [0.694000006, 15.0, 16.0, 0.0], [0.696500003, 17.0, 18.0, 0.0], [0.5, 19.0, 20.0, 1.0], [0.740499973, 21.0, 22.0, 0.0], [-0.000218743109, 0.0, 0.0, 0.0], [0.121925876, 0.0, 0.0, 0.0], [0.00252718502, 0.0, 0.0, 0.0], [0.0317724347, 0.0, 0.0, 0.0], [0.194475308, 0.0, 0.0, 0.0], [0.0328021944, 0.0, 0.0, 0.0], [0.113545671, 0.0, 0.0, 0.0], [-0.0195894409, 0.0, 0.0, 0.0], [-0.166993737, 0.0, 0.0, 0.0], [-0.00687530357, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 15, 16, 17, 18, 5, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 6, 11, 12])
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
    function_dict = np.array([[0.774500012, 1.0, 2.0, 0.0], [0.765499949, 3.0, 4.0, 0.0], [0.786499977, 5.0, 6.0, 0.0], [0.709499955, 7.0, 8.0, 0.0], [-0.105161771, 0.0, 0.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.790500045, 11.0, 12.0, 0.0], [0.692000031, 13.0, 14.0, 0.0], [0.713, 15.0, 16.0, 0.0], [-0.0311140995, 0.0, 0.0, 0.0], [0.185229748, 0.0, 0.0, 0.0], [-0.0999794826, 0.0, 0.0, 0.0], [0.792999983, 17.0, 18.0, 0.0], [-0.000437020703, 0.0, 0.0, 0.0], [-0.0673484653, 0.0, 0.0, 0.0], [0.113889888, 0.0, 0.0, 0.0], [0.00691884151, 0.0, 0.0, 0.0], [0.171432927, 0.0, 0.0, 0.0], [-0.00331549579, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 9, 10, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 6, 12])
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
    function_dict = np.array([[0.774500012, 1.0, 2.0, 0.0], [0.765499949, 3.0, 4.0, 0.0], [0.786499977, 5.0, 6.0, 0.0], [0.709499955, 7.0, 8.0, 0.0], [0.105161771, 0.0, 0.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.790500045, 11.0, 12.0, 0.0], [0.692000031, 13.0, 14.0, 0.0], [0.713, 15.0, 16.0, 0.0], [0.0311141405, 0.0, 0.0, 0.0], [-0.185229748, 0.0, 0.0, 0.0], [0.0999794453, 0.0, 0.0, 0.0], [0.792999983, 17.0, 18.0, 0.0], [0.000437019247, 0.0, 0.0, 0.0], [0.0673484653, 0.0, 0.0, 0.0], [-0.113889918, 0.0, 0.0, 0.0], [-0.00691885548, 0.0, 0.0, 0.0], [-0.171432927, 0.0, 0.0, 0.0], [0.0033154944, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 9, 10, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 6, 12])
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
    function_dict = np.array([[0.636500001, 1.0, 2.0, 0.0], [0.631000042, 3.0, 4.0, 0.0], [0.638499975, 5.0, 6.0, 0.0], [0.625999987, 7.0, 8.0, 0.0], [-0.195753947, 0.0, 0.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.65200001, 11.0, 12.0, 0.0], [0.616999984, 13.0, 14.0, 0.0], [0.0757340863, 0.0, 0.0, 0.0], [0.193526238, 0.0, 0.0, 0.0], [0.0101930564, 0.0, 0.0, 0.0], [0.648499966, 15.0, 16.0, 0.0], [0.656499982, 17.0, 18.0, 0.0], [0.00158173381, 0.0, 0.0, 0.0], [-0.0617171563, 0.0, 0.0, 0.0], [0.00757536804, 0.0, 0.0, 0.0], [-0.163361847, 0.0, 0.0, 0.0], [0.132341638, 0.0, 0.0, 0.0], [0.00265522115, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 4, 9, 10, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 2, 5, 6, 11, 12])
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
    function_dict = np.array([[0.636500001, 1.0, 2.0, 0.0], [0.631000042, 3.0, 4.0, 0.0], [0.638499975, 5.0, 6.0, 0.0], [0.625999987, 7.0, 8.0, 0.0], [0.195753947, 0.0, 0.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.65200001, 11.0, 12.0, 0.0], [0.616999984, 13.0, 14.0, 0.0], [-0.075734064, 0.0, 0.0, 0.0], [-0.193526238, 0.0, 0.0, 0.0], [-0.0101930872, 0.0, 0.0, 0.0], [0.648499966, 15.0, 16.0, 0.0], [0.656499982, 17.0, 18.0, 0.0], [-0.00158173253, 0.0, 0.0, 0.0], [0.0617171563, 0.0, 0.0, 0.0], [-0.00757539598, 0.0, 0.0, 0.0], [0.163361847, 0.0, 0.0, 0.0], [-0.132341638, 0.0, 0.0, 0.0], [-0.00265522744, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 4, 9, 10, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 2, 5, 6, 11, 12])
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
    function_dict = np.array([[0.578500032, 1.0, 2.0, 0.0], [0.56400001, 3.0, 4.0, 0.0], [0.582000017, 5.0, 6.0, 0.0], [0.555500031, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.131537795, 0.0, 0.0, 0.0], [0.590499997, 11.0, 12.0, 0.0], [0.548500001, 13.0, 14.0, 0.0], [0.095169127, 0.0, 0.0, 0.0], [0.571500003, 15.0, 16.0, 0.0], [0.0546118245, 0.0, 0.0, 0.0], [-0.0635961369, 0.0, 0.0, 0.0], [0.595000029, 17.0, 18.0, 0.0], [-0.000207438658, 0.0, 0.0, 0.0], [-0.0937414914, 0.0, 0.0, 0.0], [-0.23099874, 0.0, 0.0, 0.0], [-0.0684775412, 0.0, 0.0, 0.0], [0.114791282, 0.0, 0.0, 0.0], [0.00238805474, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 15, 16, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 2, 6, 12])
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
    function_dict = np.array([[0.578500032, 1.0, 2.0, 0.0], [0.56400001, 3.0, 4.0, 0.0], [0.582000017, 5.0, 6.0, 0.0], [0.555500031, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [-0.131537795, 0.0, 0.0, 0.0], [0.590499997, 11.0, 12.0, 0.0], [0.548500001, 13.0, 14.0, 0.0], [-0.0951691493, 0.0, 0.0, 0.0], [0.571500003, 15.0, 16.0, 0.0], [-0.0546117835, 0.0, 0.0, 0.0], [0.0635961518, 0.0, 0.0, 0.0], [0.595000029, 17.0, 18.0, 0.0], [0.000207485617, 0.0, 0.0, 0.0], [0.0937414765, 0.0, 0.0, 0.0], [0.230998725, 0.0, 0.0, 0.0], [0.0684775338, 0.0, 0.0, 0.0], [-0.114791282, 0.0, 0.0, 0.0], [-0.00238805171, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 15, 16, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 2, 6, 12])
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
    function_dict = np.array([[0.302999973, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.327000022, 5.0, 6.0, 0.0], [0.0521999449, 0.0, 0.0, 0.0], [-0.140802056, 0.0, 0.0, 0.0], [0.0857087746, 0.0, 0.0, 0.0], [0.342999995, 7.0, 8.0, 0.0], [-0.0743451566, 0.0, 0.0, 0.0], [0.377499998, 9.0, 10.0, 0.0], [0.0510676876, 0.0, 0.0, 0.0], [-0.00189897395, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 7, 9, 10])
    branch_indices = np.array([0, 1, 2, 6, 8])
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
    function_dict = np.array([[0.302999973, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.327000022, 5.0, 6.0, 0.0], [-0.0522001609, 0.0, 0.0, 0.0], [0.140802085, 0.0, 0.0, 0.0], [-0.085708797, 0.0, 0.0, 0.0], [0.342999995, 7.0, 8.0, 0.0], [0.0743451715, 0.0, 0.0, 0.0], [0.377499998, 9.0, 10.0, 0.0], [-0.0510676764, 0.0, 0.0, 0.0], [0.00189897919, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 7, 9, 10])
    branch_indices = np.array([0, 1, 2, 6, 8])
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
    function_dict = np.array([[0.601999998, 1.0, 2.0, 0.0], [0.595000029, 3.0, 4.0, 0.0], [0.605499983, 5.0, 6.0, 0.0], [0.579999983, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.167971447, 0.0, 0.0, 0.0], [0.607499957, 11.0, 12.0, 0.0], [0.572000027, 13.0, 14.0, 0.0], [0.58949995, 15.0, 16.0, 0.0], [0.00807657186, 0.0, 0.0, 0.0], [-0.166955203, 0.0, 0.0, 0.0], [-0.132054836, 0.0, 0.0, 0.0], [0.61500001, 17.0, 18.0, 0.0], [-0.00471046939, 0.0, 0.0, 0.0], [-0.0883036181, 0.0, 0.0, 0.0], [0.106601238, 0.0, 0.0, 0.0], [-0.00536466436, 0.0, 0.0, 0.0], [0.0868869722, 0.0, 0.0, 0.0], [-0.000371835224, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 12])
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
    function_dict = np.array([[0.601999998, 1.0, 2.0, 0.0], [0.595000029, 3.0, 4.0, 0.0], [0.605499983, 5.0, 6.0, 0.0], [0.579999983, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [-0.167971462, 0.0, 0.0, 0.0], [0.607499957, 11.0, 12.0, 0.0], [0.572000027, 13.0, 14.0, 0.0], [0.58949995, 15.0, 16.0, 0.0], [-0.00807659049, 0.0, 0.0, 0.0], [0.166955203, 0.0, 0.0, 0.0], [0.132054806, 0.0, 0.0, 0.0], [0.61500001, 17.0, 18.0, 0.0], [0.0047104829, 0.0, 0.0, 0.0], [0.0883036181, 0.0, 0.0, 0.0], [-0.106601238, 0.0, 0.0, 0.0], [0.00536467368, 0.0, 0.0, 0.0], [-0.086887002, 0.0, 0.0, 0.0], [0.000371818605, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 12])
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
    function_dict = np.array([[0.863499999, 1.0, 2.0, 0.0], [0.855999947, 3.0, 4.0, 0.0], [0.921499968, 5.0, 6.0, 0.0], [0.834499955, 7.0, 8.0, 0.0], [-0.0741241872, 0.0, 0.0, 0.0], [0.881000042, 9.0, 10.0, 0.0], [0.930500031, 11.0, 12.0, 0.0], [0.82249999, 13.0, 14.0, 0.0], [0.0846837312, 0.0, 0.0, 0.0], [0.0284221247, 0.0, 0.0, 0.0], [0.135894194, 0.0, 0.0, 0.0], [-0.059017241, 0.0, 0.0, 0.0], [0.00406015199, 0.0, 0.0, 0.0], [0.000243578455, 0.0, 0.0, 0.0], [-0.0963125825, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 4, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 2, 5, 6])
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
    function_dict = np.array([[0.863499999, 1.0, 2.0, 0.0], [0.855999947, 3.0, 4.0, 0.0], [0.921499968, 5.0, 6.0, 0.0], [0.834499955, 7.0, 8.0, 0.0], [0.0741242245, 0.0, 0.0, 0.0], [0.881000042, 9.0, 10.0, 0.0], [0.930500031, 11.0, 12.0, 0.0], [0.82249999, 13.0, 14.0, 0.0], [-0.0846837312, 0.0, 0.0, 0.0], [-0.0284221154, 0.0, 0.0, 0.0], [-0.135894209, 0.0, 0.0, 0.0], [0.0590172559, 0.0, 0.0, 0.0], [-0.00406019855, 0.0, 0.0, 0.0], [-0.000243569724, 0.0, 0.0, 0.0], [0.0963125601, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 4, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 2, 5, 6])
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
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.878499985, 3.0, 4.0, 0.0], [0.0568066984, 0.0, 0.0, 0.0], [0.858500004, 5.0, 6.0, 0.0], [-0.0707207397, 0.0, 0.0, 0.0], [0.833000004, 7.0, 8.0, 0.0], [0.862499952, 9.0, 10.0, 0.0], [0.000411887624, 0.0, 0.0, 0.0], [-0.0700895116, 0.0, 0.0, 0.0], [0.145371005, 0.0, 0.0, 0.0], [-0.0129586821, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 4, 2])
    branch_indices = np.array([0, 1, 3, 5, 6])
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
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.878499985, 3.0, 4.0, 0.0], [-0.0568066873, 0.0, 0.0, 0.0], [0.858500004, 5.0, 6.0, 0.0], [0.0707207397, 0.0, 0.0, 0.0], [0.833000004, 7.0, 8.0, 0.0], [0.862499952, 9.0, 10.0, 0.0], [-0.000411887246, 0.0, 0.0, 0.0], [0.0700894743, 0.0, 0.0, 0.0], [-0.14537102, 0.0, 0.0, 0.0], [0.0129586253, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 4, 2])
    branch_indices = np.array([0, 1, 3, 5, 6])
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
    function_dict = np.array([[0.796000004, 1.0, 2.0, 0.0], [0.789499998, 3.0, 4.0, 0.0], [0.817499995, 5.0, 6.0, 0.0], [0.774500012, 7.0, 8.0, 0.0], [0.791499972, 9.0, 10.0, 0.0], [0.813500047, 11.0, 12.0, 0.0], [0.823500037, 13.0, 14.0, 0.0], [0.754000008, 15.0, 16.0, 0.0], [0.786499977, 17.0, 18.0, 0.0], [-0.0318650864, 0.0, 0.0, 0.0], [-0.104915783, 0.0, 0.0, 0.0], [0.807999969, 19.0, 20.0, 0.0], [0.154805169, 0.0, 0.0, 0.0], [-0.10332796, 0.0, 0.0, 0.0], [0.833000004, 21.0, 22.0, 0.0], [-0.000608123839, 0.0, 0.0, 0.0], [-0.0511044823, 0.0, 0.0, 0.0], [0.0886248425, 0.0, 0.0, 0.0], [0.016214285, 0.0, 0.0, 0.0], [0.0397320278, 0.0, 0.0, 0.0], [-0.0209521241, 0.0, 0.0, 0.0], [0.128700361, 0.0, 0.0, 0.0], [0.000968825712, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_99(xs):
    #Predicts Class 1
    function_dict = np.array([[0.796000004, 1.0, 2.0, 0.0], [0.789499998, 3.0, 4.0, 0.0], [0.817499995, 5.0, 6.0, 0.0], [0.774500012, 7.0, 8.0, 0.0], [0.791499972, 9.0, 10.0, 0.0], [0.813500047, 11.0, 12.0, 0.0], [0.823500037, 13.0, 14.0, 0.0], [0.754000008, 15.0, 16.0, 0.0], [0.786499977, 17.0, 18.0, 0.0], [0.031865079, 0.0, 0.0, 0.0], [0.104915768, 0.0, 0.0, 0.0], [0.807999969, 19.0, 20.0, 0.0], [-0.154805169, 0.0, 0.0, 0.0], [0.103327967, 0.0, 0.0, 0.0], [0.833000004, 21.0, 22.0, 0.0], [0.000608129601, 0.0, 0.0, 0.0], [0.0511044972, 0.0, 0.0, 0.0], [-0.0886248425, 0.0, 0.0, 0.0], [-0.0162142906, 0.0, 0.0, 0.0], [-0.0397320166, 0.0, 0.0, 0.0], [0.020952139, 0.0, 0.0, 0.0], [-0.128700346, 0.0, 0.0, 0.0], [-0.000968823908, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_100(xs):
    #Predicts Class 0
    function_dict = np.array([[0.730499983, 1.0, 2.0, 0.0], [0.725000024, 3.0, 4.0, 0.0], [0.740499973, 5.0, 6.0, 0.0], [0.715499997, 7.0, 8.0, 0.0], [-0.110658735, 0.0, 0.0, 0.0], [0.144630224, 0.0, 0.0, 0.0], [0.743499994, 9.0, 10.0, 0.0], [0.710500002, 11.0, 12.0, 0.0], [0.121056616, 0.0, 0.0, 0.0], [-0.0585841462, 0.0, 0.0, 0.0], [0.754000008, 13.0, 14.0, 0.0], [-0.00151526718, 0.0, 0.0, 0.0], [-0.0917415991, 0.0, 0.0, 0.0], [0.0813153759, 0.0, 0.0, 0.0], [0.00469068391, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 10])
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
    function_dict = np.array([[0.730499983, 1.0, 2.0, 0.0], [0.725000024, 3.0, 4.0, 0.0], [0.740499973, 5.0, 6.0, 0.0], [0.715499997, 7.0, 8.0, 0.0], [0.110658728, 0.0, 0.0, 0.0], [-0.144630224, 0.0, 0.0, 0.0], [0.743499994, 9.0, 10.0, 0.0], [0.710500002, 11.0, 12.0, 0.0], [-0.121056587, 0.0, 0.0, 0.0], [0.058584176, 0.0, 0.0, 0.0], [0.754000008, 13.0, 14.0, 0.0], [0.00151526346, 0.0, 0.0, 0.0], [0.091741614, 0.0, 0.0, 0.0], [-0.0813153833, 0.0, 0.0, 0.0], [-0.00469069043, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 10])
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
    function_dict = np.array([[0.670000017, 1.0, 2.0, 0.0], [0.666499972, 3.0, 4.0, 0.0], [0.676999986, 5.0, 6.0, 0.0], [0.664000034, 7.0, 8.0, 0.0], [-0.148167506, 0.0, 0.0, 0.0], [0.138397127, 0.0, 0.0, 0.0], [0.681499958, 9.0, 10.0, 0.0], [0.662, 11.0, 12.0, 0.0], [0.0940480828, 0.0, 0.0, 0.0], [-0.0780214518, 0.0, 0.0, 0.0], [0.682500005, 13.0, 14.0, 0.0], [-0.000334394717, 0.0, 0.0, 0.0], [-0.101910144, 0.0, 0.0, 0.0], [0.113409303, 0.0, 0.0, 0.0], [0.00208342308, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 10])
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
    function_dict = np.array([[0.670000017, 1.0, 2.0, 0.0], [0.666499972, 3.0, 4.0, 0.0], [0.676999986, 5.0, 6.0, 0.0], [0.664000034, 7.0, 8.0, 0.0], [0.148167491, 0.0, 0.0, 0.0], [-0.138397112, 0.0, 0.0, 0.0], [0.681499958, 9.0, 10.0, 0.0], [0.662, 11.0, 12.0, 0.0], [-0.0940480828, 0.0, 0.0, 0.0], [0.0780214518, 0.0, 0.0, 0.0], [0.682500005, 13.0, 14.0, 0.0], [0.000334397249, 0.0, 0.0, 0.0], [0.101910159, 0.0, 0.0, 0.0], [-0.113409325, 0.0, 0.0, 0.0], [-0.00208342215, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 10])
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
    function_dict = np.array([[0.5, 1.0, 2.0, 1.0], [0.334500015, 3.0, 4.0, 0.0], [0.792999983, 5.0, 6.0, 0.0], [0.310499996, 7.0, 8.0, 0.0], [0.613999963, 9.0, 10.0, 0.0], [0.709499955, 11.0, 12.0, 0.0], [0.813500047, 13.0, 14.0, 0.0], [0.0391345508, 0.0, 0.0, 0.0], [-0.158035859, 0.0, 0.0, 0.0], [0.5995, 15.0, 16.0, 0.0], [0.63499999, 17.0, 18.0, 0.0], [0.671499968, 19.0, 20.0, 0.0], [0.767499983, 21.0, 22.0, 0.0], [-0.150115669, 0.0, 0.0, 0.0], [0.818500042, 23.0, 24.0, 0.0], [0.0130542815, 0.0, 0.0, 0.0], [0.163390234, 0.0, 0.0, 0.0], [-0.174937665, 0.0, 0.0, 0.0], [0.0249862708, 0.0, 0.0, 0.0], [4.33790119e-05, 0.0, 0.0, 0.0], [-0.0861386955, 0.0, 0.0, 0.0], [0.0359854139, 0.0, 0.0, 0.0], [0.122215509, 0.0, 0.0, 0.0], [0.0752286091, 0.0, 0.0, 0.0], [-0.0351136923, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 13, 23, 24])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 14])
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
    function_dict = np.array([[0.5, 1.0, 2.0, 1.0], [0.334500015, 3.0, 4.0, 0.0], [0.792999983, 5.0, 6.0, 0.0], [0.310499996, 7.0, 8.0, 0.0], [0.613999963, 9.0, 10.0, 0.0], [0.709499955, 11.0, 12.0, 0.0], [0.813500047, 13.0, 14.0, 0.0], [-0.0391345695, 0.0, 0.0, 0.0], [0.158035845, 0.0, 0.0, 0.0], [0.5995, 15.0, 16.0, 0.0], [0.63499999, 17.0, 18.0, 0.0], [0.671499968, 19.0, 20.0, 0.0], [0.767499983, 21.0, 22.0, 0.0], [0.150115639, 0.0, 0.0, 0.0], [0.818500042, 23.0, 24.0, 0.0], [-0.0130543076, 0.0, 0.0, 0.0], [-0.163390219, 0.0, 0.0, 0.0], [0.17493768, 0.0, 0.0, 0.0], [-0.0249862783, 0.0, 0.0, 0.0], [-4.33827445e-05, 0.0, 0.0, 0.0], [0.0861386955, 0.0, 0.0, 0.0], [-0.0359854139, 0.0, 0.0, 0.0], [-0.122215524, 0.0, 0.0, 0.0], [-0.0752286091, 0.0, 0.0, 0.0], [0.035113696, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 13, 23, 24])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 12, 6, 14])
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
    function_dict = np.array([[0.622500002, 1.0, 2.0, 0.0], [0.618999958, 3.0, 4.0, 0.0], [0.624500036, 5.0, 6.0, 0.0], [0.613499999, 7.0, 8.0, 0.0], [-0.176245973, 0.0, 0.0, 0.0], [0.133841366, 0.0, 0.0, 0.0], [0.636500001, 9.0, 10.0, 0.0], [0.5, 11.0, 12.0, 1.0], [0.5, 13.0, 14.0, 1.0], [0.631000042, 15.0, 16.0, 0.0], [0.638499975, 17.0, 18.0, 0.0], [0.014274355, 0.0, 0.0, 0.0], [-0.0238155257, 0.0, 0.0, 0.0], [-0.204605281, 0.0, 0.0, 0.0], [0.272566587, 0.0, 0.0, 0.0], [0.0232062768, 0.0, 0.0, 0.0], [-0.145912468, 0.0, 0.0, 0.0], [0.0792533234, 0.0, 0.0, 0.0], [0.000514993735, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_107(xs):
    #Predicts Class 1
    function_dict = np.array([[0.622500002, 1.0, 2.0, 0.0], [0.618999958, 3.0, 4.0, 0.0], [0.624500036, 5.0, 6.0, 0.0], [0.613499999, 7.0, 8.0, 0.0], [0.176245987, 0.0, 0.0, 0.0], [-0.133841366, 0.0, 0.0, 0.0], [0.636500001, 9.0, 10.0, 0.0], [0.5, 11.0, 12.0, 1.0], [0.5, 13.0, 14.0, 1.0], [0.631000042, 15.0, 16.0, 0.0], [0.638499975, 17.0, 18.0, 0.0], [-0.0142743504, 0.0, 0.0, 0.0], [0.0238155276, 0.0, 0.0, 0.0], [0.204605296, 0.0, 0.0, 0.0], [-0.272566557, 0.0, 0.0, 0.0], [-0.0232062768, 0.0, 0.0, 0.0], [0.145912454, 0.0, 0.0, 0.0], [-0.0792533234, 0.0, 0.0, 0.0], [-0.000514994783, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_108(xs):
    #Predicts Class 0
    function_dict = np.array([[0.377499998, 1.0, 2.0, 0.0], [0.370499998, 3.0, 4.0, 0.0], [0.402999997, 5.0, 6.0, 0.0], [0.345499992, 7.0, 8.0, 0.0], [0.142631292, 0.0, 0.0, 0.0], [0.380999982, 9.0, 10.0, 0.0], [0.411000013, 11.0, 12.0, 0.0], [0.323000014, 13.0, 14.0, 0.0], [0.358500004, 15.0, 16.0, 0.0], [-0.0316245072, 0.0, 0.0, 0.0], [-0.147213638, 0.0, 0.0, 0.0], [0.408500016, 17.0, 18.0, 0.0], [0.424000025, 19.0, 20.0, 0.0], [-0.0348349325, 0.0, 0.0, 0.0], [0.0925525501, 0.0, 0.0, 0.0], [-0.0815250576, 0.0, 0.0, 0.0], [0.0321850814, 0.0, 0.0, 0.0], [0.130472884, 0.0, 0.0, 0.0], [0.0334390998, 0.0, 0.0, 0.0], [-0.147237569, 0.0, 0.0, 0.0], [-0.000583187502, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 9, 10, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 6, 11, 12])
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
    function_dict = np.array([[0.377499998, 1.0, 2.0, 0.0], [0.370499998, 3.0, 4.0, 0.0], [0.402999997, 5.0, 6.0, 0.0], [0.345499992, 7.0, 8.0, 0.0], [-0.142631277, 0.0, 0.0, 0.0], [0.380999982, 9.0, 10.0, 0.0], [0.411000013, 11.0, 12.0, 0.0], [0.323000014, 13.0, 14.0, 0.0], [0.358500004, 15.0, 16.0, 0.0], [0.0316245221, 0.0, 0.0, 0.0], [0.147213712, 0.0, 0.0, 0.0], [0.408500016, 17.0, 18.0, 0.0], [0.424000025, 19.0, 20.0, 0.0], [0.0348348245, 0.0, 0.0, 0.0], [-0.0925525278, 0.0, 0.0, 0.0], [0.0815250799, 0.0, 0.0, 0.0], [-0.0321850665, 0.0, 0.0, 0.0], [-0.130472854, 0.0, 0.0, 0.0], [-0.0334391035, 0.0, 0.0, 0.0], [0.147237465, 0.0, 0.0, 0.0], [0.00058318337, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 9, 10, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 6, 11, 12])
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
    function_dict = np.array([[0.944499969, 1.0, 2.0, 0.0], [0.863499999, 3.0, 4.0, 0.0], [-0.0567014515, 0.0, 0.0, 0.0], [0.855999947, 5.0, 6.0, 0.0], [0.921499968, 7.0, 8.0, 0.0], [0.834499955, 9.0, 10.0, 0.0], [-0.0599548034, 0.0, 0.0, 0.0], [0.083467938, 0.0, 0.0, 0.0], [-0.014035726, 0.0, 0.0, 0.0], [-0.00191051862, 0.0, 0.0, 0.0], [0.0750721097, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 6, 7, 8, 2])
    branch_indices = np.array([0, 1, 3, 5, 4])
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
    function_dict = np.array([[0.944499969, 1.0, 2.0, 0.0], [0.863499999, 3.0, 4.0, 0.0], [0.0567014255, 0.0, 0.0, 0.0], [0.855999947, 5.0, 6.0, 0.0], [0.921499968, 7.0, 8.0, 0.0], [0.834499955, 9.0, 10.0, 0.0], [0.0599548146, 0.0, 0.0, 0.0], [-0.0834679604, 0.0, 0.0, 0.0], [0.0140356598, 0.0, 0.0, 0.0], [0.0019105213, 0.0, 0.0, 0.0], [-0.0750721321, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 6, 7, 8, 2])
    branch_indices = np.array([0, 1, 3, 5, 4])
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
    function_dict = np.array([[0.364499986, 1.0, 2.0, 0.0], [0.358500004, 3.0, 4.0, 0.0], [0.37349999, 5.0, 6.0, 0.0], [0.353500009, 7.0, 8.0, 0.0], [0.139990807, 0.0, 0.0, 0.0], [-0.176410422, 0.0, 0.0, 0.0], [0.405000001, 9.0, 10.0, 0.0], [0.342999995, 11.0, 12.0, 0.0], [-0.0760650858, 0.0, 0.0, 0.0], [0.401499987, 13.0, 14.0, 0.0], [0.424000025, 15.0, 16.0, 0.0], [-0.0128415171, 0.0, 0.0, 0.0], [0.0820156708, 0.0, 0.0, 0.0], [0.0202311203, 0.0, 0.0, 0.0], [0.156213999, 0.0, 0.0, 0.0], [-0.0718283728, 0.0, 0.0, 0.0], [1.43343032e-05, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 9, 10])
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
    function_dict = np.array([[0.364499986, 1.0, 2.0, 0.0], [0.358500004, 3.0, 4.0, 0.0], [0.37349999, 5.0, 6.0, 0.0], [0.353500009, 7.0, 8.0, 0.0], [-0.139990807, 0.0, 0.0, 0.0], [0.176410407, 0.0, 0.0, 0.0], [0.405000001, 9.0, 10.0, 0.0], [0.342999995, 11.0, 12.0, 0.0], [0.0760651082, 0.0, 0.0, 0.0], [0.401499987, 13.0, 14.0, 0.0], [0.424000025, 15.0, 16.0, 0.0], [0.0128414566, 0.0, 0.0, 0.0], [-0.0820156932, 0.0, 0.0, 0.0], [-0.0202311557, 0.0, 0.0, 0.0], [-0.156214029, 0.0, 0.0, 0.0], [0.0718283951, 0.0, 0.0, 0.0], [-1.43372608e-05, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 13, 14, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 9, 10])
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
    function_dict = np.array([[0.27700001, 1.0, 2.0, 0.0], [0.0568546429, 0.0, 0.0, 0.0], [0.323000014, 3.0, 4.0, 0.0], [0.5, 5.0, 6.0, 1.0], [0.345499992, 7.0, 8.0, 0.0], [0.0274136979, 0.0, 0.0, 0.0], [-0.133324191, 0.0, 0.0, 0.0], [0.334500015, 9.0, 10.0, 0.0], [0.358500004, 11.0, 12.0, 0.0], [0.011330368, 0.0, 0.0, 0.0], [0.0696601123, 0.0, 0.0, 0.0], [-0.0505889505, 0.0, 0.0, 0.0], [0.000365804241, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 5, 6, 9, 10, 11, 12])
    branch_indices = np.array([0, 2, 3, 4, 7, 8])
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
    function_dict = np.array([[0.27700001, 1.0, 2.0, 0.0], [-0.0568545684, 0.0, 0.0, 0.0], [0.323000014, 3.0, 4.0, 0.0], [0.5, 5.0, 6.0, 1.0], [0.345499992, 7.0, 8.0, 0.0], [-0.0274137314, 0.0, 0.0, 0.0], [0.133324161, 0.0, 0.0, 0.0], [0.334500015, 9.0, 10.0, 0.0], [0.358500004, 11.0, 12.0, 0.0], [-0.0113304248, 0.0, 0.0, 0.0], [-0.0696601197, 0.0, 0.0, 0.0], [0.0505889282, 0.0, 0.0, 0.0], [-0.000365805143, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 5, 6, 9, 10, 11, 12])
    branch_indices = np.array([0, 2, 3, 4, 7, 8])
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
    function_dict = np.array([[0.425500005, 1.0, 2.0, 0.0], [0.406499982, 3.0, 4.0, 0.0], [0.436500013, 5.0, 6.0, 0.0], [0.377499998, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [-0.171129122, 0.0, 0.0, 0.0], [0.442499995, 11.0, 12.0, 0.0], [0.351999998, 13.0, 14.0, 0.0], [0.402999997, 15.0, 16.0, 0.0], [0.132279038, 0.0, 0.0, 0.0], [0.033780314, 0.0, 0.0, 0.0], [0.136059076, 0.0, 0.0, 0.0], [0.469500005, 17.0, 18.0, 0.0], [-0.0230246224, 0.0, 0.0, 0.0], [0.0680726469, 0.0, 0.0, 0.0], [-0.0954492763, 0.0, 0.0, 0.0], [0.0134407971, 0.0, 0.0, 0.0], [-0.111999713, 0.0, 0.0, 0.0], [0.00058551348, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 12])
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
    function_dict = np.array([[0.425500005, 1.0, 2.0, 0.0], [0.406499982, 3.0, 4.0, 0.0], [0.436500013, 5.0, 6.0, 0.0], [0.377499998, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.171129137, 0.0, 0.0, 0.0], [0.442499995, 11.0, 12.0, 0.0], [0.351999998, 13.0, 14.0, 0.0], [0.402999997, 15.0, 16.0, 0.0], [-0.132279038, 0.0, 0.0, 0.0], [-0.033780355, 0.0, 0.0, 0.0], [-0.13605909, 0.0, 0.0, 0.0], [0.469500005, 17.0, 18.0, 0.0], [0.0230246484, 0.0, 0.0, 0.0], [-0.0680726618, 0.0, 0.0, 0.0], [0.0954493284, 0.0, 0.0, 0.0], [-0.0134408223, 0.0, 0.0, 0.0], [0.111999691, 0.0, 0.0, 0.0], [-0.000585514179, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 9, 10, 5, 11, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 6, 12])
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
    function_dict = np.array([[0.327000022, 1.0, 2.0, 0.0], [0.5, 3.0, 4.0, 1.0], [0.351999998, 5.0, 6.0, 0.0], [-0.035929855, 0.0, 0.0, 0.0], [0.0904563442, 0.0, 0.0, 0.0], [0.335500002, 7.0, 8.0, 0.0], [0.355499983, 9.0, 10.0, 0.0], [-0.014994354, 0.0, 0.0, 0.0], [-0.091178067, 0.0, 0.0, 0.0], [0.144224286, 0.0, 0.0, 0.0], [0.37349999, 11.0, 12.0, 0.0], [-0.0884574428, 0.0, 0.0, 0.0], [0.000827177137, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 11, 12])
    branch_indices = np.array([0, 1, 2, 5, 6, 10])
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
    function_dict = np.array([[0.327000022, 1.0, 2.0, 0.0], [0.5, 3.0, 4.0, 1.0], [0.351999998, 5.0, 6.0, 0.0], [0.0359298401, 0.0, 0.0, 0.0], [-0.0904564261, 0.0, 0.0, 0.0], [0.335500002, 7.0, 8.0, 0.0], [0.355499983, 9.0, 10.0, 0.0], [0.0149943866, 0.0, 0.0, 0.0], [0.0911780968, 0.0, 0.0, 0.0], [-0.144224286, 0.0, 0.0, 0.0], [0.37349999, 11.0, 12.0, 0.0], [0.0884574726, 0.0, 0.0, 0.0], [-0.000827172713, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 11, 12])
    branch_indices = np.array([0, 1, 2, 5, 6, 10])
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
    function_dict = np.array([[0.271499991, 1.0, 2.0, 0.0], [-0.127903864, 0.0, 0.0, 0.0], [0.305999994, 3.0, 4.0, 0.0], [0.139194354, 0.0, 0.0, 0.0], [0.334500015, 5.0, 6.0, 0.0], [0.5, 7.0, 8.0, 1.0], [0.345499992, 9.0, 10.0, 0.0], [-0.139212683, 0.0, 0.0, 0.0], [0.0413031913, 0.0, 0.0, 0.0], [0.0722064227, 0.0, 0.0, 0.0], [-0.00135998405, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 7, 8, 9, 10])
    branch_indices = np.array([0, 2, 4, 5, 6])
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
    function_dict = np.array([[0.271499991, 1.0, 2.0, 0.0], [0.127903998, 0.0, 0.0, 0.0], [0.305999994, 3.0, 4.0, 0.0], [-0.139194235, 0.0, 0.0, 0.0], [0.334500015, 5.0, 6.0, 0.0], [0.5, 7.0, 8.0, 1.0], [0.345499992, 9.0, 10.0, 0.0], [0.139212683, 0.0, 0.0, 0.0], [-0.0413032323, 0.0, 0.0, 0.0], [-0.0722064599, 0.0, 0.0, 0.0], [0.00135998649, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 7, 8, 9, 10])
    branch_indices = np.array([0, 2, 4, 5, 6])
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
    function_dict = np.array([[0.302999973, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.327000022, 5.0, 6.0, 0.0], [0.0373290814, 0.0, 0.0, 0.0], [-0.128790021, 0.0, 0.0, 0.0], [0.0799692422, 0.0, 0.0, 0.0], [0.342999995, 7.0, 8.0, 0.0], [-0.0710777342, 0.0, 0.0, 0.0], [0.353500009, 9.0, 10.0, 0.0], [0.0666984096, 0.0, 0.0, 0.0], [-0.000496755762, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 7, 9, 10])
    branch_indices = np.array([0, 1, 2, 6, 8])
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
    function_dict = np.array([[0.302999973, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.327000022, 5.0, 6.0, 0.0], [-0.0373287611, 0.0, 0.0, 0.0], [0.128790051, 0.0, 0.0, 0.0], [-0.0799692795, 0.0, 0.0, 0.0], [0.342999995, 7.0, 8.0, 0.0], [0.0710777715, 0.0, 0.0, 0.0], [0.353500009, 9.0, 10.0, 0.0], [-0.0666984171, 0.0, 0.0, 0.0], [0.000496747205, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 7, 9, 10])
    branch_indices = np.array([0, 1, 2, 6, 8])
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
    function_dict = np.array([[0.323000014, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.335500002, 5.0, 6.0, 0.0], [0.0259373449, 0.0, 0.0, 0.0], [0.5, 7.0, 8.0, 1.0], [0.334500015, 9.0, 10.0, 0.0], [0.351999998, 11.0, 12.0, 0.0], [0.0150179612, 0.0, 0.0, 0.0], [-0.135295302, 0.0, 0.0, 0.0], [0.0214993022, 0.0, 0.0, 0.0], [0.0800391436, 0.0, 0.0, 0.0], [-0.0850930065, 0.0, 0.0, 0.0], [0.355499983, 13.0, 14.0, 0.0], [0.0947188064, 0.0, 0.0, 0.0], [-0.000463512522, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 10, 11, 13, 14])
    branch_indices = np.array([0, 1, 4, 2, 5, 6, 12])
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
    function_dict = np.array([[0.323000014, 1.0, 2.0, 0.0], [0.27700001, 3.0, 4.0, 0.0], [0.335500002, 5.0, 6.0, 0.0], [-0.0259378497, 0.0, 0.0, 0.0], [0.5, 7.0, 8.0, 1.0], [0.334500015, 9.0, 10.0, 0.0], [0.351999998, 11.0, 12.0, 0.0], [-0.0150179435, 0.0, 0.0, 0.0], [0.135295317, 0.0, 0.0, 0.0], [-0.0214992929, 0.0, 0.0, 0.0], [-0.0800391585, 0.0, 0.0, 0.0], [0.0850930288, 0.0, 0.0, 0.0], [0.355499983, 13.0, 14.0, 0.0], [-0.0947188362, 0.0, 0.0, 0.0], [0.000463514414, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 10, 11, 13, 14])
    branch_indices = np.array([0, 1, 4, 2, 5, 6, 12])
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
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.862499952, 3.0, 4.0, 0.0], [0.045581799, 0.0, 0.0, 0.0], [0.834499955, 5.0, 6.0, 0.0], [0.881000042, 7.0, 8.0, 0.0], [0.82249999, 9.0, 10.0, 0.0], [0.5, 11.0, 12.0, 1.0], [-0.0803975537, 0.0, 0.0, 0.0], [-0.0200571474, 0.0, 0.0, 0.0], [0.00107493345, 0.0, 0.0, 0.0], [-0.0862010494, 0.0, 0.0, 0.0], [0.0159884468, 0.0, 0.0, 0.0], [0.0875698254, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 7, 8, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4])
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
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.862499952, 3.0, 4.0, 0.0], [-0.0455817953, 0.0, 0.0, 0.0], [0.834499955, 5.0, 6.0, 0.0], [0.881000042, 7.0, 8.0, 0.0], [0.82249999, 9.0, 10.0, 0.0], [0.5, 11.0, 12.0, 1.0], [0.0803975835, 0.0, 0.0, 0.0], [0.0200572237, 0.0, 0.0, 0.0], [-0.00107493321, 0.0, 0.0, 0.0], [0.0862010121, 0.0, 0.0, 0.0], [-0.0159884617, 0.0, 0.0, 0.0], [-0.0875698254, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 7, 8, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4])
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
    function_dict = np.array([[0.796000004, 1.0, 2.0, 0.0], [0.771000028, 3.0, 4.0, 0.0], [0.804499984, 5.0, 6.0, 0.0], [0.75999999, 7.0, 8.0, 0.0], [0.787500024, 9.0, 10.0, 0.0], [0.165402859, 0.0, 0.0, 0.0], [0.5, 11.0, 12.0, 1.0], [0.756500006, 13.0, 14.0, 0.0], [0.109890506, 0.0, 0.0, 0.0], [-0.116476588, 0.0, 0.0, 0.0], [0.791499972, 15.0, 16.0, 0.0], [0.833000004, 17.0, 18.0, 0.0], [0.858500004, 19.0, 20.0, 0.0], [0.000855845399, 0.0, 0.0, 0.0], [-0.138161927, 0.0, 0.0, 0.0], [0.0473067239, 0.0, 0.0, 0.0], [-0.0612152964, 0.0, 0.0, 0.0], [0.121698409, 0.0, 0.0, 0.0], [-0.0267689433, 0.0, 0.0, 0.0], [-0.1089084, 0.0, 0.0, 0.0], [0.0523612276, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 9, 15, 16, 5, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2, 6, 11, 12])
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
    function_dict = np.array([[0.796000004, 1.0, 2.0, 0.0], [0.771000028, 3.0, 4.0, 0.0], [0.804499984, 5.0, 6.0, 0.0], [0.75999999, 7.0, 8.0, 0.0], [0.787500024, 9.0, 10.0, 0.0], [-0.165402859, 0.0, 0.0, 0.0], [0.5, 11.0, 12.0, 1.0], [0.756500006, 13.0, 14.0, 0.0], [-0.109890528, 0.0, 0.0, 0.0], [0.116476588, 0.0, 0.0, 0.0], [0.791499972, 15.0, 16.0, 0.0], [0.833000004, 17.0, 18.0, 0.0], [0.858500004, 19.0, 20.0, 0.0], [-0.000855832128, 0.0, 0.0, 0.0], [0.138161957, 0.0, 0.0, 0.0], [-0.0473066904, 0.0, 0.0, 0.0], [0.0612153001, 0.0, 0.0, 0.0], [-0.121698424, 0.0, 0.0, 0.0], [0.0267689787, 0.0, 0.0, 0.0], [0.108908437, 0.0, 0.0, 0.0], [-0.052361194, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 9, 15, 16, 5, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2, 6, 11, 12])
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
    function_dict = np.array([[0.640499949, 1.0, 2.0, 0.0], [0.631000042, 3.0, 4.0, 0.0], [0.6505, 5.0, 6.0, 0.0], [0.622500002, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.644500017, 11.0, 12.0, 0.0], [0.670000017, 13.0, 14.0, 0.0], [0.618999958, 15.0, 16.0, 0.0], [0.624500036, 17.0, 18.0, 0.0], [0.0582130551, 0.0, 0.0, 0.0], [0.637500048, 19.0, 20.0, 0.0], [0.0347874239, 0.0, 0.0, 0.0], [0.171447292, 0.0, 0.0, 0.0], [0.5, 21.0, 22.0, 1.0], [0.680500031, 23.0, 24.0, 0.0], [-0.00270975544, 0.0, 0.0, 0.0], [-0.148988768, 0.0, 0.0, 0.0], [0.0855403543, 0.0, 0.0, 0.0], [0.014790576, 0.0, 0.0, 0.0], [-0.173974276, 0.0, 0.0, 0.0], [-0.0447553247, 0.0, 0.0, 0.0], [-0.121647321, 0.0, 0.0, 0.0], [0.00528051751, 0.0, 0.0, 0.0], [0.0938239023, 0.0, 0.0, 0.0], [0.000943672727, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 11, 12, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 6, 13, 14])
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
    function_dict = np.array([[0.640499949, 1.0, 2.0, 0.0], [0.631000042, 3.0, 4.0, 0.0], [0.6505, 5.0, 6.0, 0.0], [0.622500002, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.644500017, 11.0, 12.0, 0.0], [0.670000017, 13.0, 14.0, 0.0], [0.618999958, 15.0, 16.0, 0.0], [0.624500036, 17.0, 18.0, 0.0], [-0.0582130477, 0.0, 0.0, 0.0], [0.637500048, 19.0, 20.0, 0.0], [-0.0347874239, 0.0, 0.0, 0.0], [-0.171447277, 0.0, 0.0, 0.0], [0.5, 21.0, 22.0, 1.0], [0.680500031, 23.0, 24.0, 0.0], [0.00270976103, 0.0, 0.0, 0.0], [0.148988768, 0.0, 0.0, 0.0], [-0.0855403617, 0.0, 0.0, 0.0], [-0.0147905927, 0.0, 0.0, 0.0], [0.173974261, 0.0, 0.0, 0.0], [0.0447553173, 0.0, 0.0, 0.0], [0.121647343, 0.0, 0.0, 0.0], [-0.00528050214, 0.0, 0.0, 0.0], [-0.0938239023, 0.0, 0.0, 0.0], [-0.000943666557, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 11, 12, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 6, 13, 14])
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
    function_dict = np.array([[0.790500045, 1.0, 2.0, 0.0], [0.786499977, 3.0, 4.0, 0.0], [0.792999983, 5.0, 6.0, 0.0], [0.774500012, 7.0, 8.0, 0.0], [-0.0897245854, 0.0, 0.0, 0.0], [0.142801657, 0.0, 0.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.765499949, 11.0, 12.0, 0.0], [0.5, 13.0, 14.0, 1.0], [0.82249999, 15.0, 16.0, 0.0], [0.813500047, 17.0, 18.0, 0.0], [0.000374294323, 0.0, 0.0, 0.0], [-0.0873567685, 0.0, 0.0, 0.0], [-0.0240476299, 0.0, 0.0, 0.0], [0.14522098, 0.0, 0.0, 0.0], [0.152344063, 0.0, 0.0, 0.0], [-0.0104280561, 0.0, 0.0, 0.0], [-0.10229183, 0.0, 0.0, 0.0], [0.0190570224, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_133(xs):
    #Predicts Class 1
    function_dict = np.array([[0.790500045, 1.0, 2.0, 0.0], [0.786499977, 3.0, 4.0, 0.0], [0.792999983, 5.0, 6.0, 0.0], [0.774500012, 7.0, 8.0, 0.0], [0.0897245631, 0.0, 0.0, 0.0], [-0.142801642, 0.0, 0.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.765499949, 11.0, 12.0, 0.0], [0.5, 13.0, 14.0, 1.0], [0.82249999, 15.0, 16.0, 0.0], [0.813500047, 17.0, 18.0, 0.0], [-0.000374294264, 0.0, 0.0, 0.0], [0.0873567611, 0.0, 0.0, 0.0], [0.0240476038, 0.0, 0.0, 0.0], [-0.14522098, 0.0, 0.0, 0.0], [-0.152344033, 0.0, 0.0, 0.0], [0.0104280934, 0.0, 0.0, 0.0], [0.102291852, 0.0, 0.0, 0.0], [-0.0190570261, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_134(xs):
    #Predicts Class 0
    function_dict = np.array([[0.302999973, 1.0, 2.0, 0.0], [-0.0440456159, 0.0, 0.0, 0.0], [0.327000022, 3.0, 4.0, 0.0], [0.0606802478, 0.0, 0.0, 0.0], [0.358500004, 5.0, 6.0, 0.0], [0.345499992, 7.0, 8.0, 0.0], [0.364499986, 9.0, 10.0, 0.0], [0.00957066845, 0.0, 0.0, 0.0], [-0.0759100243, 0.0, 0.0, 0.0], [0.106234878, 0.0, 0.0, 0.0], [0.000371635193, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 7, 8, 9, 10])
    branch_indices = np.array([0, 2, 4, 5, 6])
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
    function_dict = np.array([[0.302999973, 1.0, 2.0, 0.0], [0.044045575, 0.0, 0.0, 0.0], [0.327000022, 3.0, 4.0, 0.0], [-0.0606802851, 0.0, 0.0, 0.0], [0.358500004, 5.0, 6.0, 0.0], [0.345499992, 7.0, 8.0, 0.0], [0.364499986, 9.0, 10.0, 0.0], [-0.00957063772, 0.0, 0.0, 0.0], [0.0759099945, 0.0, 0.0, 0.0], [-0.106234856, 0.0, 0.0, 0.0], [-0.000371640926, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 7, 8, 9, 10])
    branch_indices = np.array([0, 2, 4, 5, 6])
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
    function_dict = np.array([[0.640499949, 1.0, 2.0, 0.0], [0.631000042, 3.0, 4.0, 0.0], [0.6505, 5.0, 6.0, 0.0], [0.625999987, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [0.0902198851, 0.0, 0.0, 0.0], [0.654999971, 11.0, 12.0, 0.0], [0.514500022, 13.0, 14.0, 0.0], [0.0689416304, 0.0, 0.0, 0.0], [0.0419453159, 0.0, 0.0, 0.0], [-0.0887235105, 0.0, 0.0, 0.0], [-0.0560926013, 0.0, 0.0, 0.0], [0.657500029, 15.0, 16.0, 0.0], [0.00756748579, 0.0, 0.0, 0.0], [-0.0156759378, 0.0, 0.0, 0.0], [0.0783200487, 0.0, 0.0, 0.0], [0.00208460586, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 9, 10, 5, 11, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 4, 2, 6, 12])
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
    function_dict = np.array([[0.640499949, 1.0, 2.0, 0.0], [0.631000042, 3.0, 4.0, 0.0], [0.6505, 5.0, 6.0, 0.0], [0.625999987, 7.0, 8.0, 0.0], [0.5, 9.0, 10.0, 1.0], [-0.0902198702, 0.0, 0.0, 0.0], [0.654999971, 11.0, 12.0, 0.0], [0.514500022, 13.0, 14.0, 0.0], [-0.0689416379, 0.0, 0.0, 0.0], [-0.0419453122, 0.0, 0.0, 0.0], [0.0887235105, 0.0, 0.0, 0.0], [0.0560925864, 0.0, 0.0, 0.0], [0.657500029, 15.0, 16.0, 0.0], [-0.0075674355, 0.0, 0.0, 0.0], [0.015675934, 0.0, 0.0, 0.0], [-0.0783200487, 0.0, 0.0, 0.0], [-0.0020846054, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 9, 10, 5, 11, 15, 16])
    branch_indices = np.array([0, 1, 3, 7, 4, 2, 6, 12])
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
    function_dict = np.array([[0.37349999, 1.0, 2.0, 0.0], [0.364499986, 3.0, 4.0, 0.0], [0.380999982, 5.0, 6.0, 0.0], [0.358500004, 7.0, 8.0, 0.0], [-0.14801167, 0.0, 0.0, 0.0], [0.0949921161, 0.0, 0.0, 0.0], [0.406499982, 9.0, 10.0, 0.0], [0.353500009, 11.0, 12.0, 0.0], [0.0734986588, 0.0, 0.0, 0.0], [-0.0673198253, 0.0, 0.0, 0.0], [0.425500005, 13.0, 14.0, 0.0], [0.000720544194, 0.0, 0.0, 0.0], [-0.0694495738, 0.0, 0.0, 0.0], [0.0666112602, 0.0, 0.0, 0.0], [0.00012486137, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 10])
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
    function_dict = np.array([[0.37349999, 1.0, 2.0, 0.0], [0.364499986, 3.0, 4.0, 0.0], [0.380999982, 5.0, 6.0, 0.0], [0.358500004, 7.0, 8.0, 0.0], [0.148011699, 0.0, 0.0, 0.0], [-0.0949921161, 0.0, 0.0, 0.0], [0.406499982, 9.0, 10.0, 0.0], [0.353500009, 11.0, 12.0, 0.0], [-0.0734986439, 0.0, 0.0, 0.0], [0.0673198178, 0.0, 0.0, 0.0], [0.425500005, 13.0, 14.0, 0.0], [-0.00072056253, 0.0, 0.0, 0.0], [0.0694495738, 0.0, 0.0, 0.0], [-0.0666112825, 0.0, 0.0, 0.0], [-0.000124862039, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 4, 5, 9, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 2, 6, 10])
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
    function_dict = np.array([[0.484499991, 1.0, 2.0, 0.0], [0.475000024, 3.0, 4.0, 0.0], [0.488499999, 5.0, 6.0, 0.0], [0.440500021, 7.0, 8.0, 0.0], [-0.139263272, 0.0, 0.0, 0.0], [0.138643518, 0.0, 0.0, 0.0], [0.510499954, 9.0, 10.0, 0.0], [0.425500005, 11.0, 12.0, 0.0], [0.453000009, 13.0, 14.0, 0.0], [0.507500052, 15.0, 16.0, 0.0], [0.514500022, 17.0, 18.0, 0.0], [0.000176883565, 0.0, 0.0, 0.0], [-0.101872563, 0.0, 0.0, 0.0], [0.093750909, 0.0, 0.0, 0.0], [-0.0264301933, 0.0, 0.0, 0.0], [0.0297804214, 0.0, 0.0, 0.0], [-0.184300154, 0.0, 0.0, 0.0], [0.192420229, 0.0, 0.0, 0.0], [-0.000459292438, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_141(xs):
    #Predicts Class 1
    function_dict = np.array([[0.484499991, 1.0, 2.0, 0.0], [0.475000024, 3.0, 4.0, 0.0], [0.488499999, 5.0, 6.0, 0.0], [0.440500021, 7.0, 8.0, 0.0], [0.139263302, 0.0, 0.0, 0.0], [-0.138643518, 0.0, 0.0, 0.0], [0.510499954, 9.0, 10.0, 0.0], [0.425500005, 11.0, 12.0, 0.0], [0.453000009, 13.0, 14.0, 0.0], [0.507500052, 15.0, 16.0, 0.0], [0.514500022, 17.0, 18.0, 0.0], [-0.00017690807, 0.0, 0.0, 0.0], [0.101872534, 0.0, 0.0, 0.0], [-0.0937509313, 0.0, 0.0, 0.0], [0.0264302008, 0.0, 0.0, 0.0], [-0.0297804158, 0.0, 0.0, 0.0], [0.184300154, 0.0, 0.0, 0.0], [-0.192420244, 0.0, 0.0, 0.0], [0.000459295406, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_142(xs):
    #Predicts Class 0
    function_dict = np.array([[0.547500014, 1.0, 2.0, 0.0], [0.54550004, 3.0, 4.0, 0.0], [0.559000015, 5.0, 6.0, 0.0], [0.537, 7.0, 8.0, 0.0], [-0.170695096, 0.0, 0.0, 0.0], [0.553499997, 9.0, 10.0, 0.0], [0.570500016, 11.0, 12.0, 0.0], [0.514500022, 13.0, 14.0, 0.0], [0.110266633, 0.0, 0.0, 0.0], [0.0201206822, 0.0, 0.0, 0.0], [0.134630039, 0.0, 0.0, 0.0], [0.5, 15.0, 16.0, 1.0], [0.582000017, 17.0, 18.0, 0.0], [0.000271639612, 0.0, 0.0, 0.0], [-0.161486223, 0.0, 0.0, 0.0], [-0.111964971, 0.0, 0.0, 0.0], [-0.0216615349, 0.0, 0.0, 0.0], [0.0688994601, 0.0, 0.0, 0.0], [0.000903266657, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 4, 9, 10, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 2, 5, 6, 11, 12])
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
    function_dict = np.array([[0.547500014, 1.0, 2.0, 0.0], [0.54550004, 3.0, 4.0, 0.0], [0.559000015, 5.0, 6.0, 0.0], [0.537, 7.0, 8.0, 0.0], [0.170695096, 0.0, 0.0, 0.0], [0.553499997, 9.0, 10.0, 0.0], [0.570500016, 11.0, 12.0, 0.0], [0.514500022, 13.0, 14.0, 0.0], [-0.110266663, 0.0, 0.0, 0.0], [-0.0201206543, 0.0, 0.0, 0.0], [-0.134630024, 0.0, 0.0, 0.0], [0.5, 15.0, 16.0, 1.0], [0.582000017, 17.0, 18.0, 0.0], [-0.000271624391, 0.0, 0.0, 0.0], [0.161486283, 0.0, 0.0, 0.0], [0.111964956, 0.0, 0.0, 0.0], [0.0216615088, 0.0, 0.0, 0.0], [-0.0688994527, 0.0, 0.0, 0.0], [-0.000903266482, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 4, 9, 10, 15, 16, 17, 18])
    branch_indices = np.array([0, 1, 3, 7, 2, 5, 6, 11, 12])
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
def booster_144(xs):
    #Predicts Class 0
    function_dict = np.array([[0.863499999, 1.0, 2.0, 0.0], [0.851999998, 3.0, 4.0, 0.0], [0.921499968, 5.0, 6.0, 0.0], [0.823500037, 7.0, 8.0, 0.0], [0.858500004, 9.0, 10.0, 0.0], [0.0847660378, 0.0, 0.0, 0.0], [-0.0370973386, 0.0, 0.0, 0.0], [0.817499995, 11.0, 12.0, 0.0], [0.833000004, 13.0, 14.0, 0.0], [-0.0760662034, 0.0, 0.0, 0.0], [-0.0187627152, 0.0, 0.0, 0.0], [-3.93329792e-05, 0.0, 0.0, 0.0], [-0.121649399, 0.0, 0.0, 0.0], [0.102304444, 0.0, 0.0, 0.0], [-0.004966137, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2])
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
def booster_145(xs):
    #Predicts Class 1
    function_dict = np.array([[0.863499999, 1.0, 2.0, 0.0], [0.851999998, 3.0, 4.0, 0.0], [0.921499968, 5.0, 6.0, 0.0], [0.823500037, 7.0, 8.0, 0.0], [0.858500004, 9.0, 10.0, 0.0], [-0.0847660378, 0.0, 0.0, 0.0], [0.0370973386, 0.0, 0.0, 0.0], [0.817499995, 11.0, 12.0, 0.0], [0.833000004, 13.0, 14.0, 0.0], [0.0760662258, 0.0, 0.0, 0.0], [0.0187626947, 0.0, 0.0, 0.0], [3.93363298e-05, 0.0, 0.0, 0.0], [0.121649422, 0.0, 0.0, 0.0], [-0.102304466, 0.0, 0.0, 0.0], [0.00496610766, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2])
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
def booster_146(xs):
    #Predicts Class 0
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.878499985, 3.0, 4.0, 0.0], [0.0476141237, 0.0, 0.0, 0.0], [0.796000004, 5.0, 6.0, 0.0], [-0.0758063421, 0.0, 0.0, 0.0], [0.789499998, 7.0, 8.0, 0.0], [0.82249999, 9.0, 10.0, 0.0], [-0.000928941125, 0.0, 0.0, 0.0], [-0.0757497922, 0.0, 0.0, 0.0], [0.0852274448, 0.0, 0.0, 0.0], [-0.00761717744, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 4, 2])
    branch_indices = np.array([0, 1, 3, 5, 6])
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
def booster_147(xs):
    #Predicts Class 1
    function_dict = np.array([[0.924000025, 1.0, 2.0, 0.0], [0.878499985, 3.0, 4.0, 0.0], [-0.0476140901, 0.0, 0.0, 0.0], [0.796000004, 5.0, 6.0, 0.0], [0.0758062527, 0.0, 0.0, 0.0], [0.789499998, 7.0, 8.0, 0.0], [0.82249999, 9.0, 10.0, 0.0], [0.000928934664, 0.0, 0.0, 0.0], [0.0757498518, 0.0, 0.0, 0.0], [-0.0852274448, 0.0, 0.0, 0.0], [0.0076172105, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 4, 2])
    branch_indices = np.array([0, 1, 3, 5, 6])
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
def booster_148(xs):
    #Predicts Class 0
    function_dict = np.array([[0.787500024, 1.0, 2.0, 0.0], [0.771000028, 3.0, 4.0, 0.0], [0.792999983, 5.0, 6.0, 0.0], [0.766499996, 7.0, 8.0, 0.0], [0.77700001, 9.0, 10.0, 0.0], [0.0861056671, 0.0, 0.0, 0.0], [0.834499955, 11.0, 12.0, 0.0], [0.75849998, 13.0, 14.0, 0.0], [0.117644534, 0.0, 0.0, 0.0], [-0.0941545591, 0.0, 0.0, 0.0], [0.784000039, 15.0, 16.0, 0.0], [0.82249999, 17.0, 18.0, 0.0], [0.862499952, 19.0, 20.0, 0.0], [-0.00100651686, 0.0, 0.0, 0.0], [-0.0673985556, 0.0, 0.0, 0.0], [0.00996327028, 0.0, 0.0, 0.0], [-0.066211693, 0.0, 0.0, 0.0], [0.0127302874, 0.0, 0.0, 0.0], [-0.0896756276, 0.0, 0.0, 0.0], [0.0748851299, 0.0, 0.0, 0.0], [-0.0119303269, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 9, 15, 16, 5, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2, 6, 11, 12])
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
def booster_149(xs):
    #Predicts Class 1
    function_dict = np.array([[0.787500024, 1.0, 2.0, 0.0], [0.771000028, 3.0, 4.0, 0.0], [0.792999983, 5.0, 6.0, 0.0], [0.766499996, 7.0, 8.0, 0.0], [0.77700001, 9.0, 10.0, 0.0], [-0.0861056745, 0.0, 0.0, 0.0], [0.834499955, 11.0, 12.0, 0.0], [0.75849998, 13.0, 14.0, 0.0], [-0.117644548, 0.0, 0.0, 0.0], [0.0941545442, 0.0, 0.0, 0.0], [0.784000039, 15.0, 16.0, 0.0], [0.82249999, 17.0, 18.0, 0.0], [0.862499952, 19.0, 20.0, 0.0], [0.00100650801, 0.0, 0.0, 0.0], [0.0673985407, 0.0, 0.0, 0.0], [-0.00996330939, 0.0, 0.0, 0.0], [0.066211693, 0.0, 0.0, 0.0], [-0.0127302865, 0.0, 0.0, 0.0], [0.0896756575, 0.0, 0.0, 0.0], [-0.0748851523, 0.0, 0.0, 0.0], [0.0119303064, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 9, 15, 16, 5, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2, 6, 11, 12])
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
    for booster_index in range(0,150,2):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def logit_class_1(xs):
    sum_of_leaf_values = np.zeros(xs.shape[0])
    for booster_index in range(1,150,2):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def classify(rows, return_probabilities=False):
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
    global pool
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
        model_cap=10
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
                class_i_indices = np.argwhere(y_true==class_i)
                not_class_i_indices = np.argwhere(y_true!=class_i)
                stats[int(class_i)]['TP'] = int(np.sum(y_pred[class_i_indices]==y_true[class_i_indices]))
                stats[int(class_i)]['FP'] = int(np.sum(y_pred[class_i_indices]!=y_true[class_i_indices]))
                stats[int(class_i)]['TN'] = int(np.sum(y_pred[not_class_i_indices]==y_true[not_class_i_indices]))
                stats[int(class_i)]['FN'] = int(np.sum(y_pred[not_class_i_indices]!=y_true[not_class_i_indices]))
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
