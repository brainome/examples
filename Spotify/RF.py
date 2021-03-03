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
# Invocation: btc train.csv -headerless -f RF -o RF.py -riskoverfit --yes
# Total compiler execution time: 0:00:28.29. Finished on: Mar-03-2021 20:02:07.
# This source code requires Python 3.
#
"""
Classifier Type:                    Random Forest
System Type:                         Binary classifier
Best-guess accuracy:                 56.62%
Overall Model accuracy:              100.00% (611/611 correct)
Overall Improvement over best guess: 43.38% (of possible 43.38%)
Model capacity (MEC):                12 bits
Generalization ratio:                50.26 bits/bit
Model efficiency:                    3.61%/parameter
System behavior
True Negatives:                      43.37% (265/611)
True Positives:                      56.63% (346/611)
False Negatives:                     0.00% (0/611)
False Positives:                     0.00% (0/611)
True Pos. Rate/Sensitivity/Recall:   1.00
True Neg. Rate/Specificity:          1.00
Precision:                           1.00
F-1 Measure:                         1.00
False Negative Rate/Miss Rate:       0.00
Critical Success Index:              1.00
Confusion Matrix:
 [43.37% 0.00%]
 [0.00% 56.63%]
Generalization index:                25.01
Percent of Data Memorized:           4.00%
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
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
num_attr = 17
n_classes = 2
transform_true = False

# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target=""
important_idxs=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[], trim=False):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target=""
    important_idxs=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
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
    function_dict = np.array([[0.601999998, 1.0, 2.0, 5.0], [-4.16450024, 3.0, 4.0, 8.0], [0.0484500006, 5.0, 6.0, 10.0], [246344.5, 7.0, 8.0, 15.0], [0.867500007, 9.0, 10.0, 6.0], [-6.60500002, 11.0, 12.0, 8.0], [0.713500023, 13.0, 14.0, 13.0], [0.285000026, 15.0, 16.0, 13.0], [-0.681724191, 0.0, 0.0, 0.0], [0.558500051, 17.0, 18.0, 5.0], [195946.5, 19.0, 20.0, 15.0], [1067.0, 21.0, 22.0, 0.0], [0.792500019, 23.0, 24.0, 13.0], [374938.5, 25.0, 26.0, 15.0], [-6.06900024, 27.0, 28.0, 8.0], [-0.0384944454, 0.0, 0.0, 0.0], [-0.50392729, 0.0, 0.0, 0.0], [0.611382365, 0.0, 0.0, 0.0], [-0.230966672, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [-0.590248168, 0.0, 0.0, 0.0], [-0.619963169, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [0.366829425, 0.0, 0.0, 0.0], [-0.277160019, 0.0, 0.0, 0.0], [0.518685162, 0.0, 0.0, 0.0], [-0.346450001, 0.0, 0.0, 0.0], [-0.362947643, 0.0, 0.0, 0.0], [0.247464299, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_1(xs):
    #Predicts Class 1
    function_dict = np.array([[0.601999998, 1.0, 2.0, 5.0], [-4.16450024, 3.0, 4.0, 8.0], [0.0484500006, 5.0, 6.0, 10.0], [246344.5, 7.0, 8.0, 15.0], [0.867500007, 9.0, 10.0, 6.0], [-6.60500002, 11.0, 12.0, 8.0], [0.713500023, 13.0, 14.0, 13.0], [0.285000026, 15.0, 16.0, 13.0], [0.681724191, 0.0, 0.0, 0.0], [0.558500051, 17.0, 18.0, 5.0], [195946.5, 19.0, 20.0, 15.0], [1067.0, 21.0, 22.0, 0.0], [0.792500019, 23.0, 24.0, 13.0], [374938.5, 25.0, 26.0, 15.0], [-6.06900024, 27.0, 28.0, 8.0], [0.0384944454, 0.0, 0.0, 0.0], [0.50392729, 0.0, 0.0, 0.0], [-0.611382365, 0.0, 0.0, 0.0], [0.230966672, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [0.590248168, 0.0, 0.0, 0.0], [0.619963169, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [-0.366829425, 0.0, 0.0, 0.0], [0.277160019, 0.0, 0.0, 0.0], [-0.518685162, 0.0, 0.0, 0.0], [0.346450001, 0.0, 0.0, 0.0], [0.362947643, 0.0, 0.0, 0.0], [-0.247464299, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_2(xs):
    #Predicts Class 0
    function_dict = np.array([[0.718999982, 1.0, 2.0, 5.0], [118.921997, 3.0, 4.0, 14.0], [-5.09249973, 5.0, 6.0, 8.0], [2807422980.0, 7.0, 8.0, 2.0], [129.970001, 9.0, 10.0, 14.0], [187493.5, 11.0, 12.0, 15.0], [172727.0, 13.0, 14.0, 15.0], [0.199000001, 15.0, 16.0, 10.0], [2907905280.0, 17.0, 18.0, 2.0], [0.542500019, 19.0, 20.0, 5.0], [0.585000038, 21.0, 22.0, 13.0], [0.0592500009, 23.0, 24.0, 10.0], [139.983002, 25.0, 26.0, 14.0], [160043.5, 27.0, 28.0, 15.0], [0.478077173, 0.0, 0.0, 0.0], [-0.492621601, 0.0, 0.0, 0.0], [0.045848608, 0.0, 0.0, 0.0], [0.37296176, 0.0, 0.0, 0.0], [-0.299951345, 0.0, 0.0, 0.0], [-0.178382486, 0.0, 0.0, 0.0], [0.389400542, 0.0, 0.0, 0.0], [-0.0315266885, 0.0, 0.0, 0.0], [-0.439035177, 0.0, 0.0, 0.0], [-0.116223052, 0.0, 0.0, 0.0], [0.437904239, 0.0, 0.0, 0.0], [0.0683603957, 0.0, 0.0, 0.0], [-0.508716047, 0.0, 0.0, 0.0], [0.42084375, 0.0, 0.0, 0.0], [-0.339120924, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.718999982, 1.0, 2.0, 5.0], [118.921997, 3.0, 4.0, 14.0], [-5.09249973, 5.0, 6.0, 8.0], [2807422980.0, 7.0, 8.0, 2.0], [129.970001, 9.0, 10.0, 14.0], [187493.5, 11.0, 12.0, 15.0], [172727.0, 13.0, 14.0, 15.0], [0.199000001, 15.0, 16.0, 10.0], [2907905280.0, 17.0, 18.0, 2.0], [0.542500019, 19.0, 20.0, 5.0], [0.585000038, 21.0, 22.0, 13.0], [0.0592500009, 23.0, 24.0, 10.0], [139.983002, 25.0, 26.0, 14.0], [160043.5, 27.0, 28.0, 15.0], [-0.478077173, 0.0, 0.0, 0.0], [0.492621601, 0.0, 0.0, 0.0], [-0.0458486304, 0.0, 0.0, 0.0], [-0.37296176, 0.0, 0.0, 0.0], [0.299951345, 0.0, 0.0, 0.0], [0.178382441, 0.0, 0.0, 0.0], [-0.389400542, 0.0, 0.0, 0.0], [0.0315266475, 0.0, 0.0, 0.0], [0.439035147, 0.0, 0.0, 0.0], [0.11622306, 0.0, 0.0, 0.0], [-0.437904269, 0.0, 0.0, 0.0], [-0.0683604032, 0.0, 0.0, 0.0], [0.508715987, 0.0, 0.0, 0.0], [-0.42084381, 0.0, 0.0, 0.0], [0.339120895, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[248529.0, 1.0, 2.0, 15.0], [0.538499951, 3.0, 4.0, 13.0], [-5.26249981, 5.0, 6.0, 8.0], [-12.4969997, 7.0, 8.0, 8.0], [0.180000007, 9.0, 10.0, 10.0], [296982304.0, 11.0, 12.0, 3.0], [0.939499974, 13.0, 14.0, 6.0], [-0.413311601, 0.0, 0.0, 0.0], [170.143997, 15.0, 16.0, 14.0], [0.5, 17.0, 18.0, 9.0], [3617449470.0, 19.0, 20.0, 3.0], [121.835999, 21.0, 22.0, 14.0], [246776096.0, 23.0, 24.0, 4.0], [0.0422499999, 25.0, 26.0, 10.0], [-0.360595345, 0.0, 0.0, 0.0], [0.260468841, 0.0, 0.0, 0.0], [-0.168126628, 0.0, 0.0, 0.0], [0.00326017477, 0.0, 0.0, 0.0], [-0.277202487, 0.0, 0.0, 0.0], [0.337625504, 0.0, 0.0, 0.0], [-0.172942504, 0.0, 0.0, 0.0], [-0.19142814, 0.0, 0.0, 0.0], [0.129236922, 0.0, 0.0, 0.0], [-0.129249215, 0.0, 0.0, 0.0], [-0.409604907, 0.0, 0.0, 0.0], [-0.29806453, 0.0, 0.0, 0.0], [0.351361811, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 14])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[248529.0, 1.0, 2.0, 15.0], [0.538499951, 3.0, 4.0, 13.0], [-5.26249981, 5.0, 6.0, 8.0], [-12.4969997, 7.0, 8.0, 8.0], [0.180000007, 9.0, 10.0, 10.0], [296982304.0, 11.0, 12.0, 3.0], [0.939499974, 13.0, 14.0, 6.0], [0.41331166, 0.0, 0.0, 0.0], [170.143997, 15.0, 16.0, 14.0], [0.5, 17.0, 18.0, 9.0], [3617449470.0, 19.0, 20.0, 3.0], [121.835999, 21.0, 22.0, 14.0], [246776096.0, 23.0, 24.0, 4.0], [0.0422499999, 25.0, 26.0, 10.0], [0.360595316, 0.0, 0.0, 0.0], [-0.260468841, 0.0, 0.0, 0.0], [0.168126658, 0.0, 0.0, 0.0], [-0.00326017942, 0.0, 0.0, 0.0], [0.277202487, 0.0, 0.0, 0.0], [-0.337625504, 0.0, 0.0, 0.0], [0.172942489, 0.0, 0.0, 0.0], [0.19142811, 0.0, 0.0, 0.0], [-0.129236951, 0.0, 0.0, 0.0], [0.1292492, 0.0, 0.0, 0.0], [0.409604907, 0.0, 0.0, 0.0], [0.2980645, 0.0, 0.0, 0.0], [-0.351361811, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 14])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.568499982, 1.0, 2.0, 5.0], [0.220499992, 3.0, 4.0, 13.0], [3818502660.0, 5.0, 6.0, 1.0], [-7.32800007, 7.0, 8.0, 8.0], [120.604996, 9.0, 10.0, 14.0], [0.854499996, 11.0, 12.0, 13.0], [0.331499994, 13.0, 14.0, 12.0], [-0.35823679, 0.0, 0.0, 0.0], [249655.5, 15.0, 16.0, 15.0], [-0.427274048, 0.0, 0.0, 0.0], [192293.5, 17.0, 18.0, 15.0], [3807763970.0, 19.0, 20.0, 4.0], [519033664.0, 21.0, 22.0, 1.0], [0.557500005, 23.0, 24.0, 13.0], [0.356351972, 0.0, 0.0, 0.0], [0.416382521, 0.0, 0.0, 0.0], [-0.12086498, 0.0, 0.0, 0.0], [0.12958914, 0.0, 0.0, 0.0], [-0.270960033, 0.0, 0.0, 0.0], [0.223562315, 0.0, 0.0, 0.0], [-0.211500123, 0.0, 0.0, 0.0], [-0.0744293928, 0.0, 0.0, 0.0], [-0.379971355, 0.0, 0.0, 0.0], [-0.121823177, 0.0, 0.0, 0.0], [-0.522463322, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 9, 17, 18, 19, 20, 21, 22, 23, 24, 14])
    branch_indices = np.array([0, 1, 3, 8, 4, 10, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.568499982, 1.0, 2.0, 5.0], [0.220499992, 3.0, 4.0, 13.0], [3818502660.0, 5.0, 6.0, 1.0], [-7.32800007, 7.0, 8.0, 8.0], [120.604996, 9.0, 10.0, 14.0], [0.854499996, 11.0, 12.0, 13.0], [0.331499994, 13.0, 14.0, 12.0], [0.35823679, 0.0, 0.0, 0.0], [249655.5, 15.0, 16.0, 15.0], [0.427274048, 0.0, 0.0, 0.0], [192293.5, 17.0, 18.0, 15.0], [3807763970.0, 19.0, 20.0, 4.0], [519033664.0, 21.0, 22.0, 1.0], [0.557500005, 23.0, 24.0, 13.0], [-0.356352031, 0.0, 0.0, 0.0], [-0.416382492, 0.0, 0.0, 0.0], [0.120865002, 0.0, 0.0, 0.0], [-0.129589185, 0.0, 0.0, 0.0], [0.270960033, 0.0, 0.0, 0.0], [-0.223562315, 0.0, 0.0, 0.0], [0.211500138, 0.0, 0.0, 0.0], [0.0744293705, 0.0, 0.0, 0.0], [0.379971355, 0.0, 0.0, 0.0], [0.121823162, 0.0, 0.0, 0.0], [0.522463322, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 9, 17, 18, 19, 20, 21, 22, 23, 24, 14])
    branch_indices = np.array([0, 1, 3, 8, 4, 10, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-7.23799992, 1.0, 2.0, 8.0], [0.814499974, 3.0, 4.0, 5.0], [2788307460.0, 5.0, 6.0, 4.0], [0.315500021, 7.0, 8.0, 10.0], [248983.0, 9.0, 10.0, 15.0], [119.979996, 11.0, 12.0, 14.0], [-3.59500003, 13.0, 14.0, 8.0], [1471571710.0, 15.0, 16.0, 1.0], [0.251365304, 0.0, 0.0, 0.0], [0.342879742, 0.0, 0.0, 0.0], [-0.0810687914, 0.0, 0.0, 0.0], [0.875, 17.0, 18.0, 6.0], [227440.0, 19.0, 20.0, 15.0], [0.834499955, 21.0, 22.0, 5.0], [0.302951366, 0.0, 0.0, 0.0], [-0.0973739028, 0.0, 0.0, 0.0], [-0.412312776, 0.0, 0.0, 0.0], [0.00839760248, 0.0, 0.0, 0.0], [-0.422469527, 0.0, 0.0, 0.0], [0.327243805, 0.0, 0.0, 0.0], [-0.0123561155, 0.0, 0.0, 0.0], [-0.232517198, 0.0, 0.0, 0.0], [0.308586836, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_9(xs):
    #Predicts Class 1
    function_dict = np.array([[-7.23799992, 1.0, 2.0, 8.0], [0.814499974, 3.0, 4.0, 5.0], [2788307460.0, 5.0, 6.0, 4.0], [0.315500021, 7.0, 8.0, 10.0], [248983.0, 9.0, 10.0, 15.0], [119.979996, 11.0, 12.0, 14.0], [-3.59500003, 13.0, 14.0, 8.0], [1471571710.0, 15.0, 16.0, 1.0], [-0.251365334, 0.0, 0.0, 0.0], [-0.342879742, 0.0, 0.0, 0.0], [0.0810687765, 0.0, 0.0, 0.0], [0.875, 17.0, 18.0, 6.0], [227440.0, 19.0, 20.0, 15.0], [0.834499955, 21.0, 22.0, 5.0], [-0.302951396, 0.0, 0.0, 0.0], [0.0973738804, 0.0, 0.0, 0.0], [0.412312776, 0.0, 0.0, 0.0], [-0.00839760434, 0.0, 0.0, 0.0], [0.422469527, 0.0, 0.0, 0.0], [-0.327243805, 0.0, 0.0, 0.0], [0.012356109, 0.0, 0.0, 0.0], [0.232517183, 0.0, 0.0, 0.0], [-0.308586836, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_10(xs):
    #Predicts Class 0
    function_dict = np.array([[270730.5, 1.0, 2.0, 15.0], [0.978999972, 3.0, 4.0, 6.0], [1.5, 5.0, 6.0, 7.0], [0.200500011, 7.0, 8.0, 10.0], [0.0843999982, 9.0, 10.0, 10.0], [-5.38399982, 11.0, 12.0, 8.0], [529234848.0, 13.0, 14.0, 4.0], [0.147499993, 15.0, 16.0, 13.0], [870351232.0, 17.0, 18.0, 1.0], [0.639762282, 0.0, 0.0, 0.0], [0.0188902151, 0.0, 0.0, 0.0], [-0.0889052823, 0.0, 0.0, 0.0], [0.17772454, 0.0, 0.0, 0.0], [-0.0705835223, 0.0, 0.0, 0.0], [-0.342271984, 0.0, 0.0, 0.0], [0.251365036, 0.0, 0.0, 0.0], [-0.0668086186, 0.0, 0.0, 0.0], [0.423485279, 0.0, 0.0, 0.0], [0.0742940307, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_11(xs):
    #Predicts Class 1
    function_dict = np.array([[270730.5, 1.0, 2.0, 15.0], [0.978999972, 3.0, 4.0, 6.0], [1.5, 5.0, 6.0, 7.0], [0.200500011, 7.0, 8.0, 10.0], [0.0843999982, 9.0, 10.0, 10.0], [-5.38399982, 11.0, 12.0, 8.0], [529234848.0, 13.0, 14.0, 4.0], [0.147499993, 15.0, 16.0, 13.0], [870351232.0, 17.0, 18.0, 1.0], [-0.639762282, 0.0, 0.0, 0.0], [-0.0188902617, 0.0, 0.0, 0.0], [0.0889053345, 0.0, 0.0, 0.0], [-0.177724555, 0.0, 0.0, 0.0], [0.0705835, 0.0, 0.0, 0.0], [0.342271984, 0.0, 0.0, 0.0], [-0.251364976, 0.0, 0.0, 0.0], [0.0668086186, 0.0, 0.0, 0.0], [-0.423485279, 0.0, 0.0, 0.0], [-0.0742940456, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_12(xs):
    #Predicts Class 0
    function_dict = np.array([[0.568499982, 1.0, 2.0, 5.0], [2049163780.0, 3.0, 4.0, 3.0], [-5.45300007, 5.0, 6.0, 8.0], [2820529660.0, 7.0, 8.0, 1.0], [2133288190.0, 9.0, 10.0, 3.0], [0.5, 11.0, 12.0, 9.0], [-3.80949998, 13.0, 14.0, 8.0], [658206976.0, 15.0, 16.0, 1.0], [3343204860.0, 17.0, 18.0, 1.0], [0.347463012, 0.0, 0.0, 0.0], [-3.29050016, 19.0, 20.0, 8.0], [140.109497, 21.0, 22.0, 14.0], [703699328.0, 23.0, 24.0, 3.0], [0.584500015, 25.0, 26.0, 6.0], [-3.53299999, 27.0, 28.0, 8.0], [-0.0623692609, 0.0, 0.0, 0.0], [-0.394450337, 0.0, 0.0, 0.0], [0.26454103, 0.0, 0.0, 0.0], [-0.212720141, 0.0, 0.0, 0.0], [-0.119761601, 0.0, 0.0, 0.0], [0.208626106, 0.0, 0.0, 0.0], [0.172428936, 0.0, 0.0, 0.0], [-0.176760703, 0.0, 0.0, 0.0], [0.216142789, 0.0, 0.0, 0.0], [-0.231645688, 0.0, 0.0, 0.0], [-0.290252447, 0.0, 0.0, 0.0], [0.323529124, 0.0, 0.0, 0.0], [-0.511147857, 0.0, 0.0, 0.0], [0.12819007, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.568499982, 1.0, 2.0, 5.0], [2049163780.0, 3.0, 4.0, 3.0], [-5.45300007, 5.0, 6.0, 8.0], [2820529660.0, 7.0, 8.0, 1.0], [2133288190.0, 9.0, 10.0, 3.0], [0.5, 11.0, 12.0, 9.0], [-3.80949998, 13.0, 14.0, 8.0], [658206976.0, 15.0, 16.0, 1.0], [3343204860.0, 17.0, 18.0, 1.0], [-0.347463012, 0.0, 0.0, 0.0], [-3.29050016, 19.0, 20.0, 8.0], [140.109497, 21.0, 22.0, 14.0], [703699328.0, 23.0, 24.0, 3.0], [0.584500015, 25.0, 26.0, 6.0], [-3.53299999, 27.0, 28.0, 8.0], [0.0623692572, 0.0, 0.0, 0.0], [0.394450337, 0.0, 0.0, 0.0], [-0.26454103, 0.0, 0.0, 0.0], [0.212720096, 0.0, 0.0, 0.0], [0.119761601, 0.0, 0.0, 0.0], [-0.208626106, 0.0, 0.0, 0.0], [-0.172428921, 0.0, 0.0, 0.0], [0.176760703, 0.0, 0.0, 0.0], [-0.216142803, 0.0, 0.0, 0.0], [0.231645688, 0.0, 0.0, 0.0], [0.290252447, 0.0, 0.0, 0.0], [-0.323529124, 0.0, 0.0, 0.0], [0.511147857, 0.0, 0.0, 0.0], [-0.128190085, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.703000009, 1.0, 2.0, 13.0], [-12.4969997, 3.0, 4.0, 8.0], [0.111000001, 5.0, 6.0, 12.0], [-0.311500102, 0.0, 0.0, 0.0], [776679296.0, 7.0, 8.0, 1.0], [-5.08049965, 9.0, 10.0, 8.0], [0.687000036, 11.0, 12.0, 5.0], [0.042750001, 13.0, 14.0, 10.0], [1081878400.0, 15.0, 16.0, 1.0], [-0.407558918, 0.0, 0.0, 0.0], [-0.067661725, 0.0, 0.0, 0.0], [832.0, 17.0, 18.0, 0.0], [-6.14850044, 19.0, 20.0, 8.0], [-0.158852264, 0.0, 0.0, 0.0], [0.445430517, 0.0, 0.0, 0.0], [-0.298442125, 0.0, 0.0, 0.0], [0.0647733286, 0.0, 0.0, 0.0], [-0.301275074, 0.0, 0.0, 0.0], [0.0903189555, 0.0, 0.0, 0.0], [-0.0795812979, 0.0, 0.0, 0.0], [0.258653611, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 9, 10, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.703000009, 1.0, 2.0, 13.0], [-12.4969997, 3.0, 4.0, 8.0], [0.111000001, 5.0, 6.0, 12.0], [0.311500072, 0.0, 0.0, 0.0], [776679296.0, 7.0, 8.0, 1.0], [-5.08049965, 9.0, 10.0, 8.0], [0.687000036, 11.0, 12.0, 5.0], [0.042750001, 13.0, 14.0, 10.0], [1081878400.0, 15.0, 16.0, 1.0], [0.407558918, 0.0, 0.0, 0.0], [0.0676617026, 0.0, 0.0, 0.0], [832.0, 17.0, 18.0, 0.0], [-6.14850044, 19.0, 20.0, 8.0], [0.158852279, 0.0, 0.0, 0.0], [-0.445430517, 0.0, 0.0, 0.0], [0.298442096, 0.0, 0.0, 0.0], [-0.0647733212, 0.0, 0.0, 0.0], [0.301275074, 0.0, 0.0, 0.0], [-0.0903189704, 0.0, 0.0, 0.0], [0.0795813054, 0.0, 0.0, 0.0], [-0.258653611, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 13, 14, 15, 16, 9, 10, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 4, 7, 8, 2, 5, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.0382499993, 1.0, 2.0, 10.0], [4.515e-06, 3.0, 4.0, 11.0], [0.0605000034, 5.0, 6.0, 10.0], [0.674999952, 7.0, 8.0, 5.0], [1407637500.0, 9.0, 10.0, 1.0], [-9.0795002, 11.0, 12.0, 8.0], [0.125499994, 13.0, 14.0, 12.0], [0.621500015, 15.0, 16.0, 13.0], [0.049439501, 0.0, 0.0, 0.0], [208553.5, 17.0, 18.0, 15.0], [-6.82849979, 19.0, 20.0, 8.0], [-0.268371105, 0.0, 0.0, 0.0], [159.920013, 21.0, 22.0, 14.0], [0.728000045, 23.0, 24.0, 6.0], [0.85650003, 25.0, 26.0, 5.0], [-0.420345634, 0.0, 0.0, 0.0], [-0.0193105079, 0.0, 0.0, 0.0], [0.394437999, 0.0, 0.0, 0.0], [-0.0571150072, 0.0, 0.0, 0.0], [-0.29114449, 0.0, 0.0, 0.0], [0.114736035, 0.0, 0.0, 0.0], [0.243869513, 0.0, 0.0, 0.0], [-0.207438812, 0.0, 0.0, 0.0], [0.235947818, 0.0, 0.0, 0.0], [-0.0915774778, 0.0, 0.0, 0.0], [-0.178987354, 0.0, 0.0, 0.0], [0.265670151, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_17(xs):
    #Predicts Class 1
    function_dict = np.array([[0.0382499993, 1.0, 2.0, 10.0], [4.515e-06, 3.0, 4.0, 11.0], [0.0605000034, 5.0, 6.0, 10.0], [0.674999952, 7.0, 8.0, 5.0], [1407637500.0, 9.0, 10.0, 1.0], [-9.0795002, 11.0, 12.0, 8.0], [0.125499994, 13.0, 14.0, 12.0], [0.621500015, 15.0, 16.0, 13.0], [-0.0494394749, 0.0, 0.0, 0.0], [208553.5, 17.0, 18.0, 15.0], [-6.82849979, 19.0, 20.0, 8.0], [0.268371075, 0.0, 0.0, 0.0], [159.920013, 21.0, 22.0, 14.0], [0.728000045, 23.0, 24.0, 6.0], [0.85650003, 25.0, 26.0, 5.0], [0.420345634, 0.0, 0.0, 0.0], [0.0193105154, 0.0, 0.0, 0.0], [-0.394437999, 0.0, 0.0, 0.0], [0.0571150072, 0.0, 0.0, 0.0], [0.29114446, 0.0, 0.0, 0.0], [-0.114736021, 0.0, 0.0, 0.0], [-0.243869513, 0.0, 0.0, 0.0], [0.207438841, 0.0, 0.0, 0.0], [-0.235947832, 0.0, 0.0, 0.0], [0.0915775001, 0.0, 0.0, 0.0], [0.178987354, 0.0, 0.0, 0.0], [-0.26567018, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_18(xs):
    #Predicts Class 0
    function_dict = np.array([[138626.5, 1.0, 2.0, 15.0], [0.73300004, 3.0, 4.0, 6.0], [163786.5, 5.0, 6.0, 15.0], [-0.0290389955, 0.0, 0.0, 0.0], [-0.319082022, 0.0, 0.0, 0.0], [0.041650001, 7.0, 8.0, 10.0], [344307136.0, 9.0, 10.0, 2.0], [0.807999969, 11.0, 12.0, 6.0], [0.711000025, 13.0, 14.0, 13.0], [0.583999991, 15.0, 16.0, 6.0], [1787758850.0, 17.0, 18.0, 2.0], [-0.177035123, 0.0, 0.0, 0.0], [0.140980572, 0.0, 0.0, 0.0], [0.328561157, 0.0, 0.0, 0.0], [0.0428041741, 0.0, 0.0, 0.0], [-0.0526573472, 0.0, 0.0, 0.0], [0.257367224, 0.0, 0.0, 0.0], [-0.135018706, 0.0, 0.0, 0.0], [0.0221812297, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_19(xs):
    #Predicts Class 1
    function_dict = np.array([[138626.5, 1.0, 2.0, 15.0], [0.73300004, 3.0, 4.0, 6.0], [163786.5, 5.0, 6.0, 15.0], [0.029038908, 0.0, 0.0, 0.0], [0.319081992, 0.0, 0.0, 0.0], [0.041650001, 7.0, 8.0, 10.0], [344307136.0, 9.0, 10.0, 2.0], [0.807999969, 11.0, 12.0, 6.0], [0.711000025, 13.0, 14.0, 13.0], [0.583999991, 15.0, 16.0, 6.0], [1787758850.0, 17.0, 18.0, 2.0], [0.177035138, 0.0, 0.0, 0.0], [-0.140980616, 0.0, 0.0, 0.0], [-0.328561157, 0.0, 0.0, 0.0], [-0.0428041369, 0.0, 0.0, 0.0], [0.0526573882, 0.0, 0.0, 0.0], [-0.257367194, 0.0, 0.0, 0.0], [0.135018691, 0.0, 0.0, 0.0], [-0.0221812371, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_20(xs):
    #Predicts Class 0
    function_dict = np.array([[0.694000006, 1.0, 2.0, 13.0], [0.568499982, 3.0, 4.0, 5.0], [0.687000036, 5.0, 6.0, 5.0], [0.286500007, 7.0, 8.0, 13.0], [0.676499963, 9.0, 10.0, 6.0], [1063.0, 11.0, 12.0, 0.0], [-4.92150021, 13.0, 14.0, 8.0], [105.941002, 15.0, 16.0, 14.0], [0.9745, 17.0, 18.0, 6.0], [1.5, 19.0, 20.0, 7.0], [275177472.0, 21.0, 22.0, 4.0], [-0.30885756, 0.0, 0.0, 0.0], [0.0738963038, 0.0, 0.0, 0.0], [703685632.0, 23.0, 24.0, 3.0], [0.228119344, 0.0, 0.0, 0.0], [-0.13825886, 0.0, 0.0, 0.0], [0.166506052, 0.0, 0.0, 0.0], [-0.228033319, 0.0, 0.0, 0.0], [0.139150158, 0.0, 0.0, 0.0], [0.284846097, 0.0, 0.0, 0.0], [-0.108467579, 0.0, 0.0, 0.0], [-0.158691227, 0.0, 0.0, 0.0], [0.212728173, 0.0, 0.0, 0.0], [0.141326472, 0.0, 0.0, 0.0], [-0.188520044, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 11, 12, 23, 24, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.694000006, 1.0, 2.0, 13.0], [0.568499982, 3.0, 4.0, 5.0], [0.687000036, 5.0, 6.0, 5.0], [0.286500007, 7.0, 8.0, 13.0], [0.676499963, 9.0, 10.0, 6.0], [1063.0, 11.0, 12.0, 0.0], [-4.92150021, 13.0, 14.0, 8.0], [105.941002, 15.0, 16.0, 14.0], [0.9745, 17.0, 18.0, 6.0], [1.5, 19.0, 20.0, 7.0], [275177472.0, 21.0, 22.0, 4.0], [0.30885756, 0.0, 0.0, 0.0], [-0.0738963038, 0.0, 0.0, 0.0], [703685632.0, 23.0, 24.0, 3.0], [-0.228119329, 0.0, 0.0, 0.0], [0.138258919, 0.0, 0.0, 0.0], [-0.166505963, 0.0, 0.0, 0.0], [0.228033319, 0.0, 0.0, 0.0], [-0.139150113, 0.0, 0.0, 0.0], [-0.284846067, 0.0, 0.0, 0.0], [0.108467571, 0.0, 0.0, 0.0], [0.158691213, 0.0, 0.0, 0.0], [-0.212728202, 0.0, 0.0, 0.0], [-0.141326457, 0.0, 0.0, 0.0], [0.188520044, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 11, 12, 23, 24, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.102499999, 1.0, 2.0, 10.0], [0.0951499939, 3.0, 4.0, 10.0], [0.944499969, 5.0, 6.0, 6.0], [114.005005, 7.0, 8.0, 14.0], [-0.415661991, 0.0, 0.0, 0.0], [132816.0, 9.0, 10.0, 15.0], [0.476999998, 11.0, 12.0, 5.0], [821.0, 13.0, 14.0, 0.0], [0.692000031, 15.0, 16.0, 6.0], [-0.181474999, 0.0, 0.0, 0.0], [3807763970.0, 17.0, 18.0, 4.0], [-0.196523458, 0.0, 0.0, 0.0], [-0.0533081368, 0.0, 0.0, 0.0], [-0.255067706, 0.0, 0.0, 0.0], [0.0418726876, 0.0, 0.0, 0.0], [-0.181691021, 0.0, 0.0, 0.0], [0.0925088078, 0.0, 0.0, 0.0], [0.191154122, 0.0, 0.0, 0.0], [-0.0762105882, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 9, 17, 18, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 10, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.102499999, 1.0, 2.0, 10.0], [0.0951499939, 3.0, 4.0, 10.0], [0.944499969, 5.0, 6.0, 6.0], [114.005005, 7.0, 8.0, 14.0], [0.415661991, 0.0, 0.0, 0.0], [132816.0, 9.0, 10.0, 15.0], [0.476999998, 11.0, 12.0, 5.0], [821.0, 13.0, 14.0, 0.0], [0.692000031, 15.0, 16.0, 6.0], [0.181475043, 0.0, 0.0, 0.0], [3807763970.0, 17.0, 18.0, 4.0], [0.196523473, 0.0, 0.0, 0.0], [0.053308066, 0.0, 0.0, 0.0], [0.255067676, 0.0, 0.0, 0.0], [-0.041872751, 0.0, 0.0, 0.0], [0.181690991, 0.0, 0.0, 0.0], [-0.0925087929, 0.0, 0.0, 0.0], [-0.191154122, 0.0, 0.0, 0.0], [0.0762105882, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 9, 17, 18, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 10, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.834499955, 1.0, 2.0, 5.0], [0.146499991, 3.0, 4.0, 13.0], [0.101999998, 5.0, 6.0, 12.0], [0.356999993, 7.0, 8.0, 5.0], [0.830500007, 9.0, 10.0, 5.0], [0.00889251847, 0.0, 0.0, 0.0], [0.250678033, 0.0, 0.0, 0.0], [-0.075427413, 0.0, 0.0, 0.0], [0.259043664, 0.0, 0.0, 0.0], [259731.5, 11.0, 12.0, 15.0], [-0.323886335, 0.0, 0.0, 0.0], [-0.0149250515, 0.0, 0.0, 0.0], [-0.227768734, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 11, 12, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 9, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.834499955, 1.0, 2.0, 5.0], [0.146499991, 3.0, 4.0, 13.0], [0.101999998, 5.0, 6.0, 12.0], [0.356999993, 7.0, 8.0, 5.0], [0.830500007, 9.0, 10.0, 5.0], [-0.00889259949, 0.0, 0.0, 0.0], [-0.250678062, 0.0, 0.0, 0.0], [0.075427331, 0.0, 0.0, 0.0], [-0.259043634, 0.0, 0.0, 0.0], [259731.5, 11.0, 12.0, 15.0], [0.323886365, 0.0, 0.0, 0.0], [0.0149250617, 0.0, 0.0, 0.0], [0.227768645, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 11, 12, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 9, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.538499951, 1.0, 2.0, 13.0], [176573.0, 3.0, 4.0, 15.0], [0.590499997, 5.0, 6.0, 5.0], [0.278499991, 7.0, 8.0, 11.0], [-5.38599968, 9.0, 10.0, 8.0], [-0.264690369, 0.0, 0.0, 0.0], [0.272000015, 11.0, 12.0, 12.0], [0.291484505, 0.0, 0.0, 0.0], [-0.159366548, 0.0, 0.0, 0.0], [0.000222500006, 13.0, 14.0, 11.0], [0.882500052, 15.0, 16.0, 6.0], [-10.3544998, 17.0, 18.0, 8.0], [1293382660.0, 19.0, 20.0, 3.0], [-0.173365355, 0.0, 0.0, 0.0], [0.0829046294, 0.0, 0.0, 0.0], [0.205844656, 0.0, 0.0, 0.0], [-0.082823284, 0.0, 0.0, 0.0], [-0.191907331, 0.0, 0.0, 0.0], [0.0836174414, 0.0, 0.0, 0.0], [0.015960386, 0.0, 0.0, 0.0], [-0.254325181, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 13, 14, 15, 16, 5, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.538499951, 1.0, 2.0, 13.0], [176573.0, 3.0, 4.0, 15.0], [0.590499997, 5.0, 6.0, 5.0], [0.278499991, 7.0, 8.0, 11.0], [-5.38599968, 9.0, 10.0, 8.0], [0.26469034, 0.0, 0.0, 0.0], [0.272000015, 11.0, 12.0, 12.0], [-0.291484505, 0.0, 0.0, 0.0], [0.159366593, 0.0, 0.0, 0.0], [0.000222500006, 13.0, 14.0, 11.0], [0.882500052, 15.0, 16.0, 6.0], [-10.3544998, 17.0, 18.0, 8.0], [1293382660.0, 19.0, 20.0, 3.0], [0.17336534, 0.0, 0.0, 0.0], [-0.0829046443, 0.0, 0.0, 0.0], [-0.205844656, 0.0, 0.0, 0.0], [0.082823284, 0.0, 0.0, 0.0], [0.191907331, 0.0, 0.0, 0.0], [-0.083617419, 0.0, 0.0, 0.0], [-0.0159603748, 0.0, 0.0, 0.0], [0.254325211, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 13, 14, 15, 16, 5, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.0393499993, 1.0, 2.0, 10.0], [0.03455, 3.0, 4.0, 10.0], [259731.5, 5.0, 6.0, 15.0], [-5.11149979, 7.0, 8.0, 8.0], [-0.260349661, 0.0, 0.0, 0.0], [9.5, 9.0, 10.0, 7.0], [8.5, 11.0, 12.0, 7.0], [0.20449999, 13.0, 14.0, 12.0], [-4.46000004, 15.0, 16.0, 8.0], [1166.0, 17.0, 18.0, 0.0], [0.946500003, 19.0, 20.0, 6.0], [-0.0266668908, 0.0, 0.0, 0.0], [-0.213362545, 0.0, 0.0, 0.0], [-0.219831556, 0.0, 0.0, 0.0], [0.076959677, 0.0, 0.0, 0.0], [0.234808937, 0.0, 0.0, 0.0], [-0.0284389723, 0.0, 0.0, 0.0], [0.0359734148, 0.0, 0.0, 0.0], [-0.272211254, 0.0, 0.0, 0.0], [0.258094341, 0.0, 0.0, 0.0], [-0.00796594284, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 17, 18, 19, 20, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 9, 10, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.0393499993, 1.0, 2.0, 10.0], [0.03455, 3.0, 4.0, 10.0], [259731.5, 5.0, 6.0, 15.0], [-5.11149979, 7.0, 8.0, 8.0], [0.260349661, 0.0, 0.0, 0.0], [9.5, 9.0, 10.0, 7.0], [8.5, 11.0, 12.0, 7.0], [0.20449999, 13.0, 14.0, 12.0], [-4.46000004, 15.0, 16.0, 8.0], [1166.0, 17.0, 18.0, 0.0], [0.946500003, 19.0, 20.0, 6.0], [0.026666794, 0.0, 0.0, 0.0], [0.21336253, 0.0, 0.0, 0.0], [0.219831556, 0.0, 0.0, 0.0], [-0.0769597664, 0.0, 0.0, 0.0], [-0.234808967, 0.0, 0.0, 0.0], [0.0284389704, 0.0, 0.0, 0.0], [-0.0359734222, 0.0, 0.0, 0.0], [0.272211224, 0.0, 0.0, 0.0], [-0.2580944, 0.0, 0.0, 0.0], [0.00796593353, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 17, 18, 19, 20, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 9, 10, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[2634249220.0, 1.0, 2.0, 4.0], [119.979996, 3.0, 4.0, 14.0], [0.206, 5.0, 6.0, 10.0], [0.799499989, 7.0, 8.0, 6.0], [142.065491, 9.0, 10.0, 14.0], [-4.90400028, 11.0, 12.0, 8.0], [0.531499982, 13.0, 14.0, 13.0], [2.14499996e-05, 15.0, 16.0, 11.0], [4.40500025e-06, 17.0, 18.0, 11.0], [564157504.0, 19.0, 20.0, 1.0], [145.019989, 21.0, 22.0, 14.0], [740912640.0, 23.0, 24.0, 1.0], [0.636999965, 25.0, 26.0, 5.0], [0.2242008, 0.0, 0.0, 0.0], [-0.0342726111, 0.0, 0.0, 0.0], [0.10078226, 0.0, 0.0, 0.0], [-0.173803285, 0.0, 0.0, 0.0], [-0.285141557, 0.0, 0.0, 0.0], [-0.0244409312, 0.0, 0.0, 0.0], [-0.0703646019, 0.0, 0.0, 0.0], [0.23778519, 0.0, 0.0, 0.0], [-0.233077496, 0.0, 0.0, 0.0], [0.0539069287, 0.0, 0.0, 0.0], [0.101763569, 0.0, 0.0, 0.0], [-0.249662235, 0.0, 0.0, 0.0], [-0.0803969055, 0.0, 0.0, 0.0], [0.160401955, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[2634249220.0, 1.0, 2.0, 4.0], [119.979996, 3.0, 4.0, 14.0], [0.206, 5.0, 6.0, 10.0], [0.799499989, 7.0, 8.0, 6.0], [142.065491, 9.0, 10.0, 14.0], [-4.90400028, 11.0, 12.0, 8.0], [0.531499982, 13.0, 14.0, 13.0], [2.14499996e-05, 15.0, 16.0, 11.0], [4.40500025e-06, 17.0, 18.0, 11.0], [564157504.0, 19.0, 20.0, 1.0], [145.019989, 21.0, 22.0, 14.0], [740912640.0, 23.0, 24.0, 1.0], [0.636999965, 25.0, 26.0, 5.0], [-0.22420083, 0.0, 0.0, 0.0], [0.0342725739, 0.0, 0.0, 0.0], [-0.100782275, 0.0, 0.0, 0.0], [0.17380324, 0.0, 0.0, 0.0], [0.285141587, 0.0, 0.0, 0.0], [0.0244408809, 0.0, 0.0, 0.0], [0.0703645423, 0.0, 0.0, 0.0], [-0.237785235, 0.0, 0.0, 0.0], [0.233077481, 0.0, 0.0, 0.0], [-0.0539069436, 0.0, 0.0, 0.0], [-0.101763539, 0.0, 0.0, 0.0], [0.249662235, 0.0, 0.0, 0.0], [0.0803969875, 0.0, 0.0, 0.0], [-0.160401955, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.590999961, 1.0, 2.0, 13.0], [259751.5, 3.0, 4.0, 15.0], [0.5, 5.0, 6.0, 9.0], [3322597380.0, 7.0, 8.0, 3.0], [-0.167709753, 0.0, 0.0, 0.0], [131.038513, 9.0, 10.0, 14.0], [1068.5, 11.0, 12.0, 0.0], [3.58999969e-05, 13.0, 14.0, 11.0], [566626048.0, 15.0, 16.0, 4.0], [-6.046, 17.0, 18.0, 8.0], [-0.167565793, 0.0, 0.0, 0.0], [3415680000.0, 19.0, 20.0, 1.0], [1107.0, 21.0, 22.0, 0.0], [-0.068847768, 0.0, 0.0, 0.0], [0.140859649, 0.0, 0.0, 0.0], [-0.0323853679, 0.0, 0.0, 0.0], [0.191207618, 0.0, 0.0, 0.0], [-0.0128253167, 0.0, 0.0, 0.0], [0.224549383, 0.0, 0.0, 0.0], [-0.255766064, 0.0, 0.0, 0.0], [-0.0268510412, 0.0, 0.0, 0.0], [0.1876477, 0.0, 0.0, 0.0], [-0.0319960713, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 17, 18, 10, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 9, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.590999961, 1.0, 2.0, 13.0], [259751.5, 3.0, 4.0, 15.0], [0.5, 5.0, 6.0, 9.0], [3322597380.0, 7.0, 8.0, 3.0], [0.167709723, 0.0, 0.0, 0.0], [131.038513, 9.0, 10.0, 14.0], [1068.5, 11.0, 12.0, 0.0], [3.58999969e-05, 13.0, 14.0, 11.0], [566626048.0, 15.0, 16.0, 4.0], [-6.046, 17.0, 18.0, 8.0], [0.167565882, 0.0, 0.0, 0.0], [3415680000.0, 19.0, 20.0, 1.0], [1107.0, 21.0, 22.0, 0.0], [0.0688477606, 0.0, 0.0, 0.0], [-0.140859678, 0.0, 0.0, 0.0], [0.0323853642, 0.0, 0.0, 0.0], [-0.191207603, 0.0, 0.0, 0.0], [0.0128253354, 0.0, 0.0, 0.0], [-0.224549457, 0.0, 0.0, 0.0], [0.255766094, 0.0, 0.0, 0.0], [0.0268510915, 0.0, 0.0, 0.0], [-0.18764773, 0.0, 0.0, 0.0], [0.0319960415, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 4, 17, 18, 10, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 9, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.640499949, 1.0, 2.0, 5.0], [0.220499992, 3.0, 4.0, 13.0], [-6.36999989, 5.0, 6.0, 8.0], [246344.5, 7.0, 8.0, 15.0], [0.5, 9.0, 10.0, 9.0], [-8.45949936, 11.0, 12.0, 8.0], [3819063810.0, 13.0, 14.0, 1.0], [4.5, 15.0, 16.0, 7.0], [-0.0736362711, 0.0, 0.0, 0.0], [125.913498, 17.0, 18.0, 14.0], [3684321790.0, 19.0, 20.0, 1.0], [200488.5, 21.0, 22.0, 15.0], [3123568640.0, 23.0, 24.0, 2.0], [0.745999992, 25.0, 26.0, 13.0], [-0.094241716, 0.0, 0.0, 0.0], [0.0109031647, 0.0, 0.0, 0.0], [0.22578223, 0.0, 0.0, 0.0], [-0.182446152, 0.0, 0.0, 0.0], [0.171206698, 0.0, 0.0, 0.0], [-0.2263197, 0.0, 0.0, 0.0], [0.0692935288, 0.0, 0.0, 0.0], [0.16897656, 0.0, 0.0, 0.0], [-0.000343900261, 0.0, 0.0, 0.0], [-0.182881683, 0.0, 0.0, 0.0], [0.0514951497, 0.0, 0.0, 0.0], [0.207709551, 0.0, 0.0, 0.0], [-0.102455005, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 14])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.640499949, 1.0, 2.0, 5.0], [0.220499992, 3.0, 4.0, 13.0], [-6.36999989, 5.0, 6.0, 8.0], [246344.5, 7.0, 8.0, 15.0], [0.5, 9.0, 10.0, 9.0], [-8.45949936, 11.0, 12.0, 8.0], [3819063810.0, 13.0, 14.0, 1.0], [4.5, 15.0, 16.0, 7.0], [0.0736362636, 0.0, 0.0, 0.0], [125.913498, 17.0, 18.0, 14.0], [3684321790.0, 19.0, 20.0, 1.0], [200488.5, 21.0, 22.0, 15.0], [3123568640.0, 23.0, 24.0, 2.0], [0.745999992, 25.0, 26.0, 13.0], [0.0942417085, 0.0, 0.0, 0.0], [-0.0109031247, 0.0, 0.0, 0.0], [-0.225782216, 0.0, 0.0, 0.0], [0.182446107, 0.0, 0.0, 0.0], [-0.171206683, 0.0, 0.0, 0.0], [0.226319656, 0.0, 0.0, 0.0], [-0.069293499, 0.0, 0.0, 0.0], [-0.168976471, 0.0, 0.0, 0.0], [0.000343767169, 0.0, 0.0, 0.0], [0.182881713, 0.0, 0.0, 0.0], [-0.0514951646, 0.0, 0.0, 0.0], [-0.207709536, 0.0, 0.0, 0.0], [0.102454968, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 14])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 10, 2, 5, 11, 12, 6, 13])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.342999995, 1.0, 2.0, 5.0], [-0.173624307, 0.0, 0.0, 0.0], [0.146499991, 3.0, 4.0, 13.0], [0.187545627, 0.0, 0.0, 0.0], [0.248999998, 5.0, 6.0, 13.0], [122.226501, 7.0, 8.0, 14.0], [3818502660.0, 9.0, 10.0, 1.0], [0.0702245831, 0.0, 0.0, 0.0], [-0.207410261, 0.0, 0.0, 0.0], [0.0594623424, 0.0, 0.0, 0.0], [-0.101446666, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_37(xs):
    #Predicts Class 1
    function_dict = np.array([[0.342999995, 1.0, 2.0, 5.0], [0.173624456, 0.0, 0.0, 0.0], [0.146499991, 3.0, 4.0, 13.0], [-0.187545627, 0.0, 0.0, 0.0], [0.248999998, 5.0, 6.0, 13.0], [122.226501, 7.0, 8.0, 14.0], [3818502660.0, 9.0, 10.0, 1.0], [-0.0702245608, 0.0, 0.0, 0.0], [0.207410291, 0.0, 0.0, 0.0], [-0.0594623312, 0.0, 0.0, 0.0], [0.101446614, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_38(xs):
    #Predicts Class 0
    function_dict = np.array([[0.205000013, 1.0, 2.0, 10.0], [-8.26399994, 3.0, 4.0, 8.0], [196728.0, 5.0, 6.0, 15.0], [2011063300.0, 7.0, 8.0, 1.0], [3410516990.0, 9.0, 10.0, 2.0], [1245590660.0, 11.0, 12.0, 1.0], [-0.0547494106, 0.0, 0.0, 0.0], [-0.0375558808, 0.0, 0.0, 0.0], [-0.203867316, 0.0, 0.0, 0.0], [0.557500005, 13.0, 14.0, 13.0], [0.906499982, 15.0, 16.0, 6.0], [-0.00348039111, 0.0, 0.0, 0.0], [172160.0, 17.0, 18.0, 15.0], [0.0936264917, 0.0, 0.0, 0.0], [-0.0783103704, 0.0, 0.0, 0.0], [-0.168456987, 0.0, 0.0, 0.0], [0.0894342959, 0.0, 0.0, 0.0], [0.0673878416, 0.0, 0.0, 0.0], [0.24556683, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 13, 14, 15, 16, 11, 17, 18, 6])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.205000013, 1.0, 2.0, 10.0], [-8.26399994, 3.0, 4.0, 8.0], [196728.0, 5.0, 6.0, 15.0], [2011063300.0, 7.0, 8.0, 1.0], [3410516990.0, 9.0, 10.0, 2.0], [1245590660.0, 11.0, 12.0, 1.0], [0.0547493882, 0.0, 0.0, 0.0], [0.0375557356, 0.0, 0.0, 0.0], [0.203867346, 0.0, 0.0, 0.0], [0.557500005, 13.0, 14.0, 13.0], [0.906499982, 15.0, 16.0, 6.0], [0.0034803336, 0.0, 0.0, 0.0], [172160.0, 17.0, 18.0, 15.0], [-0.0936264694, 0.0, 0.0, 0.0], [0.0783103332, 0.0, 0.0, 0.0], [0.168456972, 0.0, 0.0, 0.0], [-0.0894343033, 0.0, 0.0, 0.0], [-0.0673878789, 0.0, 0.0, 0.0], [-0.24556683, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 13, 14, 15, 16, 11, 17, 18, 6])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.553499997, 1.0, 2.0, 5.0], [2049163780.0, 3.0, 4.0, 3.0], [145.582504, 5.0, 6.0, 14.0], [1733276930.0, 7.0, 8.0, 4.0], [0.429499984, 9.0, 10.0, 13.0], [140.196991, 11.0, 12.0, 14.0], [2747083780.0, 13.0, 14.0, 3.0], [-0.0475952663, 0.0, 0.0, 0.0], [-0.215425178, 0.0, 0.0, 0.0], [3.5, 15.0, 16.0, 7.0], [-0.15114744, 0.0, 0.0, 0.0], [-3.82500005, 17.0, 18.0, 8.0], [0.728999972, 19.0, 20.0, 6.0], [0.222742885, 0.0, 0.0, 0.0], [0.0397504866, 0.0, 0.0, 0.0], [-0.0568294898, 0.0, 0.0, 0.0], [0.141980499, 0.0, 0.0, 0.0], [0.0516689792, 0.0, 0.0, 0.0], [-0.0886549652, 0.0, 0.0, 0.0], [-0.239832968, 0.0, 0.0, 0.0], [0.0103289979, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 10, 17, 18, 19, 20, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 9, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.553499997, 1.0, 2.0, 5.0], [2049163780.0, 3.0, 4.0, 3.0], [145.582504, 5.0, 6.0, 14.0], [1733276930.0, 7.0, 8.0, 4.0], [0.429499984, 9.0, 10.0, 13.0], [140.196991, 11.0, 12.0, 14.0], [2747083780.0, 13.0, 14.0, 3.0], [0.0475951768, 0.0, 0.0, 0.0], [0.215425104, 0.0, 0.0, 0.0], [3.5, 15.0, 16.0, 7.0], [0.151147366, 0.0, 0.0, 0.0], [-3.82500005, 17.0, 18.0, 8.0], [0.728999972, 19.0, 20.0, 6.0], [-0.222742856, 0.0, 0.0, 0.0], [-0.039750576, 0.0, 0.0, 0.0], [0.056829568, 0.0, 0.0, 0.0], [-0.141980514, 0.0, 0.0, 0.0], [-0.0516690165, 0.0, 0.0, 0.0], [0.088654995, 0.0, 0.0, 0.0], [0.239832968, 0.0, 0.0, 0.0], [-0.0103290081, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 10, 17, 18, 19, 20, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 9, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-5.10449982, 1.0, 2.0, 8.0], [515575168.0, 3.0, 4.0, 2.0], [0.879999995, 5.0, 6.0, 6.0], [196448.5, 7.0, 8.0, 15.0], [2594460670.0, 9.0, 10.0, 4.0], [192683.0, 11.0, 12.0, 15.0], [170545.0, 13.0, 14.0, 15.0], [0.150934055, 0.0, 0.0, 0.0], [0.0014915606, 0.0, 0.0, 0.0], [970291328.0, 15.0, 16.0, 4.0], [0.0901499987, 17.0, 18.0, 12.0], [-0.00944601558, 0.0, 0.0, 0.0], [0.21499984, 0.0, 0.0, 0.0], [0.0969678834, 0.0, 0.0, 0.0], [-3.59000015, 19.0, 20.0, 8.0], [-0.11516095, 0.0, 0.0, 0.0], [0.0403272957, 0.0, 0.0, 0.0], [0.0346566923, 0.0, 0.0, 0.0], [-0.192628965, 0.0, 0.0, 0.0], [-0.178340837, 0.0, 0.0, 0.0], [0.0135343401, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 11, 12, 13, 19, 20])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-5.10449982, 1.0, 2.0, 8.0], [515575168.0, 3.0, 4.0, 2.0], [0.879999995, 5.0, 6.0, 6.0], [196448.5, 7.0, 8.0, 15.0], [2594460670.0, 9.0, 10.0, 4.0], [192683.0, 11.0, 12.0, 15.0], [170545.0, 13.0, 14.0, 15.0], [-0.150933981, 0.0, 0.0, 0.0], [-0.00149157748, 0.0, 0.0, 0.0], [970291328.0, 15.0, 16.0, 4.0], [0.0901499987, 17.0, 18.0, 12.0], [0.00944594573, 0.0, 0.0, 0.0], [-0.21499978, 0.0, 0.0, 0.0], [-0.0969679132, 0.0, 0.0, 0.0], [-3.59000015, 19.0, 20.0, 8.0], [0.115161009, 0.0, 0.0, 0.0], [-0.0403272919, 0.0, 0.0, 0.0], [-0.0346567295, 0.0, 0.0, 0.0], [0.192628965, 0.0, 0.0, 0.0], [0.178340733, 0.0, 0.0, 0.0], [-0.0135343205, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 11, 12, 13, 19, 20])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.568499982, 1.0, 2.0, 5.0], [165130.5, 3.0, 4.0, 15.0], [0.703000009, 5.0, 6.0, 13.0], [0.0788959861, 0.0, 0.0, 0.0], [1500273660.0, 7.0, 8.0, 3.0], [1170.0, 9.0, 10.0, 0.0], [0.784999967, 11.0, 12.0, 5.0], [-0.191111043, 0.0, 0.0, 0.0], [0.0426499993, 13.0, 14.0, 10.0], [0.122500002, 15.0, 16.0, 12.0], [-0.102115542, 0.0, 0.0, 0.0], [1293118590.0, 17.0, 18.0, 3.0], [0.0467052609, 0.0, 0.0, 0.0], [-0.148639321, 0.0, 0.0, 0.0], [0.029236611, 0.0, 0.0, 0.0], [0.163205594, 0.0, 0.0, 0.0], [0.000654516276, 0.0, 0.0, 0.0], [0.000807535427, 0.0, 0.0, 0.0], [-0.1622563, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 13, 14, 15, 16, 10, 17, 18, 12])
    branch_indices = np.array([0, 1, 4, 8, 2, 5, 9, 6, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.568499982, 1.0, 2.0, 5.0], [165130.5, 3.0, 4.0, 15.0], [0.703000009, 5.0, 6.0, 13.0], [-0.078896068, 0.0, 0.0, 0.0], [1500273660.0, 7.0, 8.0, 3.0], [1170.0, 9.0, 10.0, 0.0], [0.784999967, 11.0, 12.0, 5.0], [0.191111028, 0.0, 0.0, 0.0], [0.0426499993, 13.0, 14.0, 10.0], [0.122500002, 15.0, 16.0, 12.0], [0.102115601, 0.0, 0.0, 0.0], [1293118590.0, 17.0, 18.0, 3.0], [-0.0467052869, 0.0, 0.0, 0.0], [0.148639277, 0.0, 0.0, 0.0], [-0.0292365626, 0.0, 0.0, 0.0], [-0.163205624, 0.0, 0.0, 0.0], [-0.000654516567, 0.0, 0.0, 0.0], [-0.000807661447, 0.0, 0.0, 0.0], [0.162256271, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 13, 14, 15, 16, 10, 17, 18, 12])
    branch_indices = np.array([0, 1, 4, 8, 2, 5, 9, 6, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    for booster_index in range(0,46,2):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def logit_class_1(xs):
    sum_of_leaf_values = np.zeros(xs.shape[0])
    for booster_index in range(1,46,2):
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
