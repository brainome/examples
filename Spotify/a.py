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
# Invocation: btc spotify.csv -f RF --yes -ignorecolumns artist
# Total compiler execution time: 0:00:24.68. Finished on: Mar-08-2021 23:32:09.
# This source code requires Python 3.
#
"""
Classifier Type:                    Random Forest
System Type:                         Binary classifier
Training/Validation Split:           50:50%
Best-guess accuracy:                 56.61%
Training accuracy:                   100.00% (612/612 correct)
Validation accuracy:                 84.31% (516/612 correct)
Overall Model accuracy:              92.15% (1128/1224 correct)
Overall Improvement over best guess: 35.54% (of possible 43.39%)
Model capacity (MEC):                12 bits
Generalization ratio:                50.54 bits/bit
Model efficiency:                    2.96%/parameter
System behavior
True Negatives:                      39.87% (488/1224)
True Positives:                      52.29% (640/1224)
False Negatives:                     4.33% (53/1224)
False Positives:                     3.51% (43/1224)
True Pos. Rate/Sensitivity/Recall:   0.92
True Neg. Rate/Specificity:          0.92
Precision:                           0.94
F-1 Measure:                         0.93
False Negative Rate/Miss Rate:       0.08
Critical Success Index:              0.87
Confusion Matrix:
 [39.87% 3.51%]
 [4.33% 52.29%]
Generalization index:                25.15
Percent of Data Memorized:           3.98%
Note: Labels have been remapped to 'True'=0, 'False'=1.
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
TRAINFILE = "spotify.csv"

try:
    import numpy as np # For numpy see: http://numpy.org
    from numpy import array
except:
    print("This predictor requires the Numpy library. For installation instructions please refer to: http://numpy.org")

#Number of attributes
num_attr = 16
n_classes = 2
transform_true = False

# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=["artist",]
target=""
important_idxs=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[], trim=False):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=["artist",]
    target=""
    important_idxs=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
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
    clean.mapping={'True': 0, 'False': 1}

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
    function_dict = np.array([[0.555500031, 1.0, 2.0, 4.0], [0.272000015, 3.0, 4.0, 9.0], [0.0442500003, 5.0, 6.0, 9.0], [-7.03750038, 7.0, 8.0, 7.0], [396428096.0, 9.0, 10.0, 3.0], [119.237999, 11.0, 12.0, 13.0], [0.703000009, 13.0, 14.0, 12.0], [2.0, 15.0, 16.0, 15.0], [0.800500035, 17.0, 18.0, 5.0], [-0.371039987, 0.0, 0.0, 0.0], [180075.0, 19.0, 20.0, 14.0], [76.7680054, 21.0, 22.0, 13.0], [129.0215, 23.0, 24.0, 13.0], [88.4454956, 25.0, 26.0, 13.0], [0.774500012, 27.0, 28.0, 4.0], [-0.0, 0.0, 0.0, 0.0], [-0.607053161, 0.0, 0.0, 0.0], [0.018739393, 0.0, 0.0, 0.0], [-0.502449989, 0.0, 0.0, 0.0], [-0.123679996, 0.0, 0.0, 0.0], [0.441714287, 0.0, 0.0, 0.0], [0.309199989, 0.0, 0.0, 0.0], [-0.424392164, 0.0, 0.0, 0.0], [0.458812892, 0.0, 0.0, 0.0], [-0.337309092, 0.0, 0.0, 0.0], [0.0687111095, 0.0, 0.0, 0.0], [0.417837828, 0.0, 0.0, 0.0], [-0.154599994, 0.0, 0.0, 0.0], [0.566866636, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_1(xs):
    #Predicts Class 1
    function_dict = np.array([[0.555500031, 1.0, 2.0, 4.0], [0.272000015, 3.0, 4.0, 9.0], [0.0442500003, 5.0, 6.0, 9.0], [-7.03750038, 7.0, 8.0, 7.0], [396428096.0, 9.0, 10.0, 3.0], [119.237999, 11.0, 12.0, 13.0], [0.703000009, 13.0, 14.0, 12.0], [2.0, 15.0, 16.0, 15.0], [0.800500035, 17.0, 18.0, 5.0], [0.371039987, 0.0, 0.0, 0.0], [180075.0, 19.0, 20.0, 14.0], [76.7680054, 21.0, 22.0, 13.0], [129.0215, 23.0, 24.0, 13.0], [88.4454956, 25.0, 26.0, 13.0], [0.774500012, 27.0, 28.0, 4.0], [-0.0, 0.0, 0.0, 0.0], [0.607053161, 0.0, 0.0, 0.0], [-0.018739393, 0.0, 0.0, 0.0], [0.502449989, 0.0, 0.0, 0.0], [0.123679996, 0.0, 0.0, 0.0], [-0.441714287, 0.0, 0.0, 0.0], [-0.309199989, 0.0, 0.0, 0.0], [0.424392164, 0.0, 0.0, 0.0], [-0.458812892, 0.0, 0.0, 0.0], [0.337309092, 0.0, 0.0, 0.0], [-0.0687111095, 0.0, 0.0, 0.0], [-0.417837828, 0.0, 0.0, 0.0], [0.154599994, 0.0, 0.0, 0.0], [-0.566866636, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_2(xs):
    #Predicts Class 0
    function_dict = np.array([[0.579999983, 1.0, 2.0, 4.0], [-5.93799973, 3.0, 4.0, 7.0], [244068.0, 5.0, 6.0, 14.0], [210133.0, 7.0, 8.0, 14.0], [0.800500035, 9.0, 10.0, 5.0], [-6.38800001, 11.0, 12.0, 7.0], [-9.11299992, 13.0, 14.0, 7.0], [0.868499994, 15.0, 16.0, 5.0], [74925456.0, 17.0, 18.0, 3.0], [0.0388999991, 19.0, 20.0, 9.0], [0.5, 21.0, 22.0, 8.0], [0.201000005, 23.0, 24.0, 9.0], [0.899999976, 25.0, 26.0, 12.0], [-0.550497413, 0.0, 0.0, 0.0], [0.736500025, 27.0, 28.0, 4.0], [-0.287842214, 0.0, 0.0, 0.0], [0.298743814, 0.0, 0.0, 0.0], [0.00916126464, 0.0, 0.0, 0.0], [-0.416207522, 0.0, 0.0, 0.0], [-0.251394868, 0.0, 0.0, 0.0], [0.362028211, 0.0, 0.0, 0.0], [-0.0368651301, 0.0, 0.0, 0.0], [-0.371120393, 0.0, 0.0, 0.0], [-0.0829101652, 0.0, 0.0, 0.0], [0.23708874, 0.0, 0.0, 0.0], [0.300966501, 0.0, 0.0, 0.0], [-0.295642853, 0.0, 0.0, 0.0], [-0.228602648, 0.0, 0.0, 0.0], [0.294711888, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 13, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 12, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.579999983, 1.0, 2.0, 4.0], [-5.93799973, 3.0, 4.0, 7.0], [244068.0, 5.0, 6.0, 14.0], [210133.0, 7.0, 8.0, 14.0], [0.800500035, 9.0, 10.0, 5.0], [-6.38800001, 11.0, 12.0, 7.0], [-9.11299992, 13.0, 14.0, 7.0], [0.868499994, 15.0, 16.0, 5.0], [74925456.0, 17.0, 18.0, 3.0], [0.0388999991, 19.0, 20.0, 9.0], [0.5, 21.0, 22.0, 8.0], [0.201000005, 23.0, 24.0, 9.0], [0.899999976, 25.0, 26.0, 12.0], [0.550497413, 0.0, 0.0, 0.0], [0.736500025, 27.0, 28.0, 4.0], [0.287842214, 0.0, 0.0, 0.0], [-0.298743844, 0.0, 0.0, 0.0], [-0.00916125625, 0.0, 0.0, 0.0], [0.416207582, 0.0, 0.0, 0.0], [0.251394898, 0.0, 0.0, 0.0], [-0.362028182, 0.0, 0.0, 0.0], [0.0368650854, 0.0, 0.0, 0.0], [0.371120393, 0.0, 0.0, 0.0], [0.0829101503, 0.0, 0.0, 0.0], [-0.237088755, 0.0, 0.0, 0.0], [-0.300966501, 0.0, 0.0, 0.0], [0.295642853, 0.0, 0.0, 0.0], [0.228602663, 0.0, 0.0, 0.0], [-0.294711888, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 13, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 12, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.733999968, 1.0, 2.0, 4.0], [-9.45599937, 3.0, 4.0, 7.0], [133.539505, 5.0, 6.0, 13.0], [0.155499995, 7.0, 8.0, 9.0], [0.559499979, 9.0, 10.0, 12.0], [0.761000037, 11.0, 12.0, 5.0], [152.466003, 13.0, 14.0, 13.0], [197866064.0, 15.0, 16.0, 2.0], [3.0, 17.0, 18.0, 6.0], [255015.0, 19.0, 20.0, 14.0], [-3.19899988, 21.0, 22.0, 7.0], [61.0, 23.0, 24.0, 0.0], [3440054270.0, 25.0, 26.0, 1.0], [0.235499993, 27.0, 28.0, 9.0], [0.303023905, 0.0, 0.0, 0.0], [-0.00989918225, 0.0, 0.0, 0.0], [-0.380994767, 0.0, 0.0, 0.0], [0.287946522, 0.0, 0.0, 0.0], [-0.278635353, 0.0, 0.0, 0.0], [0.136931837, 0.0, 0.0, 0.0], [-0.220957726, 0.0, 0.0, 0.0], [-0.266870201, 0.0, 0.0, 0.0], [0.292440712, 0.0, 0.0, 0.0], [-0.0812512413, 0.0, 0.0, 0.0], [0.325778723, 0.0, 0.0, 0.0], [0.200738385, 0.0, 0.0, 0.0], [-0.374728054, 0.0, 0.0, 0.0], [0.00900802575, 0.0, 0.0, 0.0], [-0.592496336, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_5(xs):
    #Predicts Class 1
    function_dict = np.array([[0.733999968, 1.0, 2.0, 4.0], [-9.45599937, 3.0, 4.0, 7.0], [133.539505, 5.0, 6.0, 13.0], [0.155499995, 7.0, 8.0, 9.0], [0.559499979, 9.0, 10.0, 12.0], [0.761000037, 11.0, 12.0, 5.0], [152.466003, 13.0, 14.0, 13.0], [197866064.0, 15.0, 16.0, 2.0], [3.0, 17.0, 18.0, 6.0], [255015.0, 19.0, 20.0, 14.0], [-3.19899988, 21.0, 22.0, 7.0], [61.0, 23.0, 24.0, 0.0], [3440054270.0, 25.0, 26.0, 1.0], [0.235499993, 27.0, 28.0, 9.0], [-0.303023934, 0.0, 0.0, 0.0], [0.00989919715, 0.0, 0.0, 0.0], [0.380994767, 0.0, 0.0, 0.0], [-0.287946522, 0.0, 0.0, 0.0], [0.278635353, 0.0, 0.0, 0.0], [-0.136931852, 0.0, 0.0, 0.0], [0.220957726, 0.0, 0.0, 0.0], [0.266870171, 0.0, 0.0, 0.0], [-0.292440772, 0.0, 0.0, 0.0], [0.0812512338, 0.0, 0.0, 0.0], [-0.325778723, 0.0, 0.0, 0.0], [-0.20073843, 0.0, 0.0, 0.0], [0.374727994, 0.0, 0.0, 0.0], [-0.00900806207, 0.0, 0.0, 0.0], [0.592496276, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_6(xs):
    #Predicts Class 0
    function_dict = np.array([[0.555500031, 1.0, 2.0, 4.0], [168495.5, 3.0, 4.0, 14.0], [-5.48500013, 5.0, 6.0, 7.0], [0.247999996, 7.0, 8.0, 11.0], [0.0419500023, 9.0, 10.0, 9.0], [0.71450001, 11.0, 12.0, 12.0], [0.752499998, 13.0, 14.0, 5.0], [0.5, 15.0, 16.0, 8.0], [-0.2770693, 0.0, 0.0, 0.0], [-0.357275516, 0.0, 0.0, 0.0], [0.0606499985, 17.0, 18.0, 9.0], [0.626000047, 19.0, 20.0, 12.0], [0.177000001, 21.0, 22.0, 9.0], [236616.0, 23.0, 24.0, 14.0], [10.5, 25.0, 26.0, 6.0], [0.472694397, 0.0, 0.0, 0.0], [-0.0368252993, 0.0, 0.0, 0.0], [0.0833782777, 0.0, 0.0, 0.0], [-0.235196799, 0.0, 0.0, 0.0], [-0.0164633151, 0.0, 0.0, 0.0], [0.316850811, 0.0, 0.0, 0.0], [-0.307484508, 0.0, 0.0, 0.0], [0.160493016, 0.0, 0.0, 0.0], [0.376898348, 0.0, 0.0, 0.0], [-0.104274541, 0.0, 0.0, 0.0], [0.148377836, 0.0, 0.0, 0.0], [-0.446148098, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_7(xs):
    #Predicts Class 1
    function_dict = np.array([[0.555500031, 1.0, 2.0, 4.0], [168495.5, 3.0, 4.0, 14.0], [-5.48500013, 5.0, 6.0, 7.0], [0.247999996, 7.0, 8.0, 11.0], [0.0419500023, 9.0, 10.0, 9.0], [0.71450001, 11.0, 12.0, 12.0], [0.752499998, 13.0, 14.0, 5.0], [0.5, 15.0, 16.0, 8.0], [0.2770693, 0.0, 0.0, 0.0], [0.357275516, 0.0, 0.0, 0.0], [0.0606499985, 17.0, 18.0, 9.0], [0.626000047, 19.0, 20.0, 12.0], [0.177000001, 21.0, 22.0, 9.0], [236616.0, 23.0, 24.0, 14.0], [10.5, 25.0, 26.0, 6.0], [-0.472694367, 0.0, 0.0, 0.0], [0.0368253402, 0.0, 0.0, 0.0], [-0.083378233, 0.0, 0.0, 0.0], [0.235196799, 0.0, 0.0, 0.0], [0.01646333, 0.0, 0.0, 0.0], [-0.316850811, 0.0, 0.0, 0.0], [0.307484478, 0.0, 0.0, 0.0], [-0.160493016, 0.0, 0.0, 0.0], [-0.376898348, 0.0, 0.0, 0.0], [0.104274519, 0.0, 0.0, 0.0], [-0.148377836, 0.0, 0.0, 0.0], [0.446148038, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_8(xs):
    #Predicts Class 0
    function_dict = np.array([[0.5, 1.0, 2.0, 8.0], [10.5, 3.0, 4.0, 6.0], [820.0, 5.0, 6.0, 0.0], [269077.0, 7.0, 8.0, 14.0], [763.0, 9.0, 10.0, 0.0], [0.808499992, 11.0, 12.0, 4.0], [0.792999983, 13.0, 14.0, 4.0], [-13.1929998, 15.0, 16.0, 7.0], [3.63499998e-06, 17.0, 18.0, 10.0], [207746.0, 19.0, 20.0, 14.0], [-0.47097224, 0.0, 0.0, 0.0], [119.400497, 21.0, 22.0, 13.0], [0.29017368, 0.0, 0.0, 0.0], [3364301060.0, 23.0, 24.0, 2.0], [1751500540.0, 25.0, 26.0, 1.0], [-0.277117908, 0.0, 0.0, 0.0], [0.202090263, 0.0, 0.0, 0.0], [0.104895249, 0.0, 0.0, 0.0], [-0.313208491, 0.0, 0.0, 0.0], [0.21323964, 0.0, 0.0, 0.0], [-0.198377848, 0.0, 0.0, 0.0], [-0.3480708, 0.0, 0.0, 0.0], [-0.0961636379, 0.0, 0.0, 0.0], [0.175496832, 0.0, 0.0, 0.0], [-0.119599208, 0.0, 0.0, 0.0], [-0.604116499, 0.0, 0.0, 0.0], [0.244154632, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 10, 21, 22, 12, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.5, 1.0, 2.0, 8.0], [10.5, 3.0, 4.0, 6.0], [820.0, 5.0, 6.0, 0.0], [269077.0, 7.0, 8.0, 14.0], [763.0, 9.0, 10.0, 0.0], [0.808499992, 11.0, 12.0, 4.0], [0.792999983, 13.0, 14.0, 4.0], [-13.1929998, 15.0, 16.0, 7.0], [3.63499998e-06, 17.0, 18.0, 10.0], [207746.0, 19.0, 20.0, 14.0], [0.47097224, 0.0, 0.0, 0.0], [119.400497, 21.0, 22.0, 13.0], [-0.29017368, 0.0, 0.0, 0.0], [3364301060.0, 23.0, 24.0, 2.0], [1751500540.0, 25.0, 26.0, 1.0], [0.277117908, 0.0, 0.0, 0.0], [-0.202090263, 0.0, 0.0, 0.0], [-0.104895256, 0.0, 0.0, 0.0], [0.313208491, 0.0, 0.0, 0.0], [-0.213239625, 0.0, 0.0, 0.0], [0.198377833, 0.0, 0.0, 0.0], [0.348070771, 0.0, 0.0, 0.0], [0.0961636379, 0.0, 0.0, 0.0], [-0.175496846, 0.0, 0.0, 0.0], [0.119599178, 0.0, 0.0, 0.0], [0.604116499, 0.0, 0.0, 0.0], [-0.244154632, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 10, 21, 22, 12, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.64349997, 1.0, 2.0, 4.0], [0.44600001, 3.0, 4.0, 12.0], [-3.64650011, 5.0, 6.0, 7.0], [214419.0, 7.0, 8.0, 14.0], [0.0829000026, 9.0, 10.0, 10.0], [3836128260.0, 11.0, 12.0, 2.0], [0.340326071, 0.0, 0.0, 0.0], [3705455360.0, 13.0, 14.0, 1.0], [-4.79400015, 15.0, 16.0, 7.0], [0.400000006, 17.0, 18.0, 9.0], [218174.0, 19.0, 20.0, 14.0], [418.5, 21.0, 22.0, 0.0], [0.538500011, 23.0, 24.0, 5.0], [0.244689077, 0.0, 0.0, 0.0], [-0.219250217, 0.0, 0.0, 0.0], [-0.294248402, 0.0, 0.0, 0.0], [0.121675193, 0.0, 0.0, 0.0], [-0.280293733, 0.0, 0.0, 0.0], [0.309919715, 0.0, 0.0, 0.0], [-0.0342388302, 0.0, 0.0, 0.0], [0.489859283, 0.0, 0.0, 0.0], [-0.125120088, 0.0, 0.0, 0.0], [0.0730811805, 0.0, 0.0, 0.0], [-0.0132277803, 0.0, 0.0, 0.0], [0.305527687, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.64349997, 1.0, 2.0, 4.0], [0.44600001, 3.0, 4.0, 12.0], [-3.64650011, 5.0, 6.0, 7.0], [214419.0, 7.0, 8.0, 14.0], [0.0829000026, 9.0, 10.0, 10.0], [3836128260.0, 11.0, 12.0, 2.0], [-0.340326071, 0.0, 0.0, 0.0], [3705455360.0, 13.0, 14.0, 1.0], [-4.79400015, 15.0, 16.0, 7.0], [0.400000006, 17.0, 18.0, 9.0], [218174.0, 19.0, 20.0, 14.0], [418.5, 21.0, 22.0, 0.0], [0.538500011, 23.0, 24.0, 5.0], [-0.244689122, 0.0, 0.0, 0.0], [0.219250172, 0.0, 0.0, 0.0], [0.294248402, 0.0, 0.0, 0.0], [-0.121675193, 0.0, 0.0, 0.0], [0.280293733, 0.0, 0.0, 0.0], [-0.309919715, 0.0, 0.0, 0.0], [0.034238819, 0.0, 0.0, 0.0], [-0.489859283, 0.0, 0.0, 0.0], [0.125120103, 0.0, 0.0, 0.0], [-0.0730811656, 0.0, 0.0, 0.0], [0.0132278129, 0.0, 0.0, 0.0], [-0.305527687, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[1.5, 1.0, 2.0, 6.0], [108.098495, 3.0, 4.0, 13.0], [193907712.0, 5.0, 6.0, 3.0], [85.7064972, 7.0, 8.0, 13.0], [-6.204, 9.0, 10.0, 7.0], [100.770996, 11.0, 12.0, 13.0], [244068.0, 13.0, 14.0, 14.0], [0.232150048, 0.0, 0.0, 0.0], [0.72299999, 15.0, 16.0, 4.0], [129.020508, 17.0, 18.0, 13.0], [158.502502, 19.0, 20.0, 13.0], [-0.118215151, 0.0, 0.0, 0.0], [0.0494000018, 21.0, 22.0, 9.0], [140.1185, 23.0, 24.0, 13.0], [138.94101, 25.0, 26.0, 13.0], [-0.2667211, 0.0, 0.0, 0.0], [0.106440574, 0.0, 0.0, 0.0], [0.163949847, 0.0, 0.0, 0.0], [-0.125569612, 0.0, 0.0, 0.0], [0.388565898, 0.0, 0.0, 0.0], [-0.00967870932, 0.0, 0.0, 0.0], [0.0616666749, 0.0, 0.0, 0.0], [0.364739835, 0.0, 0.0, 0.0], [0.0252985246, 0.0, 0.0, 0.0], [-0.160221994, 0.0, 0.0, 0.0], [-0.299589247, 0.0, 0.0, 0.0], [-0.0372155495, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 11, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[1.5, 1.0, 2.0, 6.0], [108.098495, 3.0, 4.0, 13.0], [193907712.0, 5.0, 6.0, 3.0], [85.7064972, 7.0, 8.0, 13.0], [-6.204, 9.0, 10.0, 7.0], [100.770996, 11.0, 12.0, 13.0], [244068.0, 13.0, 14.0, 14.0], [-0.232150048, 0.0, 0.0, 0.0], [0.72299999, 15.0, 16.0, 4.0], [129.020508, 17.0, 18.0, 13.0], [158.502502, 19.0, 20.0, 13.0], [0.118215144, 0.0, 0.0, 0.0], [0.0494000018, 21.0, 22.0, 9.0], [140.1185, 23.0, 24.0, 13.0], [138.94101, 25.0, 26.0, 13.0], [0.26672107, 0.0, 0.0, 0.0], [-0.106440581, 0.0, 0.0, 0.0], [-0.163949847, 0.0, 0.0, 0.0], [0.125569597, 0.0, 0.0, 0.0], [-0.388565898, 0.0, 0.0, 0.0], [0.0096787205, 0.0, 0.0, 0.0], [-0.0616667122, 0.0, 0.0, 0.0], [-0.364739776, 0.0, 0.0, 0.0], [-0.0252985246, 0.0, 0.0, 0.0], [0.160221994, 0.0, 0.0, 0.0], [0.299589247, 0.0, 0.0, 0.0], [0.0372155681, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 11, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.379500002, 1.0, 2.0, 9.0], [-7.23600006, 3.0, 4.0, 7.0], [0.313023984, 0.0, 0.0, 0.0], [-7.57750034, 5.0, 6.0, 7.0], [-6.74250031, 7.0, 8.0, 7.0], [0.771999955, 9.0, 10.0, 4.0], [0.695500016, 11.0, 12.0, 4.0], [0.5, 13.0, 14.0, 6.0], [0.300000012, 15.0, 16.0, 12.0], [-0.137585908, 0.0, 0.0, 0.0], [0.173950389, 0.0, 0.0, 0.0], [0.0200674906, 0.0, 0.0, 0.0], [-0.510375679, 0.0, 0.0, 0.0], [-0.0875119343, 0.0, 0.0, 0.0], [0.418270856, 0.0, 0.0, 0.0], [0.151642814, 0.0, 0.0, 0.0], [-0.0447393283, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 13, 14, 15, 16, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.379500002, 1.0, 2.0, 9.0], [-7.23600006, 3.0, 4.0, 7.0], [-0.313023984, 0.0, 0.0, 0.0], [-7.57750034, 5.0, 6.0, 7.0], [-6.74250031, 7.0, 8.0, 7.0], [0.771999955, 9.0, 10.0, 4.0], [0.695500016, 11.0, 12.0, 4.0], [0.5, 13.0, 14.0, 6.0], [0.300000012, 15.0, 16.0, 12.0], [0.137585893, 0.0, 0.0, 0.0], [-0.173950389, 0.0, 0.0, 0.0], [-0.0200674459, 0.0, 0.0, 0.0], [0.510375679, 0.0, 0.0, 0.0], [0.0875119343, 0.0, 0.0, 0.0], [-0.418270856, 0.0, 0.0, 0.0], [-0.151642814, 0.0, 0.0, 0.0], [0.0447393321, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 13, 14, 15, 16, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.547500014, 1.0, 2.0, 4.0], [0.190499991, 3.0, 4.0, 9.0], [0.930999994, 5.0, 6.0, 5.0], [616852608.0, 7.0, 8.0, 3.0], [121.735001, 9.0, 10.0, 13.0], [7.5, 11.0, 12.0, 6.0], [119.499001, 13.0, 14.0, 13.0], [0.5, 15.0, 16.0, 8.0], [0.802000046, 17.0, 18.0, 5.0], [-0.139406487, 0.0, 0.0, 0.0], [0.350829124, 0.0, 0.0, 0.0], [1.5, 19.0, 20.0, 6.0], [0.761000037, 21.0, 22.0, 5.0], [0.948500037, 23.0, 24.0, 5.0], [232062.5, 25.0, 26.0, 14.0], [0.315333307, 0.0, 0.0, 0.0], [-0.091190502, 0.0, 0.0, 0.0], [-0.0532120205, 0.0, 0.0, 0.0], [-0.294107229, 0.0, 0.0, 0.0], [0.144746214, 0.0, 0.0, 0.0], [-0.0978873, 0.0, 0.0, 0.0], [0.222052351, 0.0, 0.0, 0.0], [-0.0636845157, 0.0, 0.0, 0.0], [0.104225188, 0.0, 0.0, 0.0], [-0.226737335, 0.0, 0.0, 0.0], [0.351750463, 0.0, 0.0, 0.0], [0.0592035055, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.547500014, 1.0, 2.0, 4.0], [0.190499991, 3.0, 4.0, 9.0], [0.930999994, 5.0, 6.0, 5.0], [616852608.0, 7.0, 8.0, 3.0], [121.735001, 9.0, 10.0, 13.0], [7.5, 11.0, 12.0, 6.0], [119.499001, 13.0, 14.0, 13.0], [0.5, 15.0, 16.0, 8.0], [0.802000046, 17.0, 18.0, 5.0], [0.139406517, 0.0, 0.0, 0.0], [-0.350829154, 0.0, 0.0, 0.0], [1.5, 19.0, 20.0, 6.0], [0.761000037, 21.0, 22.0, 5.0], [0.948500037, 23.0, 24.0, 5.0], [232062.5, 25.0, 26.0, 14.0], [-0.315333277, 0.0, 0.0, 0.0], [0.091190502, 0.0, 0.0, 0.0], [0.0532120243, 0.0, 0.0, 0.0], [0.294107229, 0.0, 0.0, 0.0], [-0.144746214, 0.0, 0.0, 0.0], [0.0978872851, 0.0, 0.0, 0.0], [-0.222052321, 0.0, 0.0, 0.0], [0.0636845157, 0.0, 0.0, 0.0], [-0.104225181, 0.0, 0.0, 0.0], [0.22673732, 0.0, 0.0, 0.0], [-0.351750463, 0.0, 0.0, 0.0], [-0.0592034534, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 19, 20, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.49849999, 1.0, 2.0, 5.0], [2097321220.0, 3.0, 4.0, 1.0], [0.212500006, 5.0, 6.0, 12.0], [142860.0, 7.0, 8.0, 14.0], [0.705500007, 9.0, 10.0, 4.0], [0.91049999, 11.0, 12.0, 5.0], [469878688.0, 13.0, 14.0, 2.0], [-0.0251679905, 0.0, 0.0, 0.0], [-0.320775449, 0.0, 0.0, 0.0], [956.5, 15.0, 16.0, 0.0], [-9.56850052, 17.0, 18.0, 7.0], [0.340507567, 0.0, 0.0, 0.0], [-0.149520636, 0.0, 0.0, 0.0], [0.215999991, 19.0, 20.0, 11.0], [0.5, 21.0, 22.0, 8.0], [-0.251662016, 0.0, 0.0, 0.0], [0.0833341032, 0.0, 0.0, 0.0], [0.0101401052, 0.0, 0.0, 0.0], [0.249275461, 0.0, 0.0, 0.0], [-0.233702734, 0.0, 0.0, 0.0], [0.177335426, 0.0, 0.0, 0.0], [0.128130466, 0.0, 0.0, 0.0], [-0.0334994271, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 11, 12, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.49849999, 1.0, 2.0, 5.0], [2097321220.0, 3.0, 4.0, 1.0], [0.212500006, 5.0, 6.0, 12.0], [142860.0, 7.0, 8.0, 14.0], [0.705500007, 9.0, 10.0, 4.0], [0.91049999, 11.0, 12.0, 5.0], [469878688.0, 13.0, 14.0, 2.0], [0.0251679849, 0.0, 0.0, 0.0], [0.320775449, 0.0, 0.0, 0.0], [956.5, 15.0, 16.0, 0.0], [-9.56850052, 17.0, 18.0, 7.0], [-0.340507567, 0.0, 0.0, 0.0], [0.149520651, 0.0, 0.0, 0.0], [0.215999991, 19.0, 20.0, 11.0], [0.5, 21.0, 22.0, 8.0], [0.251662046, 0.0, 0.0, 0.0], [-0.0833340734, 0.0, 0.0, 0.0], [-0.0101400921, 0.0, 0.0, 0.0], [-0.249275461, 0.0, 0.0, 0.0], [0.233702719, 0.0, 0.0, 0.0], [-0.177335426, 0.0, 0.0, 0.0], [-0.128130466, 0.0, 0.0, 0.0], [0.0334994383, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 11, 12, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.0386499986, 1.0, 2.0, 9.0], [2379760130.0, 3.0, 4.0, 1.0], [0.0606000014, 5.0, 6.0, 9.0], [361.5, 7.0, 8.0, 0.0], [3176585220.0, 9.0, 10.0, 1.0], [-8.23250008, 11.0, 12.0, 7.0], [0.372500002, 13.0, 14.0, 12.0], [120.822998, 15.0, 16.0, 13.0], [178159.5, 17.0, 18.0, 14.0], [0.246156052, 0.0, 0.0, 0.0], [0.276499987, 19.0, 20.0, 12.0], [-0.233742654, 0.0, 0.0, 0.0], [0.677999973, 21.0, 22.0, 12.0], [1895590910.0, 23.0, 24.0, 2.0], [0.0927000046, 25.0, 26.0, 9.0], [-0.126325399, 0.0, 0.0, 0.0], [0.167716116, 0.0, 0.0, 0.0], [-0.0191812851, 0.0, 0.0, 0.0], [-0.343488872, 0.0, 0.0, 0.0], [0.150356159, 0.0, 0.0, 0.0], [-0.163065046, 0.0, 0.0, 0.0], [0.250407964, 0.0, 0.0, 0.0], [0.0309278276, 0.0, 0.0, 0.0], [-0.247976929, 0.0, 0.0, 0.0], [-0.00880622026, 0.0, 0.0, 0.0], [-0.118281744, 0.0, 0.0, 0.0], [0.0906829014, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 11, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.0386499986, 1.0, 2.0, 9.0], [2379760130.0, 3.0, 4.0, 1.0], [0.0606000014, 5.0, 6.0, 9.0], [361.5, 7.0, 8.0, 0.0], [3176585220.0, 9.0, 10.0, 1.0], [-8.23250008, 11.0, 12.0, 7.0], [0.372500002, 13.0, 14.0, 12.0], [120.822998, 15.0, 16.0, 13.0], [178159.5, 17.0, 18.0, 14.0], [-0.246156052, 0.0, 0.0, 0.0], [0.276499987, 19.0, 20.0, 12.0], [0.23374261, 0.0, 0.0, 0.0], [0.677999973, 21.0, 22.0, 12.0], [1895590910.0, 23.0, 24.0, 2.0], [0.0927000046, 25.0, 26.0, 9.0], [0.126325443, 0.0, 0.0, 0.0], [-0.167716116, 0.0, 0.0, 0.0], [0.0191812869, 0.0, 0.0, 0.0], [0.343488872, 0.0, 0.0, 0.0], [-0.150356174, 0.0, 0.0, 0.0], [0.163065001, 0.0, 0.0, 0.0], [-0.250407934, 0.0, 0.0, 0.0], [-0.0309278276, 0.0, 0.0, 0.0], [0.247976929, 0.0, 0.0, 0.0], [0.00880623236, 0.0, 0.0, 0.0], [0.118281737, 0.0, 0.0, 0.0], [-0.0906829238, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 11, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.579999983, 1.0, 2.0, 4.0], [-2.0150001, 3.0, 4.0, 7.0], [-5.48500013, 5.0, 6.0, 7.0], [0.289499998, 7.0, 8.0, 12.0], [0.229829073, 0.0, 0.0, 0.0], [-7.91899967, 9.0, 10.0, 7.0], [270.0, 11.0, 12.0, 0.0], [0.91049999, 13.0, 14.0, 5.0], [851.5, 15.0, 16.0, 0.0], [0.106999993, 17.0, 18.0, 9.0], [3545300990.0, 19.0, 20.0, 2.0], [182.5, 21.0, 22.0, 0.0], [0.79550004, 23.0, 24.0, 5.0], [0.138741896, 0.0, 0.0, 0.0], [-0.230389938, 0.0, 0.0, 0.0], [-0.244056568, 0.0, 0.0, 0.0], [-0.0169197135, 0.0, 0.0, 0.0], [-0.0502092391, 0.0, 0.0, 0.0], [0.262716502, 0.0, 0.0, 0.0], [-0.114581749, 0.0, 0.0, 0.0], [0.125007078, 0.0, 0.0, 0.0], [0.160366789, 0.0, 0.0, 0.0], [-0.261616081, 0.0, 0.0, 0.0], [0.287813425, 0.0, 0.0, 0.0], [0.0868098214, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_23(xs):
    #Predicts Class 1
    function_dict = np.array([[0.579999983, 1.0, 2.0, 4.0], [-2.0150001, 3.0, 4.0, 7.0], [-5.48500013, 5.0, 6.0, 7.0], [0.289499998, 7.0, 8.0, 12.0], [-0.229829118, 0.0, 0.0, 0.0], [-7.91899967, 9.0, 10.0, 7.0], [270.0, 11.0, 12.0, 0.0], [0.91049999, 13.0, 14.0, 5.0], [851.5, 15.0, 16.0, 0.0], [0.106999993, 17.0, 18.0, 9.0], [3545300990.0, 19.0, 20.0, 2.0], [182.5, 21.0, 22.0, 0.0], [0.79550004, 23.0, 24.0, 5.0], [-0.13874191, 0.0, 0.0, 0.0], [0.230389953, 0.0, 0.0, 0.0], [0.244056553, 0.0, 0.0, 0.0], [0.0169197489, 0.0, 0.0, 0.0], [0.0502092279, 0.0, 0.0, 0.0], [-0.262716502, 0.0, 0.0, 0.0], [0.114581749, 0.0, 0.0, 0.0], [-0.125007063, 0.0, 0.0, 0.0], [-0.160366818, 0.0, 0.0, 0.0], [0.261616141, 0.0, 0.0, 0.0], [-0.287813455, 0.0, 0.0, 0.0], [-0.0868097991, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_24(xs):
    #Predicts Class 0
    function_dict = np.array([[0.879999995, 1.0, 2.0, 4.0], [0.862499952, 3.0, 4.0, 4.0], [0.241139635, 0.0, 0.0, 0.0], [1.5, 5.0, 6.0, 6.0], [-0.295073271, 0.0, 0.0, 0.0], [0.0395999998, 7.0, 8.0, 9.0], [0.0458000004, 9.0, 10.0, 10.0], [-0.103984706, 0.0, 0.0, 0.0], [0.153636456, 0.0, 0.0, 0.0], [-0.0401514694, 0.0, 0.0, 0.0], [0.138089702, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_25(xs):
    #Predicts Class 1
    function_dict = np.array([[0.879999995, 1.0, 2.0, 4.0], [0.862499952, 3.0, 4.0, 4.0], [-0.241139621, 0.0, 0.0, 0.0], [1.5, 5.0, 6.0, 6.0], [0.295073301, 0.0, 0.0, 0.0], [0.0395999998, 7.0, 8.0, 9.0], [0.0458000004, 9.0, 10.0, 10.0], [0.103984706, 0.0, 0.0, 0.0], [-0.153636456, 0.0, 0.0, 0.0], [0.0401514657, 0.0, 0.0, 0.0], [-0.138089657, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_26(xs):
    #Predicts Class 0
    function_dict = np.array([[0.49849999, 1.0, 2.0, 5.0], [2097321220.0, 3.0, 4.0, 1.0], [3890199300.0, 5.0, 6.0, 1.0], [-0.218366951, 0.0, 0.0, 0.0], [-12.5804996, 7.0, 8.0, 7.0], [2805971460.0, 9.0, 10.0, 1.0], [4087883260.0, 11.0, 12.0, 2.0], [-0.189547047, 0.0, 0.0, 0.0], [865.5, 13.0, 14.0, 0.0], [2803587580.0, 15.0, 16.0, 2.0], [0.827000022, 17.0, 18.0, 5.0], [2.68499989e-05, 19.0, 20.0, 10.0], [0.161766693, 0.0, 0.0, 0.0], [-0.0501395427, 0.0, 0.0, 0.0], [0.21518293, 0.0, 0.0, 0.0], [0.0364662707, 0.0, 0.0, 0.0], [-0.131043032, 0.0, 0.0, 0.0], [0.216133803, 0.0, 0.0, 0.0], [-0.0150443604, 0.0, 0.0, 0.0], [-0.269178063, 0.0, 0.0, 0.0], [0.0960700884, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 13, 14, 15, 16, 17, 18, 19, 20, 12])
    branch_indices = np.array([0, 1, 4, 8, 2, 5, 9, 10, 6, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.49849999, 1.0, 2.0, 5.0], [2097321220.0, 3.0, 4.0, 1.0], [3890199300.0, 5.0, 6.0, 1.0], [0.218366951, 0.0, 0.0, 0.0], [-12.5804996, 7.0, 8.0, 7.0], [2805971460.0, 9.0, 10.0, 1.0], [4087883260.0, 11.0, 12.0, 2.0], [0.189547017, 0.0, 0.0, 0.0], [865.5, 13.0, 14.0, 0.0], [2803587580.0, 15.0, 16.0, 2.0], [0.827000022, 17.0, 18.0, 5.0], [2.68499989e-05, 19.0, 20.0, 10.0], [-0.161766723, 0.0, 0.0, 0.0], [0.0501394756, 0.0, 0.0, 0.0], [-0.215182975, 0.0, 0.0, 0.0], [-0.0364662632, 0.0, 0.0, 0.0], [0.131043047, 0.0, 0.0, 0.0], [-0.216133803, 0.0, 0.0, 0.0], [0.0150443306, 0.0, 0.0, 0.0], [0.269178033, 0.0, 0.0, 0.0], [-0.0960700735, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 13, 14, 15, 16, 17, 18, 19, 20, 12])
    branch_indices = np.array([0, 1, 4, 8, 2, 5, 9, 10, 6, 11])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.773999989, 1.0, 2.0, 4.0], [0.863499999, 3.0, 4.0, 12.0], [3837065470.0, 5.0, 6.0, 3.0], [118.983002, 7.0, 8.0, 13.0], [-0.252284825, 0.0, 0.0, 0.0], [888.0, 9.0, 10.0, 0.0], [-0.221962973, 0.0, 0.0, 0.0], [3533849090.0, 11.0, 12.0, 3.0], [0.764500022, 13.0, 14.0, 5.0], [0.296465039, 0.0, 0.0, 0.0], [1695778430.0, 15.0, 16.0, 1.0], [-0.1356401, 0.0, 0.0, 0.0], [0.144482866, 0.0, 0.0, 0.0], [-0.067534551, 0.0, 0.0, 0.0], [0.126888976, 0.0, 0.0, 0.0], [-0.223964006, 0.0, 0.0, 0.0], [0.111162461, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 9, 15, 16, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.773999989, 1.0, 2.0, 4.0], [0.863499999, 3.0, 4.0, 12.0], [3837065470.0, 5.0, 6.0, 3.0], [118.983002, 7.0, 8.0, 13.0], [0.252284795, 0.0, 0.0, 0.0], [888.0, 9.0, 10.0, 0.0], [0.221962973, 0.0, 0.0, 0.0], [3533849090.0, 11.0, 12.0, 3.0], [0.764500022, 13.0, 14.0, 5.0], [-0.296465039, 0.0, 0.0, 0.0], [1695778430.0, 15.0, 16.0, 1.0], [0.1356401, 0.0, 0.0, 0.0], [-0.144482896, 0.0, 0.0, 0.0], [0.0675345287, 0.0, 0.0, 0.0], [-0.12688899, 0.0, 0.0, 0.0], [0.223963991, 0.0, 0.0, 0.0], [-0.111162454, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 4, 9, 15, 16, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 2, 5, 10])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[169.152008, 1.0, 2.0, 13.0], [0.0639500022, 3.0, 4.0, 11.0], [435.0, 5.0, 6.0, 0.0], [-5.26949978, 7.0, 8.0, 7.0], [143.9245, 9.0, 10.0, 13.0], [-0.293508291, 0.0, 0.0, 0.0], [0.457000017, 11.0, 12.0, 4.0], [-0.244620621, 0.0, 0.0, 0.0], [0.0101040201, 0.0, 0.0, 0.0], [142.020996, 13.0, 14.0, 13.0], [0.0490000024, 15.0, 16.0, 9.0], [-0.22484, 0.0, 0.0, 0.0], [0.210944131, 0.0, 0.0, 0.0], [0.03346508, 0.0, 0.0, 0.0], [-0.37128526, 0.0, 0.0, 0.0], [-0.142317384, 0.0, 0.0, 0.0], [0.194680497, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 13, 14, 15, 16, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[169.152008, 1.0, 2.0, 13.0], [0.0639500022, 3.0, 4.0, 11.0], [435.0, 5.0, 6.0, 0.0], [-5.26949978, 7.0, 8.0, 7.0], [143.9245, 9.0, 10.0, 13.0], [0.293508321, 0.0, 0.0, 0.0], [0.457000017, 11.0, 12.0, 4.0], [0.244620591, 0.0, 0.0, 0.0], [-0.0101040453, 0.0, 0.0, 0.0], [142.020996, 13.0, 14.0, 13.0], [0.0490000024, 15.0, 16.0, 9.0], [0.22484, 0.0, 0.0, 0.0], [-0.210944146, 0.0, 0.0, 0.0], [-0.0334650949, 0.0, 0.0, 0.0], [0.37128526, 0.0, 0.0, 0.0], [0.142317399, 0.0, 0.0, 0.0], [-0.194680512, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 13, 14, 15, 16, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.555500031, 1.0, 2.0, 4.0], [2135151870.0, 3.0, 4.0, 3.0], [1126.5, 5.0, 6.0, 0.0], [2003817600.0, 7.0, 8.0, 3.0], [0.140500009, 9.0, 10.0, 11.0], [10.5, 11.0, 12.0, 6.0], [0.75999999, 13.0, 14.0, 4.0], [0.5, 15.0, 16.0, 8.0], [0.239419088, 0.0, 0.0, 0.0], [-4.35599995, 17.0, 18.0, 7.0], [-0.24915491, 0.0, 0.0, 0.0], [0.680500031, 19.0, 20.0, 12.0], [1184765180.0, 21.0, 22.0, 3.0], [0.253964096, 0.0, 0.0, 0.0], [-0.0776696578, 0.0, 0.0, 0.0], [0.0632373095, 0.0, 0.0, 0.0], [-0.1623521, 0.0, 0.0, 0.0], [-0.0991521552, 0.0, 0.0, 0.0], [0.0946565494, 0.0, 0.0, 0.0], [0.0651900619, 0.0, 0.0, 0.0], [-0.0850061104, 0.0, 0.0, 0.0], [-0.271591216, 0.0, 0.0, 0.0], [0.0226161554, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_33(xs):
    #Predicts Class 1
    function_dict = np.array([[0.555500031, 1.0, 2.0, 4.0], [2135151870.0, 3.0, 4.0, 3.0], [1126.5, 5.0, 6.0, 0.0], [2003817600.0, 7.0, 8.0, 3.0], [0.140500009, 9.0, 10.0, 11.0], [10.5, 11.0, 12.0, 6.0], [0.75999999, 13.0, 14.0, 4.0], [0.5, 15.0, 16.0, 8.0], [-0.239419088, 0.0, 0.0, 0.0], [-4.35599995, 17.0, 18.0, 7.0], [0.24915491, 0.0, 0.0, 0.0], [0.680500031, 19.0, 20.0, 12.0], [1184765180.0, 21.0, 22.0, 3.0], [-0.253964096, 0.0, 0.0, 0.0], [0.0776695684, 0.0, 0.0, 0.0], [-0.0632373244, 0.0, 0.0, 0.0], [0.162352115, 0.0, 0.0, 0.0], [0.0991521701, 0.0, 0.0, 0.0], [-0.0946564972, 0.0, 0.0, 0.0], [-0.0651900619, 0.0, 0.0, 0.0], [0.0850061104, 0.0, 0.0, 0.0], [0.271591187, 0.0, 0.0, 0.0], [-0.0226161666, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_34(xs):
    #Predicts Class 0
    function_dict = np.array([[129.367493, 1.0, 2.0, 13.0], [120.425003, 3.0, 4.0, 13.0], [241.5, 5.0, 6.0, 0.0], [2987724540.0, 7.0, 8.0, 3.0], [0.957000017, 9.0, 10.0, 5.0], [96.0, 11.0, 12.0, 0.0], [0.000104549996, 13.0, 14.0, 10.0], [0.5, 15.0, 16.0, 8.0], [238869.5, 17.0, 18.0, 14.0], [3512573180.0, 19.0, 20.0, 3.0], [-0.092831336, 0.0, 0.0, 0.0], [0.0540866964, 0.0, 0.0, 0.0], [-4.34549999, 21.0, 22.0, 7.0], [2784996350.0, 23.0, 24.0, 1.0], [139.852997, 25.0, 26.0, 13.0], [0.0265677441, 0.0, 0.0, 0.0], [-0.159546852, 0.0, 0.0, 0.0], [0.16416572, 0.0, 0.0, 0.0], [-0.163901344, 0.0, 0.0, 0.0], [0.248489216, 0.0, 0.0, 0.0], [-0.0409249924, 0.0, 0.0, 0.0], [-0.298205435, 0.0, 0.0, 0.0], [-0.0274277814, 0.0, 0.0, 0.0], [-0.143574953, 0.0, 0.0, 0.0], [0.0698564425, 0.0, 0.0, 0.0], [-0.165204495, 0.0, 0.0, 0.0], [0.181051284, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 10, 11, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[129.367493, 1.0, 2.0, 13.0], [120.425003, 3.0, 4.0, 13.0], [241.5, 5.0, 6.0, 0.0], [2987724540.0, 7.0, 8.0, 3.0], [0.957000017, 9.0, 10.0, 5.0], [96.0, 11.0, 12.0, 0.0], [0.000104549996, 13.0, 14.0, 10.0], [0.5, 15.0, 16.0, 8.0], [238869.5, 17.0, 18.0, 14.0], [3512573180.0, 19.0, 20.0, 3.0], [0.0928313062, 0.0, 0.0, 0.0], [-0.0540866889, 0.0, 0.0, 0.0], [-4.34549999, 21.0, 22.0, 7.0], [2784996350.0, 23.0, 24.0, 1.0], [139.852997, 25.0, 26.0, 13.0], [-0.0265677143, 0.0, 0.0, 0.0], [0.159546793, 0.0, 0.0, 0.0], [-0.16416572, 0.0, 0.0, 0.0], [0.163901344, 0.0, 0.0, 0.0], [-0.248489186, 0.0, 0.0, 0.0], [0.040924985, 0.0, 0.0, 0.0], [0.298205435, 0.0, 0.0, 0.0], [0.0274278484, 0.0, 0.0, 0.0], [0.143574938, 0.0, 0.0, 0.0], [-0.0698564649, 0.0, 0.0, 0.0], [0.16520445, 0.0, 0.0, 0.0], [-0.181051269, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 10, 11, 21, 22, 23, 24, 25, 26])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.640499949, 1.0, 2.0, 4.0], [0.369000018, 3.0, 4.0, 11.0], [-6.41899967, 5.0, 6.0, 7.0], [0.212500006, 7.0, 8.0, 12.0], [0.79550004, 9.0, 10.0, 5.0], [-6.65400028, 11.0, 12.0, 7.0], [0.752499998, 13.0, 14.0, 5.0], [0.528499961, 15.0, 16.0, 5.0], [2003817600.0, 17.0, 18.0, 3.0], [0.203010321, 0.0, 0.0, 0.0], [-0.0567628816, 0.0, 0.0, 0.0], [160423.0, 19.0, 20.0, 14.0], [-0.251168281, 0.0, 0.0, 0.0], [3454499840.0, 21.0, 22.0, 2.0], [107.990997, 23.0, 24.0, 13.0], [-0.068581447, 0.0, 0.0, 0.0], [0.164328679, 0.0, 0.0, 0.0], [-0.170997977, 0.0, 0.0, 0.0], [-0.0439518467, 0.0, 0.0, 0.0], [-0.12660791, 0.0, 0.0, 0.0], [0.0796294883, 0.0, 0.0, 0.0], [0.25126645, 0.0, 0.0, 0.0], [-0.0157166254, 0.0, 0.0, 0.0], [-0.16101104, 0.0, 0.0, 0.0], [0.070129171, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 19, 20, 12, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.640499949, 1.0, 2.0, 4.0], [0.369000018, 3.0, 4.0, 11.0], [-6.41899967, 5.0, 6.0, 7.0], [0.212500006, 7.0, 8.0, 12.0], [0.79550004, 9.0, 10.0, 5.0], [-6.65400028, 11.0, 12.0, 7.0], [0.752499998, 13.0, 14.0, 5.0], [0.528499961, 15.0, 16.0, 5.0], [2003817600.0, 17.0, 18.0, 3.0], [-0.203010276, 0.0, 0.0, 0.0], [0.0567629375, 0.0, 0.0, 0.0], [160423.0, 19.0, 20.0, 14.0], [0.251168311, 0.0, 0.0, 0.0], [3454499840.0, 21.0, 22.0, 2.0], [107.990997, 23.0, 24.0, 13.0], [0.0685813054, 0.0, 0.0, 0.0], [-0.164328694, 0.0, 0.0, 0.0], [0.170997962, 0.0, 0.0, 0.0], [0.0439518169, 0.0, 0.0, 0.0], [0.12660794, 0.0, 0.0, 0.0], [-0.0796295255, 0.0, 0.0, 0.0], [-0.25126645, 0.0, 0.0, 0.0], [0.0157166198, 0.0, 0.0, 0.0], [0.16101104, 0.0, 0.0, 0.0], [-0.0701291561, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 19, 20, 12, 21, 22, 23, 24])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-2.61100006, 1.0, 2.0, 7.0], [2351732480.0, 3.0, 4.0, 3.0], [0.175286487, 0.0, 0.0, 0.0], [-4.22049999, 5.0, 6.0, 7.0], [214468.5, 7.0, 8.0, 14.0], [2572116990.0, 9.0, 10.0, 2.0], [938.5, 11.0, 12.0, 0.0], [3184866820.0, 13.0, 14.0, 2.0], [6.80000028e-07, 15.0, 16.0, 10.0], [-0.0108964294, 0.0, 0.0, 0.0], [0.154929236, 0.0, 0.0, 0.0], [-0.178159058, 0.0, 0.0, 0.0], [-0.00486189499, 0.0, 0.0, 0.0], [0.0407428592, 0.0, 0.0, 0.0], [-0.131128609, 0.0, 0.0, 0.0], [0.0543172918, 0.0, 0.0, 0.0], [-0.238079354, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 13, 14, 15, 16, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-2.61100006, 1.0, 2.0, 7.0], [2351732480.0, 3.0, 4.0, 3.0], [-0.175286457, 0.0, 0.0, 0.0], [-4.22049999, 5.0, 6.0, 7.0], [214468.5, 7.0, 8.0, 14.0], [2572116990.0, 9.0, 10.0, 2.0], [938.5, 11.0, 12.0, 0.0], [3184866820.0, 13.0, 14.0, 2.0], [6.80000028e-07, 15.0, 16.0, 10.0], [0.0108964145, 0.0, 0.0, 0.0], [-0.154929236, 0.0, 0.0, 0.0], [0.178159118, 0.0, 0.0, 0.0], [0.00486180885, 0.0, 0.0, 0.0], [-0.0407428555, 0.0, 0.0, 0.0], [0.131128639, 0.0, 0.0, 0.0], [-0.0543172956, 0.0, 0.0, 0.0], [0.238079339, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 13, 14, 15, 16, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.879999995, 1.0, 2.0, 4.0], [913.5, 3.0, 4.0, 0.0], [0.178096071, 0.0, 0.0, 0.0], [671.5, 5.0, 6.0, 0.0], [3643216900.0, 7.0, 8.0, 2.0], [419.0, 9.0, 10.0, 0.0], [166227.0, 11.0, 12.0, 14.0], [0.763999999, 13.0, 14.0, 4.0], [-0.123812966, 0.0, 0.0, 0.0], [-0.0549799614, 0.0, 0.0, 0.0], [0.113511585, 0.0, 0.0, 0.0], [0.0293095447, 0.0, 0.0, 0.0], [-0.170986041, 0.0, 0.0, 0.0], [0.145900354, 0.0, 0.0, 0.0], [-0.088320896, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 13, 14, 8, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4, 7])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.879999995, 1.0, 2.0, 4.0], [913.5, 3.0, 4.0, 0.0], [-0.178096086, 0.0, 0.0, 0.0], [671.5, 5.0, 6.0, 0.0], [3643216900.0, 7.0, 8.0, 2.0], [419.0, 9.0, 10.0, 0.0], [166227.0, 11.0, 12.0, 14.0], [0.763999999, 13.0, 14.0, 4.0], [0.123813003, 0.0, 0.0, 0.0], [0.0549799614, 0.0, 0.0, 0.0], [-0.1135116, 0.0, 0.0, 0.0], [-0.0293095615, 0.0, 0.0, 0.0], [0.170986041, 0.0, 0.0, 0.0], [-0.145900324, 0.0, 0.0, 0.0], [0.0883209258, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 13, 14, 8, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4, 7])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[169.152008, 1.0, 2.0, 13.0], [-12.2334995, 3.0, 4.0, 7.0], [-5.14799976, 5.0, 6.0, 7.0], [-0.160177544, 0.0, 0.0, 0.0], [0.0386499986, 7.0, 8.0, 9.0], [-0.0211810302, 0.0, 0.0, 0.0], [-0.199927866, 0.0, 0.0, 0.0], [2.62000003e-06, 9.0, 10.0, 10.0], [0.0615499988, 11.0, 12.0, 9.0], [-0.174274012, 0.0, 0.0, 0.0], [0.0574880466, 0.0, 0.0, 0.0], [0.122290827, 0.0, 0.0, 0.0], [0.00114402687, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 9, 10, 11, 12, 5, 6])
    branch_indices = np.array([0, 1, 4, 7, 8, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[169.152008, 1.0, 2.0, 13.0], [-12.2334995, 3.0, 4.0, 7.0], [-5.14799976, 5.0, 6.0, 7.0], [0.160177559, 0.0, 0.0, 0.0], [0.0386499986, 7.0, 8.0, 9.0], [0.0211810414, 0.0, 0.0, 0.0], [0.199927866, 0.0, 0.0, 0.0], [2.62000003e-06, 9.0, 10.0, 10.0], [0.0615499988, 11.0, 12.0, 9.0], [0.174273983, 0.0, 0.0, 0.0], [-0.0574879907, 0.0, 0.0, 0.0], [-0.12229082, 0.0, 0.0, 0.0], [-0.00114403199, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 9, 10, 11, 12, 5, 6])
    branch_indices = np.array([0, 1, 4, 7, 8, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.773999989, 1.0, 2.0, 4.0], [0.683500051, 3.0, 4.0, 12.0], [3837065470.0, 5.0, 6.0, 3.0], [0.303499997, 7.0, 8.0, 9.0], [0.307500005, 9.0, 10.0, 9.0], [888.0, 11.0, 12.0, 0.0], [-0.0867193565, 0.0, 0.0, 0.0], [0.547500014, 13.0, 14.0, 4.0], [0.134499997, 15.0, 16.0, 11.0], [0.5, 17.0, 18.0, 6.0], [0.144438952, 0.0, 0.0, 0.0], [0.223644063, 0.0, 0.0, 0.0], [0.825999975, 19.0, 20.0, 4.0], [-0.0687306896, 0.0, 0.0, 0.0], [0.0851448923, 0.0, 0.0, 0.0], [0.0288649015, 0.0, 0.0, 0.0], [-0.229459152, 0.0, 0.0, 0.0], [0.0795517787, 0.0, 0.0, 0.0], [-0.199124292, 0.0, 0.0, 0.0], [0.0962754562, 0.0, 0.0, 0.0], [-0.13531594, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 10, 11, 19, 20, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.773999989, 1.0, 2.0, 4.0], [0.683500051, 3.0, 4.0, 12.0], [3837065470.0, 5.0, 6.0, 3.0], [0.303499997, 7.0, 8.0, 9.0], [0.307500005, 9.0, 10.0, 9.0], [888.0, 11.0, 12.0, 0.0], [0.0867193565, 0.0, 0.0, 0.0], [0.547500014, 13.0, 14.0, 4.0], [0.134499997, 15.0, 16.0, 11.0], [0.5, 17.0, 18.0, 6.0], [-0.144438937, 0.0, 0.0, 0.0], [-0.223644063, 0.0, 0.0, 0.0], [0.825999975, 19.0, 20.0, 4.0], [0.0687306598, 0.0, 0.0, 0.0], [-0.0851448774, 0.0, 0.0, 0.0], [-0.0288649108, 0.0, 0.0, 0.0], [0.229459122, 0.0, 0.0, 0.0], [-0.0795517415, 0.0, 0.0, 0.0], [0.199124262, 0.0, 0.0, 0.0], [-0.0962754562, 0.0, 0.0, 0.0], [0.13531591, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 10, 11, 19, 20, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.379500002, 1.0, 2.0, 9.0], [-7.23600006, 3.0, 4.0, 7.0], [0.166204378, 0.0, 0.0, 0.0], [927.0, 5.0, 6.0, 0.0], [550448768.0, 7.0, 8.0, 2.0], [0.772500038, 9.0, 10.0, 4.0], [0.00030449999, 11.0, 12.0, 10.0], [405.0, 13.0, 14.0, 0.0], [542.0, 15.0, 16.0, 0.0], [-0.173995495, 0.0, 0.0, 0.0], [0.0518933274, 0.0, 0.0, 0.0], [0.13350518, 0.0, 0.0, 0.0], [-0.158907652, 0.0, 0.0, 0.0], [-0.200444639, 0.0, 0.0, 0.0], [-0.00915055629, 0.0, 0.0, 0.0], [0.10168919, 0.0, 0.0, 0.0], [-0.0143755861, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 13, 14, 15, 16, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.379500002, 1.0, 2.0, 9.0], [-7.23600006, 3.0, 4.0, 7.0], [-0.166204378, 0.0, 0.0, 0.0], [927.0, 5.0, 6.0, 0.0], [550448768.0, 7.0, 8.0, 2.0], [0.772500038, 9.0, 10.0, 4.0], [0.00030449999, 11.0, 12.0, 10.0], [405.0, 13.0, 14.0, 0.0], [542.0, 15.0, 16.0, 0.0], [0.173995465, 0.0, 0.0, 0.0], [-0.0518933199, 0.0, 0.0, 0.0], [-0.13350518, 0.0, 0.0, 0.0], [0.158907592, 0.0, 0.0, 0.0], [0.200444639, 0.0, 0.0, 0.0], [0.00915054884, 0.0, 0.0, 0.0], [-0.101689197, 0.0, 0.0, 0.0], [0.0143755777, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([9, 10, 11, 12, 13, 14, 15, 16, 2])
    branch_indices = np.array([0, 1, 3, 5, 6, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[245258.5, 1.0, 2.0, 14.0], [222032.5, 3.0, 4.0, 14.0], [1.5, 5.0, 6.0, 6.0], [7.5, 7.0, 8.0, 6.0], [0.894999981, 9.0, 10.0, 5.0], [0.0805746242, 0.0, 0.0, 0.0], [0.207500011, 11.0, 12.0, 12.0], [119.036499, 13.0, 14.0, 13.0], [0.756999969, 15.0, 16.0, 5.0], [129.001007, 17.0, 18.0, 13.0], [-0.0345907584, 0.0, 0.0, 0.0], [-0.00635452801, 0.0, 0.0, 0.0], [-0.204244614, 0.0, 0.0, 0.0], [-0.086991705, 0.0, 0.0, 0.0], [0.00636152411, 0.0, 0.0, 0.0], [0.146671802, 0.0, 0.0, 0.0], [-0.0384961665, 0.0, 0.0, 0.0], [0.203888088, 0.0, 0.0, 0.0], [0.0374640524, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 10, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[245258.5, 1.0, 2.0, 14.0], [222032.5, 3.0, 4.0, 14.0], [1.5, 5.0, 6.0, 6.0], [7.5, 7.0, 8.0, 6.0], [0.894999981, 9.0, 10.0, 5.0], [-0.0805746317, 0.0, 0.0, 0.0], [0.207500011, 11.0, 12.0, 12.0], [119.036499, 13.0, 14.0, 13.0], [0.756999969, 15.0, 16.0, 5.0], [129.001007, 17.0, 18.0, 13.0], [0.0345907584, 0.0, 0.0, 0.0], [0.00635440694, 0.0, 0.0, 0.0], [0.204244629, 0.0, 0.0, 0.0], [0.0869916677, 0.0, 0.0, 0.0], [-0.00636152271, 0.0, 0.0, 0.0], [-0.146671817, 0.0, 0.0, 0.0], [0.0384961702, 0.0, 0.0, 0.0], [-0.203888088, 0.0, 0.0, 0.0], [-0.0374640636, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 15, 16, 17, 18, 10, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.49849999, 1.0, 2.0, 5.0], [0.42899999, 3.0, 4.0, 5.0], [10.5, 5.0, 6.0, 6.0], [0.010442893, 0.0, 0.0, 0.0], [-0.157243639, 0.0, 0.0, 0.0], [0.150000006, 7.0, 8.0, 11.0], [1184765180.0, 9.0, 10.0, 3.0], [0.0796999931, 11.0, 12.0, 11.0], [1153817860.0, 13.0, 14.0, 2.0], [-0.184457511, 0.0, 0.0, 0.0], [0.0279223379, 0.0, 0.0, 0.0], [-0.0767670348, 0.0, 0.0, 0.0], [0.106483623, 0.0, 0.0, 0.0], [0.0874338299, 0.0, 0.0, 0.0], [-0.0670427158, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 11, 12, 13, 14, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 7, 8, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.49849999, 1.0, 2.0, 5.0], [0.42899999, 3.0, 4.0, 5.0], [10.5, 5.0, 6.0, 6.0], [-0.0104428418, 0.0, 0.0, 0.0], [0.157243669, 0.0, 0.0, 0.0], [0.150000006, 7.0, 8.0, 11.0], [1184765180.0, 9.0, 10.0, 3.0], [0.0796999931, 11.0, 12.0, 11.0], [1153817860.0, 13.0, 14.0, 2.0], [0.184457511, 0.0, 0.0, 0.0], [-0.0279224012, 0.0, 0.0, 0.0], [0.0767670423, 0.0, 0.0, 0.0], [-0.106483586, 0.0, 0.0, 0.0], [-0.0874338895, 0.0, 0.0, 0.0], [0.0670426786, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 11, 12, 13, 14, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 7, 8, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[118.983002, 1.0, 2.0, 13.0], [0.770500004, 3.0, 4.0, 5.0], [129.977509, 5.0, 6.0, 13.0], [6.02999989e-06, 7.0, 8.0, 10.0], [0.44600001, 9.0, 10.0, 12.0], [0.370999992, 11.0, 12.0, 12.0], [135.806, 13.0, 14.0, 13.0], [0.0397999994, 15.0, 16.0, 9.0], [-7.39349985, 17.0, 18.0, 7.0], [0.0250731688, 0.0, 0.0, 0.0], [-0.193040192, 0.0, 0.0, 0.0], [-0.0219941642, 0.0, 0.0, 0.0], [0.796500027, 19.0, 20.0, 12.0], [-0.153383419, 0.0, 0.0, 0.0], [0.000104549996, 21.0, 22.0, 10.0], [-0.0770215988, 0.0, 0.0, 0.0], [0.124227464, 0.0, 0.0, 0.0], [-0.168275893, 0.0, 0.0, 0.0], [0.0409812927, 0.0, 0.0, 0.0], [0.191458508, 0.0, 0.0, 0.0], [-0.021313563, 0.0, 0.0, 0.0], [-0.0481434017, 0.0, 0.0, 0.0], [0.100563847, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 11, 19, 20, 13, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 12, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[118.983002, 1.0, 2.0, 13.0], [0.770500004, 3.0, 4.0, 5.0], [129.977509, 5.0, 6.0, 13.0], [6.02999989e-06, 7.0, 8.0, 10.0], [0.44600001, 9.0, 10.0, 12.0], [0.370999992, 11.0, 12.0, 12.0], [135.806, 13.0, 14.0, 13.0], [0.0397999994, 15.0, 16.0, 9.0], [-7.39349985, 17.0, 18.0, 7.0], [-0.0250731818, 0.0, 0.0, 0.0], [0.193040192, 0.0, 0.0, 0.0], [0.021994153, 0.0, 0.0, 0.0], [0.796500027, 19.0, 20.0, 12.0], [0.153383508, 0.0, 0.0, 0.0], [0.000104549996, 21.0, 22.0, 10.0], [0.0770215616, 0.0, 0.0, 0.0], [-0.124227472, 0.0, 0.0, 0.0], [0.168275863, 0.0, 0.0, 0.0], [-0.0409812815, 0.0, 0.0, 0.0], [-0.191458523, 0.0, 0.0, 0.0], [0.021313576, 0.0, 0.0, 0.0], [0.0481433831, 0.0, 0.0, 0.0], [-0.100563988, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 10, 11, 19, 20, 13, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 2, 5, 12, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    for booster_index in range(0,54,2):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def logit_class_1(xs):
    sum_of_leaf_values = np.zeros(xs.shape[0])
    for booster_index in range(1,54,2):
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
