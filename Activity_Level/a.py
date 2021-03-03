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
# Invocation: btc spotify.csv
# Total compiler execution time: 0:00:40.37. Finished on: Mar-02-2021 18:02:37.
# This source code requires Python 3.
#
"""
Classifier Type:                    Random Forest
System Type:                         Binary classifier
Training/Validation Split:           50:50%
Best-guess accuracy:                 56.61%
Training accuracy:                   100.00% (612/612 correct)
Validation accuracy:                 86.60% (530/612 correct)
Overall Model accuracy:              93.30% (1142/1224 correct)
Overall Improvement over best guess: 36.69% (of possible 43.39%)
Model capacity (MEC):                11 bits
Generalization ratio:                55.13 bits/bit
Model efficiency:                    3.33%/parameter
System behavior
True Negatives:                      40.20% (492/1224)
True Positives:                      53.10% (650/1224)
False Negatives:                     3.51% (43/1224)
False Positives:                     3.19% (39/1224)
True Pos. Rate/Sensitivity/Recall:   0.94
True Neg. Rate/Specificity:          0.93
Precision:                           0.94
F-1 Measure:                         0.94
False Negative Rate/Miss Rate:       0.06
Critical Success Index:              0.89
Confusion Matrix:
 [40.20% 3.19%]
 [3.51% 53.10%]
Generalization index:                27.43
Percent of Data Memorized:           3.65%
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
    function_dict = np.array([[0.555500031, 1.0, 2.0, 5.0], [0.272000015, 3.0, 4.0, 10.0], [0.0442500003, 5.0, 6.0, 10.0], [-7.03750038, 7.0, 8.0, 8.0], [2525430020.0, 9.0, 10.0, 1.0], [119.237999, 11.0, 12.0, 14.0], [0.801499963, 13.0, 14.0, 13.0], [2.0, 15.0, 16.0, 16.0], [0.800500035, 17.0, 18.0, 6.0], [819345600.0, 19.0, 20.0, 1.0], [0.466000021, 0.0, 0.0, 0.0], [76.7680054, 21.0, 22.0, 14.0], [129.0215, 23.0, 24.0, 14.0], [-12.9845009, 25.0, 26.0, 8.0], [0.769500017, 27.0, 28.0, 5.0], [-0.0, 0.0, 0.0, 0.0], [-0.640429378, 0.0, 0.0, 0.0], [0.0197696984, 0.0, 0.0, 0.0], [-0.530075014, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [-0.434933364, 0.0, 0.0, 0.0], [0.326200008, 0.0, 0.0, 0.0], [-0.447725505, 0.0, 0.0, 0.0], [0.48403874, 0.0, 0.0, 0.0], [-0.312017411, 0.0, 0.0, 0.0], [-0.391440034, 0.0, 0.0, 0.0], [0.394800007, 0.0, 0.0, 0.0], [-0.287055999, 0.0, 0.0, 0.0], [0.533781826, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 10, 21, 22, 23, 24, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.555500031, 1.0, 2.0, 5.0], [0.272000015, 3.0, 4.0, 10.0], [0.0442500003, 5.0, 6.0, 10.0], [-7.03750038, 7.0, 8.0, 8.0], [2525430020.0, 9.0, 10.0, 1.0], [119.237999, 11.0, 12.0, 14.0], [0.801499963, 13.0, 14.0, 13.0], [2.0, 15.0, 16.0, 16.0], [0.800500035, 17.0, 18.0, 6.0], [819345600.0, 19.0, 20.0, 1.0], [-0.466000021, 0.0, 0.0, 0.0], [76.7680054, 21.0, 22.0, 14.0], [129.0215, 23.0, 24.0, 14.0], [-12.9845009, 25.0, 26.0, 8.0], [0.769500017, 27.0, 28.0, 5.0], [-0.0, 0.0, 0.0, 0.0], [0.640429378, 0.0, 0.0, 0.0], [-0.0197696984, 0.0, 0.0, 0.0], [0.530075014, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [0.434933364, 0.0, 0.0, 0.0], [-0.326200008, 0.0, 0.0, 0.0], [0.447725505, 0.0, 0.0, 0.0], [-0.48403874, 0.0, 0.0, 0.0], [0.312017411, 0.0, 0.0, 0.0], [0.391440034, 0.0, 0.0, 0.0], [-0.394800007, 0.0, 0.0, 0.0], [0.287055999, 0.0, 0.0, 0.0], [-0.533781826, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 10, 21, 22, 23, 24, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 2, 5, 11, 12, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.579999983, 1.0, 2.0, 5.0], [-5.93799973, 3.0, 4.0, 8.0], [-7.23600006, 5.0, 6.0, 8.0], [210133.0, 7.0, 8.0, 15.0], [0.800500035, 9.0, 10.0, 6.0], [0.771999955, 11.0, 12.0, 5.0], [0.881500006, 13.0, 14.0, 13.0], [0.868499994, 15.0, 16.0, 6.0], [74925456.0, 17.0, 18.0, 4.0], [0.0388999991, 19.0, 20.0, 10.0], [0.5, 21.0, 22.0, 9.0], [1051.0, 23.0, 24.0, 0.0], [0.000236150008, 25.0, 26.0, 11.0], [229520.5, 27.0, 28.0, 15.0], [0.769500017, 29.0, 30.0, 5.0], [-0.312400579, 0.0, 0.0, 0.0], [0.330777645, 0.0, 0.0, 0.0], [0.016549157, 0.0, 0.0, 0.0], [-0.436627805, 0.0, 0.0, 0.0], [-0.266035706, 0.0, 0.0, 0.0], [0.413705915, 0.0, 0.0, 0.0], [-0.0409836359, 0.0, 0.0, 0.0], [-0.396665066, 0.0, 0.0, 0.0], [-0.382869899, 0.0, 0.0, 0.0], [0.16592443, 0.0, 0.0, 0.0], [0.327867448, 0.0, 0.0, 0.0], [-0.245176479, 0.0, 0.0, 0.0], [0.288905323, 0.0, 0.0, 0.0], [-0.0268535111, 0.0, 0.0, 0.0], [-0.442273408, 0.0, 0.0, 0.0], [0.23763819, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_3(xs):
    #Predicts Class 1
    function_dict = np.array([[0.579999983, 1.0, 2.0, 5.0], [-5.93799973, 3.0, 4.0, 8.0], [-7.23600006, 5.0, 6.0, 8.0], [210133.0, 7.0, 8.0, 15.0], [0.800500035, 9.0, 10.0, 6.0], [0.771999955, 11.0, 12.0, 5.0], [0.881500006, 13.0, 14.0, 13.0], [0.868499994, 15.0, 16.0, 6.0], [74925456.0, 17.0, 18.0, 4.0], [0.0388999991, 19.0, 20.0, 10.0], [0.5, 21.0, 22.0, 9.0], [1051.0, 23.0, 24.0, 0.0], [0.000236150008, 25.0, 26.0, 11.0], [229520.5, 27.0, 28.0, 15.0], [0.769500017, 29.0, 30.0, 5.0], [0.312400579, 0.0, 0.0, 0.0], [-0.330777645, 0.0, 0.0, 0.0], [-0.0165491477, 0.0, 0.0, 0.0], [0.436627805, 0.0, 0.0, 0.0], [0.266035706, 0.0, 0.0, 0.0], [-0.413705885, 0.0, 0.0, 0.0], [0.0409836397, 0.0, 0.0, 0.0], [0.396665066, 0.0, 0.0, 0.0], [0.38286984, 0.0, 0.0, 0.0], [-0.16592446, 0.0, 0.0, 0.0], [-0.327867478, 0.0, 0.0, 0.0], [0.245176449, 0.0, 0.0, 0.0], [-0.288905382, 0.0, 0.0, 0.0], [0.0268534813, 0.0, 0.0, 0.0], [0.442273378, 0.0, 0.0, 0.0], [-0.237638205, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_4(xs):
    #Predicts Class 0
    function_dict = np.array([[248251.5, 1.0, 2.0, 15.0], [0.494000018, 3.0, 4.0, 13.0], [0.737999976, 5.0, 6.0, 5.0], [-12.9735003, 7.0, 8.0, 8.0], [0.732499957, 9.0, 10.0, 5.0], [1.5, 11.0, 12.0, 7.0], [0.620499969, 13.0, 14.0, 6.0], [-0.306805849, 0.0, 0.0, 0.0], [2.63000002e-05, 15.0, 16.0, 11.0], [-3.19899988, 17.0, 18.0, 8.0], [132.027008, 19.0, 20.0, 14.0], [0.722499967, 21.0, 22.0, 6.0], [108898096.0, 23.0, 24.0, 4.0], [-0.188841298, 0.0, 0.0, 0.0], [0.351377487, 0.0, 0.0, 0.0], [0.09782774, 0.0, 0.0, 0.0], [0.34508428, 0.0, 0.0, 0.0], [-0.283715338, 0.0, 0.0, 0.0], [0.302781612, 0.0, 0.0, 0.0], [0.257727772, 0.0, 0.0, 0.0], [-0.15621388, 0.0, 0.0, 0.0], [-0.369231671, 0.0, 0.0, 0.0], [0.310749322, 0.0, 0.0, 0.0], [0.0694384873, 0.0, 0.0, 0.0], [-0.377498835, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[248251.5, 1.0, 2.0, 15.0], [0.494000018, 3.0, 4.0, 13.0], [0.737999976, 5.0, 6.0, 5.0], [-12.9735003, 7.0, 8.0, 8.0], [0.732499957, 9.0, 10.0, 5.0], [1.5, 11.0, 12.0, 7.0], [0.620499969, 13.0, 14.0, 6.0], [0.306805849, 0.0, 0.0, 0.0], [2.63000002e-05, 15.0, 16.0, 11.0], [-3.19899988, 17.0, 18.0, 8.0], [132.027008, 19.0, 20.0, 14.0], [0.722499967, 21.0, 22.0, 6.0], [108898096.0, 23.0, 24.0, 4.0], [0.188841283, 0.0, 0.0, 0.0], [-0.351377487, 0.0, 0.0, 0.0], [-0.0978277624, 0.0, 0.0, 0.0], [-0.34508428, 0.0, 0.0, 0.0], [0.283715308, 0.0, 0.0, 0.0], [-0.302781671, 0.0, 0.0, 0.0], [-0.257727772, 0.0, 0.0, 0.0], [0.156213865, 0.0, 0.0, 0.0], [0.369231671, 0.0, 0.0, 0.0], [-0.310749322, 0.0, 0.0, 0.0], [-0.0694384649, 0.0, 0.0, 0.0], [0.377498835, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14])
    branch_indices = np.array([0, 1, 3, 8, 4, 9, 10, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.555500031, 1.0, 2.0, 5.0], [168495.5, 3.0, 4.0, 15.0], [-5.48500013, 5.0, 6.0, 8.0], [0.119499996, 7.0, 8.0, 12.0], [0.0419500023, 9.0, 10.0, 10.0], [0.700000048, 11.0, 12.0, 13.0], [118.983002, 13.0, 14.0, 14.0], [163297.0, 15.0, 16.0, 15.0], [2216409600.0, 17.0, 18.0, 2.0], [-0.36781159, 0.0, 0.0, 0.0], [0.0606499985, 19.0, 20.0, 10.0], [3944393730.0, 21.0, 22.0, 1.0], [0.177000001, 23.0, 24.0, 10.0], [0.762499988, 25.0, 26.0, 6.0], [469878688.0, 27.0, 28.0, 3.0], [0.132439896, 0.0, 0.0, 0.0], [0.494135737, 0.0, 0.0, 0.0], [-0.294731408, 0.0, 0.0, 0.0], [0.107098989, 0.0, 0.0, 0.0], [0.112693056, 0.0, 0.0, 0.0], [-0.253497392, 0.0, 0.0, 0.0], [0.0861527026, 0.0, 0.0, 0.0], [-0.353191644, 0.0, 0.0, 0.0], [-0.378328353, 0.0, 0.0, 0.0], [0.330306172, 0.0, 0.0, 0.0], [0.322185546, 0.0, 0.0, 0.0], [-0.305122644, 0.0, 0.0, 0.0], [-0.174470678, 0.0, 0.0, 0.0], [0.297555745, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_7(xs):
    #Predicts Class 1
    function_dict = np.array([[0.555500031, 1.0, 2.0, 5.0], [168495.5, 3.0, 4.0, 15.0], [-5.48500013, 5.0, 6.0, 8.0], [0.119499996, 7.0, 8.0, 12.0], [0.0419500023, 9.0, 10.0, 10.0], [0.700000048, 11.0, 12.0, 13.0], [118.983002, 13.0, 14.0, 14.0], [163297.0, 15.0, 16.0, 15.0], [2216409600.0, 17.0, 18.0, 2.0], [0.36781159, 0.0, 0.0, 0.0], [0.0606499985, 19.0, 20.0, 10.0], [3944393730.0, 21.0, 22.0, 1.0], [0.177000001, 23.0, 24.0, 10.0], [0.762499988, 25.0, 26.0, 6.0], [469878688.0, 27.0, 28.0, 3.0], [-0.132439852, 0.0, 0.0, 0.0], [-0.494135737, 0.0, 0.0, 0.0], [0.294731408, 0.0, 0.0, 0.0], [-0.107098989, 0.0, 0.0, 0.0], [-0.112693019, 0.0, 0.0, 0.0], [0.253497392, 0.0, 0.0, 0.0], [-0.0861527026, 0.0, 0.0, 0.0], [0.353191644, 0.0, 0.0, 0.0], [0.378328353, 0.0, 0.0, 0.0], [-0.330306172, 0.0, 0.0, 0.0], [-0.322185576, 0.0, 0.0, 0.0], [0.305122614, 0.0, 0.0, 0.0], [0.174470723, 0.0, 0.0, 0.0], [-0.297555745, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_8(xs):
    #Predicts Class 0
    function_dict = np.array([[0.379500002, 1.0, 2.0, 10.0], [0.5, 3.0, 4.0, 9.0], [0.417303383, 0.0, 0.0, 0.0], [10.5, 5.0, 6.0, 7.0], [3220066300.0, 7.0, 8.0, 1.0], [269077.0, 9.0, 10.0, 15.0], [763.0, 11.0, 12.0, 0.0], [423.0, 13.0, 14.0, 0.0], [0.598500013, 15.0, 16.0, 13.0], [0.169901475, 0.0, 0.0, 0.0], [-0.210368335, 0.0, 0.0, 0.0], [-0.0037229571, 0.0, 0.0, 0.0], [-0.61512953, 0.0, 0.0, 0.0], [-0.328033924, 0.0, 0.0, 0.0], [-0.100741804, 0.0, 0.0, 0.0], [0.233028263, 0.0, 0.0, 0.0], [-0.115058303, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_9(xs):
    #Predicts Class 1
    function_dict = np.array([[0.379500002, 1.0, 2.0, 10.0], [0.5, 3.0, 4.0, 9.0], [-0.417303413, 0.0, 0.0, 0.0], [10.5, 5.0, 6.0, 7.0], [3220066300.0, 7.0, 8.0, 1.0], [269077.0, 9.0, 10.0, 15.0], [763.0, 11.0, 12.0, 0.0], [423.0, 13.0, 14.0, 0.0], [0.598500013, 15.0, 16.0, 13.0], [-0.169901475, 0.0, 0.0, 0.0], [0.21036832, 0.0, 0.0, 0.0], [0.00372293266, 0.0, 0.0, 0.0], [0.61512953, 0.0, 0.0, 0.0], [0.328033864, 0.0, 0.0, 0.0], [0.100741796, 0.0, 0.0, 0.0], [-0.233028308, 0.0, 0.0, 0.0], [0.115058303, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_10(xs):
    #Predicts Class 0
    function_dict = np.array([[-10.7709999, 1.0, 2.0, 8.0], [0.189500004, 3.0, 4.0, 10.0], [1.5, 5.0, 6.0, 7.0], [0.0333499983, 7.0, 8.0, 10.0], [0.117591351, 0.0, 0.0, 0.0], [0.528499961, 9.0, 10.0, 5.0], [0.244500011, 11.0, 12.0, 12.0], [0.455500007, 13.0, 14.0, 6.0], [-0.343773544, 0.0, 0.0, 0.0], [0.813500047, 15.0, 16.0, 6.0], [3886659070.0, 17.0, 18.0, 4.0], [0.212500006, 19.0, 20.0, 13.0], [2.46499985e-06, 21.0, 22.0, 11.0], [-0.185816601, 0.0, 0.0, 0.0], [0.0364990681, 0.0, 0.0, 0.0], [0.184715718, 0.0, 0.0, 0.0], [-0.328924716, 0.0, 0.0, 0.0], [0.28863883, 0.0, 0.0, 0.0], [-0.103694834, 0.0, 0.0, 0.0], [0.303727835, 0.0, 0.0, 0.0], [0.00768666854, 0.0, 0.0, 0.0], [-0.337530553, 0.0, 0.0, 0.0], [-0.0137864333, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 4, 15, 16, 17, 18, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 2, 5, 9, 10, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-10.7709999, 1.0, 2.0, 8.0], [0.189500004, 3.0, 4.0, 10.0], [1.5, 5.0, 6.0, 7.0], [0.0333499983, 7.0, 8.0, 10.0], [-0.117591351, 0.0, 0.0, 0.0], [0.528499961, 9.0, 10.0, 5.0], [0.244500011, 11.0, 12.0, 12.0], [0.455500007, 13.0, 14.0, 6.0], [0.343773544, 0.0, 0.0, 0.0], [0.813500047, 15.0, 16.0, 6.0], [3886659070.0, 17.0, 18.0, 4.0], [0.212500006, 19.0, 20.0, 13.0], [2.46499985e-06, 21.0, 22.0, 11.0], [0.185816601, 0.0, 0.0, 0.0], [-0.0364990719, 0.0, 0.0, 0.0], [-0.184715718, 0.0, 0.0, 0.0], [0.328924745, 0.0, 0.0, 0.0], [-0.28863883, 0.0, 0.0, 0.0], [0.103694819, 0.0, 0.0, 0.0], [-0.303727835, 0.0, 0.0, 0.0], [-0.00768666482, 0.0, 0.0, 0.0], [0.337530553, 0.0, 0.0, 0.0], [0.013786423, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([13, 14, 8, 4, 15, 16, 17, 18, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 7, 2, 5, 9, 10, 6, 11, 12])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.644500017, 1.0, 2.0, 5.0], [0.75150001, 3.0, 4.0, 6.0], [-6.41899967, 5.0, 6.0, 8.0], [2944280060.0, 7.0, 8.0, 3.0], [0.827000022, 9.0, 10.0, 6.0], [-6.65400028, 11.0, 12.0, 8.0], [88.9550018, 13.0, 14.0, 14.0], [3867332350.0, 15.0, 16.0, 4.0], [3490279170.0, 17.0, 18.0, 3.0], [3652687620.0, 19.0, 20.0, 2.0], [224674.0, 21.0, 22.0, 15.0], [140860.0, 23.0, 24.0, 15.0], [-0.499797106, 0.0, 0.0, 0.0], [-4.35649967, 25.0, 26.0, 8.0], [2698802690.0, 27.0, 28.0, 4.0], [-0.183856383, 0.0, 0.0, 0.0], [0.172577441, 0.0, 0.0, 0.0], [-0.566427946, 0.0, 0.0, 0.0], [-0.126047119, 0.0, 0.0, 0.0], [0.313497692, 0.0, 0.0, 0.0], [-0.248786896, 0.0, 0.0, 0.0], [0.0160268378, 0.0, 0.0, 0.0], [-0.284960896, 0.0, 0.0, 0.0], [-0.28634572, 0.0, 0.0, 0.0], [0.087192066, 0.0, 0.0, 0.0], [-0.0288220588, 0.0, 0.0, 0.0], [-0.290916234, 0.0, 0.0, 0.0], [0.279302239, 0.0, 0.0, 0.0], [0.0758599639, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 12, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.644500017, 1.0, 2.0, 5.0], [0.75150001, 3.0, 4.0, 6.0], [-6.41899967, 5.0, 6.0, 8.0], [2944280060.0, 7.0, 8.0, 3.0], [0.827000022, 9.0, 10.0, 6.0], [-6.65400028, 11.0, 12.0, 8.0], [88.9550018, 13.0, 14.0, 14.0], [3867332350.0, 15.0, 16.0, 4.0], [3490279170.0, 17.0, 18.0, 3.0], [3652687620.0, 19.0, 20.0, 2.0], [224674.0, 21.0, 22.0, 15.0], [140860.0, 23.0, 24.0, 15.0], [0.499797076, 0.0, 0.0, 0.0], [-4.35649967, 25.0, 26.0, 8.0], [2698802690.0, 27.0, 28.0, 4.0], [0.183856383, 0.0, 0.0, 0.0], [-0.172577411, 0.0, 0.0, 0.0], [0.566427946, 0.0, 0.0, 0.0], [0.126047105, 0.0, 0.0, 0.0], [-0.313497692, 0.0, 0.0, 0.0], [0.248786941, 0.0, 0.0, 0.0], [-0.0160268415, 0.0, 0.0, 0.0], [0.284960896, 0.0, 0.0, 0.0], [0.28634572, 0.0, 0.0, 0.0], [-0.087192066, 0.0, 0.0, 0.0], [0.0288220346, 0.0, 0.0, 0.0], [0.290916264, 0.0, 0.0, 0.0], [-0.279302239, 0.0, 0.0, 0.0], [-0.075859949, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 12, 25, 26, 27, 28])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[720303744.0, 1.0, 2.0, 1.0], [537857984.0, 3.0, 4.0, 2.0], [980076160.0, 5.0, 6.0, 1.0], [334.0, 7.0, 8.0, 0.0], [228240.0, 9.0, 10.0, 15.0], [953985920.0, 11.0, 12.0, 1.0], [0.774500012, 13.0, 14.0, 5.0], [-0.0413249657, 0.0, 0.0, 0.0], [-0.343092382, 0.0, 0.0, 0.0], [0.654999971, 15.0, 16.0, 13.0], [-0.194199622, 0.0, 0.0, 0.0], [0.164499998, 17.0, 18.0, 10.0], [-0.68048346, 0.0, 0.0, 0.0], [-3.48250008, 19.0, 20.0, 8.0], [855.5, 21.0, 22.0, 0.0], [0.364162117, 0.0, 0.0, 0.0], [0.0816825852, 0.0, 0.0, 0.0], [-0.228409514, 0.0, 0.0, 0.0], [0.205988139, 0.0, 0.0, 0.0], [-0.0750294402, 0.0, 0.0, 0.0], [0.173396096, 0.0, 0.0, 0.0], [0.29015249, 0.0, 0.0, 0.0], [-0.0516814031, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 10, 17, 18, 12, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 4, 9, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[720303744.0, 1.0, 2.0, 1.0], [537857984.0, 3.0, 4.0, 2.0], [980076160.0, 5.0, 6.0, 1.0], [334.0, 7.0, 8.0, 0.0], [228240.0, 9.0, 10.0, 15.0], [953985920.0, 11.0, 12.0, 1.0], [0.774500012, 13.0, 14.0, 5.0], [0.0413250215, 0.0, 0.0, 0.0], [0.343092382, 0.0, 0.0, 0.0], [0.654999971, 15.0, 16.0, 13.0], [0.194199651, 0.0, 0.0, 0.0], [0.164499998, 17.0, 18.0, 10.0], [0.68048346, 0.0, 0.0, 0.0], [-3.48250008, 19.0, 20.0, 8.0], [855.5, 21.0, 22.0, 0.0], [-0.364162117, 0.0, 0.0, 0.0], [-0.0816825926, 0.0, 0.0, 0.0], [0.228409484, 0.0, 0.0, 0.0], [-0.205988154, 0.0, 0.0, 0.0], [0.0750294477, 0.0, 0.0, 0.0], [-0.173396096, 0.0, 0.0, 0.0], [-0.29015246, 0.0, 0.0, 0.0], [0.0516813882, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 10, 17, 18, 12, 19, 20, 21, 22])
    branch_indices = np.array([0, 1, 3, 4, 9, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[326705.0, 1.0, 2.0, 15.0], [0.379500002, 3.0, 4.0, 10.0], [-0.280682057, 0.0, 0.0, 0.0], [169.438995, 5.0, 6.0, 14.0], [0.326863945, 0.0, 0.0, 0.0], [1988751230.0, 7.0, 8.0, 2.0], [0.326999992, 9.0, 10.0, 10.0], [-0.053817194, 0.0, 0.0, 0.0], [0.080071941, 0.0, 0.0, 0.0], [-0.316833824, 0.0, 0.0, 0.0], [0.232085064, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_17(xs):
    #Predicts Class 1
    function_dict = np.array([[326705.0, 1.0, 2.0, 15.0], [0.379500002, 3.0, 4.0, 10.0], [0.280682027, 0.0, 0.0, 0.0], [169.438995, 5.0, 6.0, 14.0], [-0.326863945, 0.0, 0.0, 0.0], [1988751230.0, 7.0, 8.0, 2.0], [0.326999992, 9.0, 10.0, 10.0], [0.053817194, 0.0, 0.0, 0.0], [-0.0800719336, 0.0, 0.0, 0.0], [0.316833764, 0.0, 0.0, 0.0], [-0.232085079, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_18(xs):
    #Predicts Class 0
    function_dict = np.array([[0.49849999, 1.0, 2.0, 6.0], [151280.5, 3.0, 4.0, 15.0], [0.212500006, 5.0, 6.0, 13.0], [130529.0, 7.0, 8.0, 15.0], [0.131999999, 9.0, 10.0, 13.0], [0.91049999, 11.0, 12.0, 6.0], [0.248999998, 13.0, 14.0, 13.0], [-0.0113825155, 0.0, 0.0, 0.0], [0.260646641, 0.0, 0.0, 0.0], [-12.6735001, 15.0, 16.0, 8.0], [0.0956999958, 17.0, 18.0, 12.0], [0.34074223, 0.0, 0.0, 0.0], [-0.173239455, 0.0, 0.0, 0.0], [-5.53100014, 19.0, 20.0, 8.0], [0.0033499999, 21.0, 22.0, 11.0], [-0.174120635, 0.0, 0.0, 0.0], [0.224947169, 0.0, 0.0, 0.0], [-0.107282847, 0.0, 0.0, 0.0], [-0.342940152, 0.0, 0.0, 0.0], [-0.374262214, 0.0, 0.0, 0.0], [-0.009756946, 0.0, 0.0, 0.0], [-0.0138256522, 0.0, 0.0, 0.0], [0.155433536, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
    function_dict = np.array([[0.49849999, 1.0, 2.0, 6.0], [151280.5, 3.0, 4.0, 15.0], [0.212500006, 5.0, 6.0, 13.0], [130529.0, 7.0, 8.0, 15.0], [0.131999999, 9.0, 10.0, 13.0], [0.91049999, 11.0, 12.0, 6.0], [0.248999998, 13.0, 14.0, 13.0], [0.0113825388, 0.0, 0.0, 0.0], [-0.260646641, 0.0, 0.0, 0.0], [-12.6735001, 15.0, 16.0, 8.0], [0.0956999958, 17.0, 18.0, 12.0], [-0.34074226, 0.0, 0.0, 0.0], [0.17323947, 0.0, 0.0, 0.0], [-5.53100014, 19.0, 20.0, 8.0], [0.0033499999, 21.0, 22.0, 11.0], [0.174120605, 0.0, 0.0, 0.0], [-0.224947199, 0.0, 0.0, 0.0], [0.107282817, 0.0, 0.0, 0.0], [0.342940181, 0.0, 0.0, 0.0], [0.374262184, 0.0, 0.0, 0.0], [0.00975693017, 0.0, 0.0, 0.0], [0.0138256438, 0.0, 0.0, 0.0], [-0.15543355, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
    function_dict = np.array([[0.0386499986, 1.0, 2.0, 10.0], [-4.96199989, 3.0, 4.0, 8.0], [0.0516500026, 5.0, 6.0, 10.0], [0.5, 7.0, 8.0, 9.0], [-4.48799992, 9.0, 10.0, 8.0], [7.5, 11.0, 12.0, 7.0], [720303744.0, 13.0, 14.0, 1.0], [0.636999965, 15.0, 16.0, 5.0], [-0.352189034, 0.0, 0.0, 0.0], [0.314102203, 0.0, 0.0, 0.0], [1524157310.0, 17.0, 18.0, 2.0], [-8.3220005, 19.0, 20.0, 8.0], [0.583000004, 21.0, 22.0, 5.0], [-7.66800022, 23.0, 24.0, 8.0], [1083699580.0, 25.0, 26.0, 1.0], [-0.181134552, 0.0, 0.0, 0.0], [0.250520289, 0.0, 0.0, 0.0], [-0.256120116, 0.0, 0.0, 0.0], [0.00608527334, 0.0, 0.0, 0.0], [-0.21165514, 0.0, 0.0, 0.0], [0.387430161, 0.0, 0.0, 0.0], [-0.262955904, 0.0, 0.0, 0.0], [0.0163981598, 0.0, 0.0, 0.0], [-0.0611328408, 0.0, 0.0, 0.0], [0.283167094, 0.0, 0.0, 0.0], [-0.251766533, 0.0, 0.0, 0.0], [-0.00204286189, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_21(xs):
    #Predicts Class 1
    function_dict = np.array([[0.0386499986, 1.0, 2.0, 10.0], [-4.96199989, 3.0, 4.0, 8.0], [0.0516500026, 5.0, 6.0, 10.0], [0.5, 7.0, 8.0, 9.0], [-4.48799992, 9.0, 10.0, 8.0], [7.5, 11.0, 12.0, 7.0], [720303744.0, 13.0, 14.0, 1.0], [0.636999965, 15.0, 16.0, 5.0], [0.352189034, 0.0, 0.0, 0.0], [-0.314102232, 0.0, 0.0, 0.0], [1524157310.0, 17.0, 18.0, 2.0], [-8.3220005, 19.0, 20.0, 8.0], [0.583000004, 21.0, 22.0, 5.0], [-7.66800022, 23.0, 24.0, 8.0], [1083699580.0, 25.0, 26.0, 1.0], [0.181134567, 0.0, 0.0, 0.0], [-0.250520289, 0.0, 0.0, 0.0], [0.256120116, 0.0, 0.0, 0.0], [-0.00608527847, 0.0, 0.0, 0.0], [0.21165511, 0.0, 0.0, 0.0], [-0.387430131, 0.0, 0.0, 0.0], [0.262955904, 0.0, 0.0, 0.0], [-0.0163981505, 0.0, 0.0, 0.0], [0.0611328371, 0.0, 0.0, 0.0], [-0.283167094, 0.0, 0.0, 0.0], [0.251766533, 0.0, 0.0, 0.0], [0.0020428421, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_22(xs):
    #Predicts Class 0
    function_dict = np.array([[0.876000047, 1.0, 2.0, 5.0], [118.983002, 3.0, 4.0, 14.0], [0.260350257, 0.0, 0.0, 0.0], [2987724540.0, 5.0, 6.0, 4.0], [129.367493, 7.0, 8.0, 14.0], [4044801540.0, 9.0, 10.0, 1.0], [238869.5, 11.0, 12.0, 15.0], [0.957000017, 13.0, 14.0, 6.0], [4079426050.0, 15.0, 16.0, 3.0], [-0.208478779, 0.0, 0.0, 0.0], [0.236531213, 0.0, 0.0, 0.0], [0.185312405, 0.0, 0.0, 0.0], [-0.236588404, 0.0, 0.0, 0.0], [0.235313714, 0.0, 0.0, 0.0], [-0.0955900326, 0.0, 0.0, 0.0], [-0.0885345563, 0.0, 0.0, 0.0], [0.273804307, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_23(xs):
    #Predicts Class 1
    function_dict = np.array([[0.876000047, 1.0, 2.0, 5.0], [118.983002, 3.0, 4.0, 14.0], [-0.260350287, 0.0, 0.0, 0.0], [2987724540.0, 5.0, 6.0, 4.0], [129.367493, 7.0, 8.0, 14.0], [4044801540.0, 9.0, 10.0, 1.0], [238869.5, 11.0, 12.0, 15.0], [0.957000017, 13.0, 14.0, 6.0], [4079426050.0, 15.0, 16.0, 3.0], [0.208478779, 0.0, 0.0, 0.0], [-0.236531198, 0.0, 0.0, 0.0], [-0.185312435, 0.0, 0.0, 0.0], [0.236588418, 0.0, 0.0, 0.0], [-0.235313728, 0.0, 0.0, 0.0], [0.0955900699, 0.0, 0.0, 0.0], [0.088534534, 0.0, 0.0, 0.0], [-0.273804307, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_24(xs):
    #Predicts Class 0
    function_dict = np.array([[0.201000005, 1.0, 2.0, 10.0], [0.683500051, 3.0, 4.0, 13.0], [1996417540.0, 5.0, 6.0, 1.0], [906259584.0, 7.0, 8.0, 2.0], [0.744499981, 9.0, 10.0, 5.0], [-7.96000004, 11.0, 12.0, 8.0], [3939341570.0, 13.0, 14.0, 1.0], [1580186370.0, 15.0, 16.0, 1.0], [-8.26449966, 17.0, 18.0, 8.0], [0.5, 19.0, 20.0, 7.0], [-6.01200008, 21.0, 22.0, 8.0], [0.274589568, 0.0, 0.0, 0.0], [8.5, 23.0, 24.0, 7.0], [0.356173038, 0.0, 0.0, 0.0], [-0.0738150701, 0.0, 0.0, 0.0], [0.156867415, 0.0, 0.0, 0.0], [-0.217243239, 0.0, 0.0, 0.0], [-0.161638036, 0.0, 0.0, 0.0], [0.0907699764, 0.0, 0.0, 0.0], [0.105784126, 0.0, 0.0, 0.0], [-0.317048728, 0.0, 0.0, 0.0], [-0.173420116, 0.0, 0.0, 0.0], [0.264709443, 0.0, 0.0, 0.0], [-0.299455434, 0.0, 0.0, 0.0], [0.11053244, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 11, 23, 24, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.201000005, 1.0, 2.0, 10.0], [0.683500051, 3.0, 4.0, 13.0], [1996417540.0, 5.0, 6.0, 1.0], [906259584.0, 7.0, 8.0, 2.0], [0.744499981, 9.0, 10.0, 5.0], [-7.96000004, 11.0, 12.0, 8.0], [3939341570.0, 13.0, 14.0, 1.0], [1580186370.0, 15.0, 16.0, 1.0], [-8.26449966, 17.0, 18.0, 8.0], [0.5, 19.0, 20.0, 7.0], [-6.01200008, 21.0, 22.0, 8.0], [-0.274589568, 0.0, 0.0, 0.0], [8.5, 23.0, 24.0, 7.0], [-0.356173038, 0.0, 0.0, 0.0], [0.073815003, 0.0, 0.0, 0.0], [-0.15686743, 0.0, 0.0, 0.0], [0.217243254, 0.0, 0.0, 0.0], [0.161637992, 0.0, 0.0, 0.0], [-0.0907699764, 0.0, 0.0, 0.0], [-0.105784141, 0.0, 0.0, 0.0], [0.317048728, 0.0, 0.0, 0.0], [0.173420206, 0.0, 0.0, 0.0], [-0.264709443, 0.0, 0.0, 0.0], [0.299455404, 0.0, 0.0, 0.0], [-0.110532425, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 11, 23, 24, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.207500011, 1.0, 2.0, 10.0], [193520.0, 3.0, 4.0, 15.0], [139.983002, 5.0, 6.0, 14.0], [6.58000063e-05, 7.0, 8.0, 11.0], [0.699500024, 9.0, 10.0, 5.0], [0.328500003, 11.0, 12.0, 13.0], [152.466003, 13.0, 14.0, 14.0], [168226.5, 15.0, 16.0, 15.0], [-7.375, 17.0, 18.0, 8.0], [1036344450.0, 19.0, 20.0, 3.0], [0.0491499975, 21.0, 22.0, 10.0], [-0.0561185181, 0.0, 0.0, 0.0], [0.268973798, 0.0, 0.0, 0.0], [-0.248653904, 0.0, 0.0, 0.0], [468.5, 23.0, 24.0, 0.0], [0.0522784926, 0.0, 0.0, 0.0], [-0.274821937, 0.0, 0.0, 0.0], [-0.142589808, 0.0, 0.0, 0.0], [0.226278812, 0.0, 0.0, 0.0], [-0.166944832, 0.0, 0.0, 0.0], [0.0385170169, 0.0, 0.0, 0.0], [-0.00864166487, 0.0, 0.0, 0.0], [0.273128033, 0.0, 0.0, 0.0], [-0.083586365, 0.0, 0.0, 0.0], [0.241800606, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_27(xs):
    #Predicts Class 1
    function_dict = np.array([[0.207500011, 1.0, 2.0, 10.0], [193520.0, 3.0, 4.0, 15.0], [139.983002, 5.0, 6.0, 14.0], [6.58000063e-05, 7.0, 8.0, 11.0], [0.699500024, 9.0, 10.0, 5.0], [0.328500003, 11.0, 12.0, 13.0], [152.466003, 13.0, 14.0, 14.0], [168226.5, 15.0, 16.0, 15.0], [-7.375, 17.0, 18.0, 8.0], [1036344450.0, 19.0, 20.0, 3.0], [0.0491499975, 21.0, 22.0, 10.0], [0.056118466, 0.0, 0.0, 0.0], [-0.268973798, 0.0, 0.0, 0.0], [0.248653904, 0.0, 0.0, 0.0], [468.5, 23.0, 24.0, 0.0], [-0.0522785075, 0.0, 0.0, 0.0], [0.274821937, 0.0, 0.0, 0.0], [0.142589822, 0.0, 0.0, 0.0], [-0.226278841, 0.0, 0.0, 0.0], [0.166944802, 0.0, 0.0, 0.0], [-0.0385169983, 0.0, 0.0, 0.0], [0.0086416658, 0.0, 0.0, 0.0], [-0.273128033, 0.0, 0.0, 0.0], [0.0835863948, 0.0, 0.0, 0.0], [-0.241800591, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_28(xs):
    #Predicts Class 0
    function_dict = np.array([[3818502660.0, 1.0, 2.0, 1.0], [3249240830.0, 3.0, 4.0, 1.0], [4030266880.0, 5.0, 6.0, 1.0], [0.825999975, 7.0, 8.0, 6.0], [0.490999997, 9.0, 10.0, 5.0], [3.5, 11.0, 12.0, 7.0], [138.958008, 13.0, 14.0, 14.0], [-6.49699974, 15.0, 16.0, 8.0], [114.725998, 17.0, 18.0, 14.0], [-0.0569705479, 0.0, 0.0, 0.0], [0.715000033, 19.0, 20.0, 13.0], [0.130559608, 0.0, 0.0, 0.0], [-0.355852932, 0.0, 0.0, 0.0], [0.223383337, 0.0, 0.0, 0.0], [-0.0421164446, 0.0, 0.0, 0.0], [-0.0431994647, 0.0, 0.0, 0.0], [0.124745019, 0.0, 0.0, 0.0], [-0.294366479, 0.0, 0.0, 0.0], [-0.047447782, 0.0, 0.0, 0.0], [0.362606227, 0.0, 0.0, 0.0], [0.046929583, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[3818502660.0, 1.0, 2.0, 1.0], [3249240830.0, 3.0, 4.0, 1.0], [4030266880.0, 5.0, 6.0, 1.0], [0.825999975, 7.0, 8.0, 6.0], [0.490999997, 9.0, 10.0, 5.0], [3.5, 11.0, 12.0, 7.0], [138.958008, 13.0, 14.0, 14.0], [-6.49699974, 15.0, 16.0, 8.0], [114.725998, 17.0, 18.0, 14.0], [0.0569706298, 0.0, 0.0, 0.0], [0.715000033, 19.0, 20.0, 13.0], [-0.130559698, 0.0, 0.0, 0.0], [0.355852932, 0.0, 0.0, 0.0], [-0.223383352, 0.0, 0.0, 0.0], [0.0421164334, 0.0, 0.0, 0.0], [0.0431994274, 0.0, 0.0, 0.0], [-0.124745041, 0.0, 0.0, 0.0], [0.294366479, 0.0, 0.0, 0.0], [0.0474477671, 0.0, 0.0, 0.0], [-0.362606287, 0.0, 0.0, 0.0], [-0.0469295681, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 9, 19, 20, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 10, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.030749999, 1.0, 2.0, 10.0], [6.07500042e-05, 3.0, 4.0, 11.0], [0.301499993, 5.0, 6.0, 11.0], [-0.25670898, 0.0, 0.0, 0.0], [0.0194550958, 0.0, 0.0, 0.0], [0.708000004, 7.0, 8.0, 6.0], [242084.0, 9.0, 10.0, 15.0], [415.5, 11.0, 12.0, 0.0], [-4.26499987, 13.0, 14.0, 8.0], [-0.199663728, 0.0, 0.0, 0.0], [-0.00431397837, 0.0, 0.0, 0.0], [-0.0318282507, 0.0, 0.0, 0.0], [0.149005949, 0.0, 0.0, 0.0], [-0.0622466952, 0.0, 0.0, 0.0], [0.10118147, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_31(xs):
    #Predicts Class 1
    function_dict = np.array([[0.030749999, 1.0, 2.0, 10.0], [6.07500042e-05, 3.0, 4.0, 11.0], [0.301499993, 5.0, 6.0, 11.0], [0.25670895, 0.0, 0.0, 0.0], [-0.0194550678, 0.0, 0.0, 0.0], [0.708000004, 7.0, 8.0, 6.0], [242084.0, 9.0, 10.0, 15.0], [415.5, 11.0, 12.0, 0.0], [-4.26499987, 13.0, 14.0, 8.0], [0.199663684, 0.0, 0.0, 0.0], [0.00431387872, 0.0, 0.0, 0.0], [0.0318282358, 0.0, 0.0, 0.0], [-0.149005949, 0.0, 0.0, 0.0], [0.0622466691, 0.0, 0.0, 0.0], [-0.10118144, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_32(xs):
    #Predicts Class 0
    function_dict = np.array([[111.5, 1.0, 2.0, 0.0], [95.6215057, 3.0, 4.0, 14.0], [241.5, 5.0, 6.0, 0.0], [-0.120884031, 0.0, 0.0, 0.0], [2.12000014e-05, 7.0, 8.0, 11.0], [0.132499993, 9.0, 10.0, 12.0], [591.0, 11.0, 12.0, 0.0], [0.286437809, 0.0, 0.0, 0.0], [0.0338024236, 0.0, 0.0, 0.0], [161.5, 13.0, 14.0, 0.0], [-0.268910378, 0.0, 0.0, 0.0], [419.0, 15.0, 16.0, 0.0], [3364301060.0, 17.0, 18.0, 3.0], [-0.126904979, 0.0, 0.0, 0.0], [0.0964965746, 0.0, 0.0, 0.0], [-0.0215653013, 0.0, 0.0, 0.0], [0.176283926, 0.0, 0.0, 0.0], [0.00880003721, 0.0, 0.0, 0.0], [-0.168672711, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_33(xs):
    #Predicts Class 1
    function_dict = np.array([[111.5, 1.0, 2.0, 0.0], [95.6215057, 3.0, 4.0, 14.0], [241.5, 5.0, 6.0, 0.0], [0.120884039, 0.0, 0.0, 0.0], [2.12000014e-05, 7.0, 8.0, 11.0], [0.132499993, 9.0, 10.0, 12.0], [591.0, 11.0, 12.0, 0.0], [-0.28643775, 0.0, 0.0, 0.0], [-0.0338024832, 0.0, 0.0, 0.0], [161.5, 13.0, 14.0, 0.0], [0.268910259, 0.0, 0.0, 0.0], [419.0, 15.0, 16.0, 0.0], [3364301060.0, 17.0, 18.0, 3.0], [0.126904935, 0.0, 0.0, 0.0], [-0.0964966416, 0.0, 0.0, 0.0], [0.021565333, 0.0, 0.0, 0.0], [-0.176283956, 0.0, 0.0, 0.0], [-0.00880003348, 0.0, 0.0, 0.0], [0.168672696, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_34(xs):
    #Predicts Class 0
    function_dict = np.array([[1139.0, 1.0, 2.0, 0.0], [244068.0, 3.0, 4.0, 15.0], [2227771390.0, 5.0, 6.0, 1.0], [222032.5, 7.0, 8.0, 15.0], [137.487, 9.0, 10.0, 14.0], [0.240526155, 0.0, 0.0, 0.0], [-0.0733689442, 0.0, 0.0, 0.0], [2.63000002e-05, 11.0, 12.0, 11.0], [1227374460.0, 13.0, 14.0, 3.0], [673994496.0, 15.0, 16.0, 4.0], [155.061996, 17.0, 18.0, 14.0], [-0.0681285113, 0.0, 0.0, 0.0], [0.0761382282, 0.0, 0.0, 0.0], [-0.0809646174, 0.0, 0.0, 0.0], [0.21111995, 0.0, 0.0, 0.0], [-0.0114098089, 0.0, 0.0, 0.0], [-0.235132799, 0.0, 0.0, 0.0], [0.134049952, 0.0, 0.0, 0.0], [-0.0710292831, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 15, 16, 17, 18, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[1139.0, 1.0, 2.0, 0.0], [244068.0, 3.0, 4.0, 15.0], [2227771390.0, 5.0, 6.0, 1.0], [222032.5, 7.0, 8.0, 15.0], [137.487, 9.0, 10.0, 14.0], [-0.240526125, 0.0, 0.0, 0.0], [0.0733688995, 0.0, 0.0, 0.0], [2.63000002e-05, 11.0, 12.0, 11.0], [1227374460.0, 13.0, 14.0, 3.0], [673994496.0, 15.0, 16.0, 4.0], [155.061996, 17.0, 18.0, 14.0], [0.0681285262, 0.0, 0.0, 0.0], [-0.0761382505, 0.0, 0.0, 0.0], [0.0809646472, 0.0, 0.0, 0.0], [-0.211119965, 0.0, 0.0, 0.0], [0.0114098731, 0.0, 0.0, 0.0], [0.235132754, 0.0, 0.0, 0.0], [-0.134049878, 0.0, 0.0, 0.0], [0.0710293129, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 13, 14, 15, 16, 17, 18, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.547500014, 1.0, 2.0, 5.0], [0.882500052, 3.0, 4.0, 6.0], [0.930999994, 5.0, 6.0, 6.0], [0.502499998, 7.0, 8.0, 5.0], [0.0821499974, 9.0, 10.0, 12.0], [129.977509, 11.0, 12.0, 14.0], [119.499001, 13.0, 14.0, 14.0], [0.000226500008, 15.0, 16.0, 11.0], [-0.186237395, 0.0, 0.0, 0.0], [0.065792352, 0.0, 0.0, 0.0], [536447104.0, 17.0, 18.0, 4.0], [2452249340.0, 19.0, 20.0, 3.0], [145.033997, 21.0, 22.0, 14.0], [0.0184588078, 0.0, 0.0, 0.0], [0.229364514, 0.0, 0.0, 0.0], [0.210176185, 0.0, 0.0, 0.0], [-0.148607716, 0.0, 0.0, 0.0], [-0.0511999354, 0.0, 0.0, 0.0], [-0.266670436, 0.0, 0.0, 0.0], [0.000736507296, 0.0, 0.0, 0.0], [0.166695818, 0.0, 0.0, 0.0], [-0.166250437, 0.0, 0.0, 0.0], [0.0795473605, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 9, 17, 18, 19, 20, 21, 22, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.547500014, 1.0, 2.0, 5.0], [0.882500052, 3.0, 4.0, 6.0], [0.930999994, 5.0, 6.0, 6.0], [0.502499998, 7.0, 8.0, 5.0], [0.0821499974, 9.0, 10.0, 12.0], [129.977509, 11.0, 12.0, 14.0], [119.499001, 13.0, 14.0, 14.0], [0.000226500008, 15.0, 16.0, 11.0], [0.186237395, 0.0, 0.0, 0.0], [-0.0657923371, 0.0, 0.0, 0.0], [536447104.0, 17.0, 18.0, 4.0], [2452249340.0, 19.0, 20.0, 3.0], [145.033997, 21.0, 22.0, 14.0], [-0.0184588041, 0.0, 0.0, 0.0], [-0.229364529, 0.0, 0.0, 0.0], [-0.210176185, 0.0, 0.0, 0.0], [0.148607686, 0.0, 0.0, 0.0], [0.0511999577, 0.0, 0.0, 0.0], [0.266670436, 0.0, 0.0, 0.0], [-0.000736502698, 0.0, 0.0, 0.0], [-0.166695803, 0.0, 0.0, 0.0], [0.166250482, 0.0, 0.0, 0.0], [-0.0795473456, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 8, 9, 17, 18, 19, 20, 21, 22, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 4, 10, 2, 5, 11, 12, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[4030266880.0, 1.0, 2.0, 1.0], [3935276290.0, 3.0, 4.0, 1.0], [198115.0, 5.0, 6.0, 15.0], [913.5, 7.0, 8.0, 0.0], [-0.213476703, 0.0, 0.0, 0.0], [0.20641619, 0.0, 0.0, 0.0], [0.000570913951, 0.0, 0.0, 0.0], [712.5, 9.0, 10.0, 0.0], [0.25999999, 11.0, 12.0, 12.0], [0.00834053103, 0.0, 0.0, 0.0], [-0.149982542, 0.0, 0.0, 0.0], [0.0125040542, 0.0, 0.0, 0.0], [0.249768525, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_39(xs):
    #Predicts Class 1
    function_dict = np.array([[4030266880.0, 1.0, 2.0, 1.0], [3935276290.0, 3.0, 4.0, 1.0], [198115.0, 5.0, 6.0, 15.0], [913.5, 7.0, 8.0, 0.0], [0.213476673, 0.0, 0.0, 0.0], [-0.206416175, 0.0, 0.0, 0.0], [-0.00057093025, 0.0, 0.0, 0.0], [712.5, 9.0, 10.0, 0.0], [0.25999999, 11.0, 12.0, 12.0], [-0.00834054872, 0.0, 0.0, 0.0], [0.149982542, 0.0, 0.0, 0.0], [-0.0125040943, 0.0, 0.0, 0.0], [-0.249768406, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_40(xs):
    #Predicts Class 0
    function_dict = np.array([[169.152008, 1.0, 2.0, 14.0], [118.983002, 3.0, 4.0, 14.0], [435.0, 5.0, 6.0, 0.0], [0.229499996, 7.0, 8.0, 10.0], [0.854499996, 9.0, 10.0, 13.0], [-0.253283411, 0.0, 0.0, 0.0], [0.0265601315, 0.0, 0.0, 0.0], [0.44600001, 11.0, 12.0, 13.0], [0.152352765, 0.0, 0.0, 0.0], [5.20000015e-07, 13.0, 14.0, 11.0], [-0.14597401, 0.0, 0.0, 0.0], [0.0545917191, 0.0, 0.0, 0.0], [-0.166973218, 0.0, 0.0, 0.0], [-0.00128264364, 0.0, 0.0, 0.0], [0.135258347, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 13, 14, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[169.152008, 1.0, 2.0, 14.0], [118.983002, 3.0, 4.0, 14.0], [435.0, 5.0, 6.0, 0.0], [0.229499996, 7.0, 8.0, 10.0], [0.854499996, 9.0, 10.0, 13.0], [0.253283411, 0.0, 0.0, 0.0], [-0.0265601538, 0.0, 0.0, 0.0], [0.44600001, 11.0, 12.0, 13.0], [-0.152352735, 0.0, 0.0, 0.0], [5.20000015e-07, 13.0, 14.0, 11.0], [0.14597398, 0.0, 0.0, 0.0], [-0.0545917191, 0.0, 0.0, 0.0], [0.166973174, 0.0, 0.0, 0.0], [0.00128265878, 0.0, 0.0, 0.0], [-0.135258391, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([11, 12, 8, 13, 14, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 7, 4, 9, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.574000001, 1.0, 2.0, 5.0], [3626939140.0, 3.0, 4.0, 2.0], [-6.45699978, 5.0, 6.0, 8.0], [0.141000003, 7.0, 8.0, 12.0], [-0.193791285, 0.0, 0.0, 0.0], [1537145090.0, 9.0, 10.0, 1.0], [1701828480.0, 11.0, 12.0, 1.0], [0.0811000019, 13.0, 14.0, 12.0], [0.289499998, 15.0, 16.0, 13.0], [3379023360.0, 17.0, 18.0, 2.0], [3361746430.0, 19.0, 20.0, 1.0], [737191488.0, 21.0, 22.0, 1.0], [3400752900.0, 23.0, 24.0, 3.0], [-0.111928158, 0.0, 0.0, 0.0], [0.0883271843, 0.0, 0.0, 0.0], [0.04339917, 0.0, 0.0, 0.0], [-0.170101717, 0.0, 0.0, 0.0], [0.169104636, 0.0, 0.0, 0.0], [-0.0984221324, 0.0, 0.0, 0.0], [-0.173163936, 0.0, 0.0, 0.0], [0.0304772966, 0.0, 0.0, 0.0], [0.134763047, 0.0, 0.0, 0.0], [-0.0716749802, 0.0, 0.0, 0.0], [0.216889545, 0.0, 0.0, 0.0], [-0.126330972, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_43(xs):
    #Predicts Class 1
    function_dict = np.array([[0.574000001, 1.0, 2.0, 5.0], [3626939140.0, 3.0, 4.0, 2.0], [-6.45699978, 5.0, 6.0, 8.0], [0.141000003, 7.0, 8.0, 12.0], [0.1937913, 0.0, 0.0, 0.0], [1537145090.0, 9.0, 10.0, 1.0], [1701828480.0, 11.0, 12.0, 1.0], [0.0811000019, 13.0, 14.0, 12.0], [0.289499998, 15.0, 16.0, 13.0], [3379023360.0, 17.0, 18.0, 2.0], [3361746430.0, 19.0, 20.0, 1.0], [737191488.0, 21.0, 22.0, 1.0], [3400752900.0, 23.0, 24.0, 3.0], [0.111928172, 0.0, 0.0, 0.0], [-0.0883272216, 0.0, 0.0, 0.0], [-0.0433992408, 0.0, 0.0, 0.0], [0.170101717, 0.0, 0.0, 0.0], [-0.169104621, 0.0, 0.0, 0.0], [0.0984221175, 0.0, 0.0, 0.0], [0.17316395, 0.0, 0.0, 0.0], [-0.0304772798, 0.0, 0.0, 0.0], [-0.134763062, 0.0, 0.0, 0.0], [0.0716749802, 0.0, 0.0, 0.0], [-0.21688953, 0.0, 0.0, 0.0], [0.126331061, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_44(xs):
    #Predicts Class 0
    function_dict = np.array([[0.876000047, 1.0, 2.0, 5.0], [-9.50799942, 3.0, 4.0, 8.0], [0.164933756, 0.0, 0.0, 0.0], [0.107500002, 5.0, 6.0, 10.0], [-7.91899967, 7.0, 8.0, 8.0], [-0.204836756, 0.0, 0.0, 0.0], [0.0680696592, 0.0, 0.0, 0.0], [4.0, 9.0, 10.0, 7.0], [-7.23600006, 11.0, 12.0, 8.0], [0.201648742, 0.0, 0.0, 0.0], [-0.00890831277, 0.0, 0.0, 0.0], [-0.152954459, 0.0, 0.0, 0.0], [0.00512216613, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_45(xs):
    #Predicts Class 1
    function_dict = np.array([[0.876000047, 1.0, 2.0, 5.0], [-9.50799942, 3.0, 4.0, 8.0], [-0.164933756, 0.0, 0.0, 0.0], [0.107500002, 5.0, 6.0, 10.0], [-7.91899967, 7.0, 8.0, 8.0], [0.204836741, 0.0, 0.0, 0.0], [-0.0680697262, 0.0, 0.0, 0.0], [4.0, 9.0, 10.0, 7.0], [-7.23600006, 11.0, 12.0, 8.0], [-0.201648757, 0.0, 0.0, 0.0], [0.00890831277, 0.0, 0.0, 0.0], [0.152954489, 0.0, 0.0, 0.0], [-0.00512215728, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_46(xs):
    #Predicts Class 0
    function_dict = np.array([[0.579999983, 1.0, 2.0, 5.0], [0.5, 3.0, 4.0, 9.0], [140.196991, 5.0, 6.0, 14.0], [120.651505, 7.0, 8.0, 14.0], [820.0, 9.0, 10.0, 0.0], [2752028670.0, 11.0, 12.0, 3.0], [143.725494, 13.0, 14.0, 14.0], [-0.0809508339, 0.0, 0.0, 0.0], [1.20000004e-06, 15.0, 16.0, 11.0], [-0.201941952, 0.0, 0.0, 0.0], [0.0597499982, 17.0, 18.0, 10.0], [0.815500021, 19.0, 20.0, 6.0], [0.0384499989, 21.0, 22.0, 10.0], [-0.217944399, 0.0, 0.0, 0.0], [322.5, 23.0, 24.0, 0.0], [0.157630131, 0.0, 0.0, 0.0], [0.00034273637, 0.0, 0.0, 0.0], [0.131283984, 0.0, 0.0, 0.0], [-0.140203267, 0.0, 0.0, 0.0], [0.0529448427, 0.0, 0.0, 0.0], [-0.113664538, 0.0, 0.0, 0.0], [-0.0222093947, 0.0, 0.0, 0.0], [0.17108883, 0.0, 0.0, 0.0], [-0.130063474, 0.0, 0.0, 0.0], [0.117023811, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 9, 17, 18, 19, 20, 21, 22, 13, 23, 24])
    branch_indices = np.array([0, 1, 3, 8, 4, 10, 2, 5, 11, 12, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.579999983, 1.0, 2.0, 5.0], [0.5, 3.0, 4.0, 9.0], [140.196991, 5.0, 6.0, 14.0], [120.651505, 7.0, 8.0, 14.0], [820.0, 9.0, 10.0, 0.0], [2752028670.0, 11.0, 12.0, 3.0], [143.725494, 13.0, 14.0, 14.0], [0.080950819, 0.0, 0.0, 0.0], [1.20000004e-06, 15.0, 16.0, 11.0], [0.201941922, 0.0, 0.0, 0.0], [0.0597499982, 17.0, 18.0, 10.0], [0.815500021, 19.0, 20.0, 6.0], [0.0384499989, 21.0, 22.0, 10.0], [0.217944413, 0.0, 0.0, 0.0], [322.5, 23.0, 24.0, 0.0], [-0.157630101, 0.0, 0.0, 0.0], [-0.000342770014, 0.0, 0.0, 0.0], [-0.131284028, 0.0, 0.0, 0.0], [0.140203223, 0.0, 0.0, 0.0], [-0.0529448166, 0.0, 0.0, 0.0], [0.11366459, 0.0, 0.0, 0.0], [0.0222094003, 0.0, 0.0, 0.0], [-0.171088815, 0.0, 0.0, 0.0], [0.130063504, 0.0, 0.0, 0.0], [-0.117023811, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 15, 16, 9, 17, 18, 19, 20, 21, 22, 13, 23, 24])
    branch_indices = np.array([0, 1, 3, 8, 4, 10, 2, 5, 11, 12, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-3.11000013, 1.0, 2.0, 8.0], [0.694999993, 3.0, 4.0, 13.0], [0.111273386, 0.0, 0.0, 0.0], [0.0615499988, 5.0, 6.0, 10.0], [0.0948000029, 7.0, 8.0, 10.0], [0.0461999997, 9.0, 10.0, 10.0], [775840768.0, 11.0, 12.0, 1.0], [0.5, 13.0, 14.0, 9.0], [0.0972997546, 0.0, 0.0, 0.0], [-0.000371586415, 0.0, 0.0, 0.0], [0.175823689, 0.0, 0.0, 0.0], [0.136180788, 0.0, 0.0, 0.0], [-0.0757634267, 0.0, 0.0, 0.0], [-0.0106143542, 0.0, 0.0, 0.0], [-0.23018983, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_49(xs):
    #Predicts Class 1
    function_dict = np.array([[-3.11000013, 1.0, 2.0, 8.0], [0.694999993, 3.0, 4.0, 13.0], [-0.111273415, 0.0, 0.0, 0.0], [0.0615499988, 5.0, 6.0, 10.0], [0.0948000029, 7.0, 8.0, 10.0], [0.0461999997, 9.0, 10.0, 10.0], [775840768.0, 11.0, 12.0, 1.0], [0.5, 13.0, 14.0, 9.0], [-0.0972997919, 0.0, 0.0, 0.0], [0.0003715645, 0.0, 0.0, 0.0], [-0.175823539, 0.0, 0.0, 0.0], [-0.136180803, 0.0, 0.0, 0.0], [0.0757634267, 0.0, 0.0, 0.0], [0.0106142545, 0.0, 0.0, 0.0], [0.23018989, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_50(xs):
    #Predicts Class 0
    function_dict = np.array([[0.579999983, 1.0, 2.0, 5.0], [0.212500006, 3.0, 4.0, 13.0], [-4.43300009, 5.0, 6.0, 8.0], [0.0638980344, 0.0, 0.0, 0.0], [0.0497500002, 7.0, 8.0, 10.0], [10.5, 9.0, 10.0, 7.0], [114.725998, 11.0, 12.0, 14.0], [-0.167350367, 0.0, 0.0, 0.0], [0.0593999997, 13.0, 14.0, 10.0], [697947008.0, 15.0, 16.0, 2.0], [0.494000018, 17.0, 18.0, 13.0], [-0.02465735, 0.0, 0.0, 0.0], [0.186098725, 0.0, 0.0, 0.0], [0.108967327, 0.0, 0.0, 0.0], [-0.0839608088, 0.0, 0.0, 0.0], [0.142515659, 0.0, 0.0, 0.0], [-0.00611706171, 0.0, 0.0, 0.0], [-0.0251597334, 0.0, 0.0, 0.0], [-0.142934278, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 13, 14, 15, 16, 17, 18, 11, 12])
    branch_indices = np.array([0, 1, 4, 8, 2, 5, 9, 10, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.579999983, 1.0, 2.0, 5.0], [0.212500006, 3.0, 4.0, 13.0], [-4.43300009, 5.0, 6.0, 8.0], [-0.0638980344, 0.0, 0.0, 0.0], [0.0497500002, 7.0, 8.0, 10.0], [10.5, 9.0, 10.0, 7.0], [114.725998, 11.0, 12.0, 14.0], [0.167350411, 0.0, 0.0, 0.0], [0.0593999997, 13.0, 14.0, 10.0], [697947008.0, 15.0, 16.0, 2.0], [0.494000018, 17.0, 18.0, 13.0], [0.0246572718, 0.0, 0.0, 0.0], [-0.186098769, 0.0, 0.0, 0.0], [-0.108967304, 0.0, 0.0, 0.0], [0.0839608088, 0.0, 0.0, 0.0], [-0.142515674, 0.0, 0.0, 0.0], [0.00611706078, 0.0, 0.0, 0.0], [0.0251597278, 0.0, 0.0, 0.0], [0.142934382, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 13, 14, 15, 16, 17, 18, 11, 12])
    branch_indices = np.array([0, 1, 4, 8, 2, 5, 9, 10, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[248251.5, 1.0, 2.0, 15.0], [140.196991, 3.0, 4.0, 14.0], [1.5, 5.0, 6.0, 7.0], [0.5, 7.0, 8.0, 9.0], [0.000104549996, 9.0, 10.0, 11.0], [0.0602265373, 0.0, 0.0, 0.0], [0.0566499978, 11.0, 12.0, 10.0], [10.5, 13.0, 14.0, 7.0], [173720.0, 15.0, 16.0, 15.0], [162.319504, 17.0, 18.0, 14.0], [0.0999437049, 0.0, 0.0, 0.0], [-0.19816196, 0.0, 0.0, 0.0], [-0.016802486, 0.0, 0.0, 0.0], [0.145822063, 0.0, 0.0, 0.0], [-0.0984233692, 0.0, 0.0, 0.0], [-0.119148634, 0.0, 0.0, 0.0], [0.0577588081, 0.0, 0.0, 0.0], [-0.16327697, 0.0, 0.0, 0.0], [0.015394426, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_53(xs):
    #Predicts Class 1
    function_dict = np.array([[248251.5, 1.0, 2.0, 15.0], [140.196991, 3.0, 4.0, 14.0], [1.5, 5.0, 6.0, 7.0], [0.5, 7.0, 8.0, 9.0], [0.000104549996, 9.0, 10.0, 11.0], [-0.0602265187, 0.0, 0.0, 0.0], [0.0566499978, 11.0, 12.0, 10.0], [10.5, 13.0, 14.0, 7.0], [173720.0, 15.0, 16.0, 15.0], [162.319504, 17.0, 18.0, 14.0], [-0.0999437347, 0.0, 0.0, 0.0], [0.19816196, 0.0, 0.0, 0.0], [0.0168026164, 0.0, 0.0, 0.0], [-0.145822063, 0.0, 0.0, 0.0], [0.0984234065, 0.0, 0.0, 0.0], [0.119148649, 0.0, 0.0, 0.0], [-0.0577588007, 0.0, 0.0, 0.0], [0.163277, 0.0, 0.0, 0.0], [-0.0153943747, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_54(xs):
    #Predicts Class 0
    function_dict = np.array([[-9.50799942, 1.0, 2.0, 8.0], [0.00164999999, 3.0, 4.0, 11.0], [0.184, 5.0, 6.0, 13.0], [0.02120951, 0.0, 0.0, 0.0], [-0.178261295, 0.0, 0.0, 0.0], [0.120553344, 0.0, 0.0, 0.0], [0.0792500004, 7.0, 8.0, 10.0], [0.0605999976, 9.0, 10.0, 10.0], [447967872.0, 11.0, 12.0, 3.0], [-0.00134132814, 0.0, 0.0, 0.0], [-0.126479656, 0.0, 0.0, 0.0], [-0.10505452, 0.0, 0.0, 0.0], [0.068575412, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_55(xs):
    #Predicts Class 1
    function_dict = np.array([[-9.50799942, 1.0, 2.0, 8.0], [0.00164999999, 3.0, 4.0, 11.0], [0.184, 5.0, 6.0, 13.0], [-0.0212094933, 0.0, 0.0, 0.0], [0.178261235, 0.0, 0.0, 0.0], [-0.120553322, 0.0, 0.0, 0.0], [0.0792500004, 7.0, 8.0, 10.0], [0.0605999976, 9.0, 10.0, 10.0], [447967872.0, 11.0, 12.0, 3.0], [0.00134134258, 0.0, 0.0, 0.0], [0.126479685, 0.0, 0.0, 0.0], [0.10505452, 0.0, 0.0, 0.0], [-0.068575412, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def logit_class_0(xs):
    sum_of_leaf_values = np.zeros(xs.shape[0])
    for booster_index in range(0,56,2):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def logit_class_1(xs):
    sum_of_leaf_values = np.zeros(xs.shape[0])
    for booster_index in range(1,56,2):
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
        model_cap=11
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
