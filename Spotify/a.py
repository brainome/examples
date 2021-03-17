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
# Output of Brainome Table Compiler v0.991.
# Invocation: btc spotify.csv -f RF --yes -ignoreclasses artist
# Total compiler execution time: 0:00:20.81. Finished on: Mar-17-2021 05:00:20.
# This source code requires Python 3.
#
"""
Classifier Type:                    Random Forest
System Type:                         Binary classifier
Training/Validation Split:           50:50%
Best-guess accuracy:                 56.61%
Training accuracy:                   100.00% (612/612 correct)
Validation accuracy:                 84.47% (517/612 correct)
Overall Model accuracy:              92.23% (1129/1224 correct)
Overall Improvement over best guess: 35.62% (of possible 43.39%)
Model capacity (MEC):                12 bits
Generalization ratio:                50.54 bits/bit
Model efficiency:                    2.96%/parameter
System behavior
True Negatives:                      40.20% (492/1224)
True Positives:                      52.04% (637/1224)
False Negatives:                     4.58% (56/1224)
False Positives:                     3.19% (39/1224)
True Pos. Rate/Sensitivity/Recall:   0.92
True Neg. Rate/Specificity:          0.93
Precision:                           0.94
F-1 Measure:                         0.93
False Negative Rate/Miss Rate:       0.08
Critical Success Index:              0.87
Confusion Matrix:
 [40.20% 3.19%]
 [4.58% 52.04%]
Avg. noise resilience per instance:  -1.71dB
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
num_attr = 17
n_classes = 2
transform_true = False

# Preprocessor for CSV files

ignorelabels=["artist",]
ignorecolumns=[]
target=""
important_idxs=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[], trim=False):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=["artist",]
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
    function_dict = np.array([[0.555500031, 1.0, 2.0, 5.0], [0.272000015, 3.0, 4.0, 10.0], [0.0442500003, 5.0, 6.0, 10.0], [-7.03750038, 7.0, 8.0, 8.0], [2525430020.0, 9.0, 10.0, 1.0], [119.237999, 11.0, 12.0, 14.0], [0.703000009, 13.0, 14.0, 13.0], [2.0, 15.0, 16.0, 16.0], [0.800500035, 17.0, 18.0, 6.0], [819345600.0, 19.0, 20.0, 1.0], [0.459571451, 0.0, 0.0, 0.0], [76.7680054, 21.0, 22.0, 14.0], [129.0215, 23.0, 24.0, 14.0], [88.4454956, 25.0, 26.0, 14.0], [0.774500012, 27.0, 28.0, 5.0], [-0.0, 0.0, 0.0, 0.0], [-0.631594479, 0.0, 0.0, 0.0], [0.0194969699, 0.0, 0.0, 0.0], [-0.522762537, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [-0.428933352, 0.0, 0.0, 0.0], [0.321700007, 0.0, 0.0, 0.0], [-0.441549033, 0.0, 0.0, 0.0], [0.477361292, 0.0, 0.0, 0.0], [-0.350945473, 0.0, 0.0, 0.0], [0.0714888945, 0.0, 0.0, 0.0], [0.434729755, 0.0, 0.0, 0.0], [-0.160850003, 0.0, 0.0, 0.0], [0.58978337, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
    function_dict = np.array([[0.555500031, 1.0, 2.0, 5.0], [0.272000015, 3.0, 4.0, 10.0], [0.0442500003, 5.0, 6.0, 10.0], [-7.03750038, 7.0, 8.0, 8.0], [2525430020.0, 9.0, 10.0, 1.0], [119.237999, 11.0, 12.0, 14.0], [0.703000009, 13.0, 14.0, 13.0], [2.0, 15.0, 16.0, 16.0], [0.800500035, 17.0, 18.0, 6.0], [819345600.0, 19.0, 20.0, 1.0], [-0.459571451, 0.0, 0.0, 0.0], [76.7680054, 21.0, 22.0, 14.0], [129.0215, 23.0, 24.0, 14.0], [88.4454956, 25.0, 26.0, 14.0], [0.774500012, 27.0, 28.0, 5.0], [-0.0, 0.0, 0.0, 0.0], [0.631594479, 0.0, 0.0, 0.0], [-0.0194969699, 0.0, 0.0, 0.0], [0.522762537, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [0.428933352, 0.0, 0.0, 0.0], [-0.321700007, 0.0, 0.0, 0.0], [0.441549033, 0.0, 0.0, 0.0], [-0.477361292, 0.0, 0.0, 0.0], [0.350945473, 0.0, 0.0, 0.0], [-0.0714888945, 0.0, 0.0, 0.0], [-0.434729755, 0.0, 0.0, 0.0], [0.160850003, 0.0, 0.0, 0.0], [-0.58978337, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
    function_dict = np.array([[0.601999998, 1.0, 2.0, 5.0], [255015.0, 3.0, 4.0, 15.0], [-6.22300005, 5.0, 6.0, 8.0], [0.453000009, 7.0, 8.0, 13.0], [-4.00250006, 9.0, 10.0, 8.0], [0.130999997, 11.0, 12.0, 10.0], [0.899999976, 13.0, 14.0, 13.0], [0.395999998, 15.0, 16.0, 6.0], [0.973999977, 17.0, 18.0, 6.0], [1156.0, 19.0, 20.0, 0.0], [3259899390.0, 21.0, 22.0, 4.0], [0.71450001, 23.0, 24.0, 13.0], [137.979492, 25.0, 26.0, 14.0], [229520.5, 27.0, 28.0, 15.0], [0.835500002, 29.0, 30.0, 6.0], [-0.378556967, 0.0, 0.0, 0.0], [0.140271798, 0.0, 0.0, 0.0], [-0.322133511, 0.0, 0.0, 0.0], [0.276195228, 0.0, 0.0, 0.0], [-0.443502486, 0.0, 0.0, 0.0], [-0.087617673, 0.0, 0.0, 0.0], [-0.30701378, 0.0, 0.0, 0.0], [0.194387898, 0.0, 0.0, 0.0], [-0.0171221048, 0.0, 0.0, 0.0], [-0.397916317, 0.0, 0.0, 0.0], [0.331955105, 0.0, 0.0, 0.0], [-0.119652212, 0.0, 0.0, 0.0], [0.330538779, 0.0, 0.0, 0.0], [-0.0216725282, 0.0, 0.0, 0.0], [0.0265575424, 0.0, 0.0, 0.0], [-0.445569009, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
    function_dict = np.array([[0.601999998, 1.0, 2.0, 5.0], [255015.0, 3.0, 4.0, 15.0], [-6.22300005, 5.0, 6.0, 8.0], [0.453000009, 7.0, 8.0, 13.0], [-4.00250006, 9.0, 10.0, 8.0], [0.130999997, 11.0, 12.0, 10.0], [0.899999976, 13.0, 14.0, 13.0], [0.395999998, 15.0, 16.0, 6.0], [0.973999977, 17.0, 18.0, 6.0], [1156.0, 19.0, 20.0, 0.0], [3259899390.0, 21.0, 22.0, 4.0], [0.71450001, 23.0, 24.0, 13.0], [137.979492, 25.0, 26.0, 14.0], [229520.5, 27.0, 28.0, 15.0], [0.835500002, 29.0, 30.0, 6.0], [0.378556967, 0.0, 0.0, 0.0], [-0.140271798, 0.0, 0.0, 0.0], [0.322133511, 0.0, 0.0, 0.0], [-0.276195228, 0.0, 0.0, 0.0], [0.443502486, 0.0, 0.0, 0.0], [0.0876176655, 0.0, 0.0, 0.0], [0.30701378, 0.0, 0.0, 0.0], [-0.194387898, 0.0, 0.0, 0.0], [0.0171220973, 0.0, 0.0, 0.0], [0.397916377, 0.0, 0.0, 0.0], [-0.331955105, 0.0, 0.0, 0.0], [0.119652174, 0.0, 0.0, 0.0], [-0.33053875, 0.0, 0.0, 0.0], [0.0216725264, 0.0, 0.0, 0.0], [-0.0265575107, 0.0, 0.0, 0.0], [0.445569009, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
    function_dict = np.array([[0.733999968, 1.0, 2.0, 5.0], [-7.37650013, 3.0, 4.0, 8.0], [800.0, 5.0, 6.0, 0.0], [0.235000014, 7.0, 8.0, 10.0], [0.5, 9.0, 10.0, 9.0], [3440054270.0, 11.0, 12.0, 2.0], [0.760500014, 13.0, 14.0, 6.0], [606487040.0, 15.0, 16.0, 2.0], [2544097280.0, 17.0, 18.0, 1.0], [0.134000003, 19.0, 20.0, 10.0], [0.349000007, 21.0, 22.0, 13.0], [0.367500007, 23.0, 24.0, 12.0], [0.182999998, 25.0, 26.0, 12.0], [943479424.0, 27.0, 28.0, 2.0], [1043.5, 29.0, 30.0, 0.0], [-0.147418976, 0.0, 0.0, 0.0], [-0.384183407, 0.0, 0.0, 0.0], [-0.242890209, 0.0, 0.0, 0.0], [0.325380534, 0.0, 0.0, 0.0], [0.300368816, 0.0, 0.0, 0.0], [-0.164340764, 0.0, 0.0, 0.0], [0.0862201303, 0.0, 0.0, 0.0], [-0.21112664, 0.0, 0.0, 0.0], [0.373925805, 0.0, 0.0, 0.0], [-0.125182465, 0.0, 0.0, 0.0], [0.281996191, 0.0, 0.0, 0.0], [-0.428909898, 0.0, 0.0, 0.0], [-0.237695336, 0.0, 0.0, 0.0], [0.24828814, 0.0, 0.0, 0.0], [-0.709776402, 0.0, 0.0, 0.0], [0.0458893254, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_5(xs):
    #Predicts Class 1
    function_dict = np.array([[0.733999968, 1.0, 2.0, 5.0], [-7.37650013, 3.0, 4.0, 8.0], [800.0, 5.0, 6.0, 0.0], [0.235000014, 7.0, 8.0, 10.0], [0.5, 9.0, 10.0, 9.0], [3440054270.0, 11.0, 12.0, 2.0], [0.760500014, 13.0, 14.0, 6.0], [606487040.0, 15.0, 16.0, 2.0], [2544097280.0, 17.0, 18.0, 1.0], [0.134000003, 19.0, 20.0, 10.0], [0.349000007, 21.0, 22.0, 13.0], [0.367500007, 23.0, 24.0, 12.0], [0.182999998, 25.0, 26.0, 12.0], [943479424.0, 27.0, 28.0, 2.0], [1043.5, 29.0, 30.0, 0.0], [0.147418991, 0.0, 0.0, 0.0], [0.384183407, 0.0, 0.0, 0.0], [0.242890194, 0.0, 0.0, 0.0], [-0.325380564, 0.0, 0.0, 0.0], [-0.300368786, 0.0, 0.0, 0.0], [0.164340734, 0.0, 0.0, 0.0], [-0.0862201229, 0.0, 0.0, 0.0], [0.21112664, 0.0, 0.0, 0.0], [-0.373925805, 0.0, 0.0, 0.0], [0.125182465, 0.0, 0.0, 0.0], [-0.28199622, 0.0, 0.0, 0.0], [0.428909808, 0.0, 0.0, 0.0], [0.237695262, 0.0, 0.0, 0.0], [-0.248288155, 0.0, 0.0, 0.0], [0.709776342, 0.0, 0.0, 0.0], [-0.0458893292, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_6(xs):
    #Predicts Class 0
    function_dict = np.array([[244068.0, 1.0, 2.0, 15.0], [0.547500014, 3.0, 4.0, 5.0], [1.5, 5.0, 6.0, 7.0], [2135151870.0, 7.0, 8.0, 4.0], [-3.14549994, 9.0, 10.0, 8.0], [0.722499967, 11.0, 12.0, 6.0], [221678752.0, 13.0, 14.0, 4.0], [0.108500004, 15.0, 16.0, 10.0], [0.609499991, 17.0, 18.0, 6.0], [0.494000018, 19.0, 20.0, 13.0], [0.42467463, 0.0, 0.0, 0.0], [-0.338460684, 0.0, 0.0, 0.0], [0.08715, 21.0, 22.0, 10.0], [0.08978834, 0.0, 0.0, 0.0], [138.94101, 23.0, 24.0, 14.0], [-0.0841749981, 0.0, 0.0, 0.0], [0.316199303, 0.0, 0.0, 0.0], [-0.0490758084, 0.0, 0.0, 0.0], [-0.414283097, 0.0, 0.0, 0.0], [0.170978606, 0.0, 0.0, 0.0], [-0.0115806451, 0.0, 0.0, 0.0], [0.467435926, 0.0, 0.0, 0.0], [-0.0143100768, 0.0, 0.0, 0.0], [-0.394931614, 0.0, 0.0, 0.0], [-0.155756146, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_7(xs):
    #Predicts Class 1
    function_dict = np.array([[244068.0, 1.0, 2.0, 15.0], [0.547500014, 3.0, 4.0, 5.0], [1.5, 5.0, 6.0, 7.0], [2135151870.0, 7.0, 8.0, 4.0], [-3.14549994, 9.0, 10.0, 8.0], [0.722499967, 11.0, 12.0, 6.0], [221678752.0, 13.0, 14.0, 4.0], [0.108500004, 15.0, 16.0, 10.0], [0.609499991, 17.0, 18.0, 6.0], [0.494000018, 19.0, 20.0, 13.0], [-0.42467463, 0.0, 0.0, 0.0], [0.338460684, 0.0, 0.0, 0.0], [0.08715, 21.0, 22.0, 10.0], [-0.0897883624, 0.0, 0.0, 0.0], [138.94101, 23.0, 24.0, 14.0], [0.0841749832, 0.0, 0.0, 0.0], [-0.316199273, 0.0, 0.0, 0.0], [0.0490757264, 0.0, 0.0, 0.0], [0.414283097, 0.0, 0.0, 0.0], [-0.170978606, 0.0, 0.0, 0.0], [0.0115806665, 0.0, 0.0, 0.0], [-0.467435926, 0.0, 0.0, 0.0], [0.0143100861, 0.0, 0.0, 0.0], [0.394931614, 0.0, 0.0, 0.0], [0.155756116, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_8(xs):
    #Predicts Class 0
    function_dict = np.array([[0.49849999, 1.0, 2.0, 6.0], [0.86500001, 3.0, 4.0, 5.0], [0.212500006, 5.0, 6.0, 13.0], [77.8769989, 7.0, 8.0, 14.0], [0.164921016, 0.0, 0.0, 0.0], [0.91049999, 9.0, 10.0, 6.0], [0.644500017, 11.0, 12.0, 5.0], [0.344500005, 13.0, 14.0, 6.0], [3947772670.0, 15.0, 16.0, 2.0], [0.451273769, 0.0, 0.0, 0.0], [-0.206492051, 0.0, 0.0, 0.0], [2003817600.0, 17.0, 18.0, 4.0], [-5.48500013, 19.0, 20.0, 8.0], [-0.109403953, 0.0, 0.0, 0.0], [0.224617764, 0.0, 0.0, 0.0], [-0.353575677, 0.0, 0.0, 0.0], [0.0484609157, 0.0, 0.0, 0.0], [-0.220474362, 0.0, 0.0, 0.0], [0.0299110133, 0.0, 0.0, 0.0], [-0.0265739728, 0.0, 0.0, 0.0], [0.222858876, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_9(xs):
    #Predicts Class 1
    function_dict = np.array([[0.49849999, 1.0, 2.0, 6.0], [0.86500001, 3.0, 4.0, 5.0], [0.212500006, 5.0, 6.0, 13.0], [77.8769989, 7.0, 8.0, 14.0], [-0.164920986, 0.0, 0.0, 0.0], [0.91049999, 9.0, 10.0, 6.0], [0.644500017, 11.0, 12.0, 5.0], [0.344500005, 13.0, 14.0, 6.0], [3947772670.0, 15.0, 16.0, 2.0], [-0.451273769, 0.0, 0.0, 0.0], [0.206492051, 0.0, 0.0, 0.0], [2003817600.0, 17.0, 18.0, 4.0], [-5.48500013, 19.0, 20.0, 8.0], [0.109403968, 0.0, 0.0, 0.0], [-0.224617764, 0.0, 0.0, 0.0], [0.353575677, 0.0, 0.0, 0.0], [-0.0484608933, 0.0, 0.0, 0.0], [0.220474362, 0.0, 0.0, 0.0], [-0.0299109966, 0.0, 0.0, 0.0], [0.0265739802, 0.0, 0.0, 0.0], [-0.222858906, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_10(xs):
    #Predicts Class 0
    function_dict = np.array([[1.5, 1.0, 2.0, 7.0], [3.76499997e-06, 3.0, 4.0, 11.0], [720303744.0, 5.0, 6.0, 1.0], [0.908499956, 7.0, 8.0, 6.0], [2542232320.0, 9.0, 10.0, 4.0], [537857984.0, 11.0, 12.0, 2.0], [1081964670.0, 13.0, 14.0, 1.0], [218883.0, 15.0, 16.0, 15.0], [-0.075242959, 0.0, 0.0, 0.0], [0.0395999998, 17.0, 18.0, 10.0], [117.968002, 19.0, 20.0, 14.0], [0.511999965, 21.0, 22.0, 13.0], [228240.0, 23.0, 24.0, 15.0], [950123136.0, 25.0, 26.0, 1.0], [2761012740.0, 27.0, 28.0, 2.0], [0.421313375, 0.0, 0.0, 0.0], [0.0965984687, 0.0, 0.0, 0.0], [-0.17027314, 0.0, 0.0, 0.0], [0.215970621, 0.0, 0.0, 0.0], [-0.344799131, 0.0, 0.0, 0.0], [0.0384110659, 0.0, 0.0, 0.0], [0.010407404, 0.0, 0.0, 0.0], [-0.342202932, 0.0, 0.0, 0.0], [0.309965193, 0.0, 0.0, 0.0], [-0.223370984, 0.0, 0.0, 0.0], [-0.0696975738, 0.0, 0.0, 0.0], [-0.531700611, 0.0, 0.0, 0.0], [-0.0959379524, 0.0, 0.0, 0.0], [0.09226989, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_11(xs):
    #Predicts Class 1
    function_dict = np.array([[1.5, 1.0, 2.0, 7.0], [3.76499997e-06, 3.0, 4.0, 11.0], [720303744.0, 5.0, 6.0, 1.0], [0.908499956, 7.0, 8.0, 6.0], [2542232320.0, 9.0, 10.0, 4.0], [537857984.0, 11.0, 12.0, 2.0], [1081964670.0, 13.0, 14.0, 1.0], [218883.0, 15.0, 16.0, 15.0], [0.0752429664, 0.0, 0.0, 0.0], [0.0395999998, 17.0, 18.0, 10.0], [117.968002, 19.0, 20.0, 14.0], [0.511999965, 21.0, 22.0, 13.0], [228240.0, 23.0, 24.0, 15.0], [950123136.0, 25.0, 26.0, 1.0], [2761012740.0, 27.0, 28.0, 2.0], [-0.421313375, 0.0, 0.0, 0.0], [-0.0965984389, 0.0, 0.0, 0.0], [0.170273095, 0.0, 0.0, 0.0], [-0.215970635, 0.0, 0.0, 0.0], [0.344799131, 0.0, 0.0, 0.0], [-0.0384110548, 0.0, 0.0, 0.0], [-0.0104073836, 0.0, 0.0, 0.0], [0.342202932, 0.0, 0.0, 0.0], [-0.309965163, 0.0, 0.0, 0.0], [0.223370984, 0.0, 0.0, 0.0], [0.069697544, 0.0, 0.0, 0.0], [0.531700611, 0.0, 0.0, 0.0], [0.0959379449, 0.0, 0.0, 0.0], [-0.0922699049, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_12(xs):
    #Predicts Class 0
    function_dict = np.array([[0.0388999991, 1.0, 2.0, 10.0], [0.5, 3.0, 4.0, 9.0], [327543296.0, 5.0, 6.0, 4.0], [245753.5, 7.0, 8.0, 15.0], [-4.90550041, 9.0, 10.0, 8.0], [0.126000002, 11.0, 12.0, 10.0], [419.5, 13.0, 14.0, 0.0], [0.0340000018, 15.0, 16.0, 10.0], [-0.243011281, 0.0, 0.0, 0.0], [0.24149999, 17.0, 18.0, 13.0], [0.570500016, 19.0, 20.0, 5.0], [224907.0, 21.0, 22.0, 15.0], [0.488999993, 23.0, 24.0, 13.0], [3677200900.0, 25.0, 26.0, 3.0], [0.583000004, 27.0, 28.0, 5.0], [0.350492954, 0.0, 0.0, 0.0], [-0.049112875, 0.0, 0.0, 0.0], [-0.0535948761, 0.0, 0.0, 0.0], [-0.382949442, 0.0, 0.0, 0.0], [-0.312235445, 0.0, 0.0, 0.0], [0.159198552, 0.0, 0.0, 0.0], [0.496037573, 0.0, 0.0, 0.0], [0.119954005, 0.0, 0.0, 0.0], [-0.270961344, 0.0, 0.0, 0.0], [0.231338233, 0.0, 0.0, 0.0], [-0.158924848, 0.0, 0.0, 0.0], [0.193017408, 0.0, 0.0, 0.0], [-0.057159815, 0.0, 0.0, 0.0], [0.137621298, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
    function_dict = np.array([[0.0388999991, 1.0, 2.0, 10.0], [0.5, 3.0, 4.0, 9.0], [327543296.0, 5.0, 6.0, 4.0], [245753.5, 7.0, 8.0, 15.0], [-4.90550041, 9.0, 10.0, 8.0], [0.126000002, 11.0, 12.0, 10.0], [419.5, 13.0, 14.0, 0.0], [0.0340000018, 15.0, 16.0, 10.0], [0.243011281, 0.0, 0.0, 0.0], [0.24149999, 17.0, 18.0, 13.0], [0.570500016, 19.0, 20.0, 5.0], [224907.0, 21.0, 22.0, 15.0], [0.488999993, 23.0, 24.0, 13.0], [3677200900.0, 25.0, 26.0, 3.0], [0.583000004, 27.0, 28.0, 5.0], [-0.350492954, 0.0, 0.0, 0.0], [0.0491128601, 0.0, 0.0, 0.0], [0.0535948612, 0.0, 0.0, 0.0], [0.382949442, 0.0, 0.0, 0.0], [0.312235445, 0.0, 0.0, 0.0], [-0.159198552, 0.0, 0.0, 0.0], [-0.496037573, 0.0, 0.0, 0.0], [-0.119954005, 0.0, 0.0, 0.0], [0.270961374, 0.0, 0.0, 0.0], [-0.231338188, 0.0, 0.0, 0.0], [0.158924848, 0.0, 0.0, 0.0], [-0.193017423, 0.0, 0.0, 0.0], [0.0571598187, 0.0, 0.0, 0.0], [-0.137621284, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
    function_dict = np.array([[0.879999995, 1.0, 2.0, 5.0], [-7.23600006, 3.0, 4.0, 8.0], [0.303390503, 0.0, 0.0, 0.0], [0.324000001, 5.0, 6.0, 13.0], [550448768.0, 7.0, 8.0, 3.0], [0.131999999, 9.0, 10.0, 13.0], [0.588500023, 11.0, 12.0, 13.0], [47857964.0, 13.0, 14.0, 3.0], [0.5, 15.0, 16.0, 9.0], [-0.0173049364, 0.0, 0.0, 0.0], [-0.469889641, 0.0, 0.0, 0.0], [0.139202848, 0.0, 0.0, 0.0], [-0.215182871, 0.0, 0.0, 0.0], [0.255974203, 0.0, 0.0, 0.0], [-0.249323219, 0.0, 0.0, 0.0], [0.179360837, 0.0, 0.0, 0.0], [-0.015703341, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
    function_dict = np.array([[0.879999995, 1.0, 2.0, 5.0], [-7.23600006, 3.0, 4.0, 8.0], [-0.303390503, 0.0, 0.0, 0.0], [0.324000001, 5.0, 6.0, 13.0], [550448768.0, 7.0, 8.0, 3.0], [0.131999999, 9.0, 10.0, 13.0], [0.588500023, 11.0, 12.0, 13.0], [47857964.0, 13.0, 14.0, 3.0], [0.5, 15.0, 16.0, 9.0], [0.0173049662, 0.0, 0.0, 0.0], [0.469889641, 0.0, 0.0, 0.0], [-0.139202848, 0.0, 0.0, 0.0], [0.215182856, 0.0, 0.0, 0.0], [-0.255974174, 0.0, 0.0, 0.0], [0.249323189, 0.0, 0.0, 0.0], [-0.179360837, 0.0, 0.0, 0.0], [0.0157033335, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
    function_dict = np.array([[313706.5, 1.0, 2.0, 15.0], [0.379500002, 3.0, 4.0, 10.0], [3.11500003e-06, 5.0, 6.0, 11.0], [3249240830.0, 7.0, 8.0, 1.0], [0.317619681, 0.0, 0.0, 0.0], [0.0453117453, 0.0, 0.0, 0.0], [-0.307897329, 0.0, 0.0, 0.0], [1059.0, 9.0, 10.0, 0.0], [3818502660.0, 11.0, 12.0, 1.0], [-0.0636841208, 0.0, 0.0, 0.0], [0.175964549, 0.0, 0.0, 0.0], [0.30042991, 0.0, 0.0, 0.0], [-0.0811490715, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_17(xs):
    #Predicts Class 1
    function_dict = np.array([[313706.5, 1.0, 2.0, 15.0], [0.379500002, 3.0, 4.0, 10.0], [3.11500003e-06, 5.0, 6.0, 11.0], [3249240830.0, 7.0, 8.0, 1.0], [-0.317619681, 0.0, 0.0, 0.0], [-0.0453116968, 0.0, 0.0, 0.0], [0.307897329, 0.0, 0.0, 0.0], [1059.0, 9.0, 10.0, 0.0], [3818502660.0, 11.0, 12.0, 1.0], [0.0636841133, 0.0, 0.0, 0.0], [-0.175964564, 0.0, 0.0, 0.0], [-0.30042991, 0.0, 0.0, 0.0], [0.0811490417, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_18(xs):
    #Predicts Class 0
    function_dict = np.array([[0.773999989, 1.0, 2.0, 5.0], [119.800507, 3.0, 4.0, 14.0], [0.000307500013, 5.0, 6.0, 11.0], [0.799499989, 7.0, 8.0, 6.0], [129.367493, 9.0, 10.0, 14.0], [3837065470.0, 11.0, 12.0, 4.0], [-0.258167088, 0.0, 0.0, 0.0], [0.595499992, 13.0, 14.0, 6.0], [3652268540.0, 15.0, 16.0, 1.0], [0.814999998, 17.0, 18.0, 13.0], [720303744.0, 19.0, 20.0, 1.0], [10.5, 21.0, 22.0, 7.0], [3945333760.0, 23.0, 24.0, 4.0], [-0.162817031, 0.0, 0.0, 0.0], [0.119632244, 0.0, 0.0, 0.0], [-0.349634677, 0.0, 0.0, 0.0], [0.0980795324, 0.0, 0.0, 0.0], [0.23333706, 0.0, 0.0, 0.0], [-0.206356436, 0.0, 0.0, 0.0], [0.190705538, 0.0, 0.0, 0.0], [-0.0661625192, 0.0, 0.0, 0.0], [0.328629196, 0.0, 0.0, 0.0], [-0.0537798665, 0.0, 0.0, 0.0], [-0.304725915, 0.0, 0.0, 0.0], [-0.0176134072, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_19(xs):
    #Predicts Class 1
    function_dict = np.array([[0.773999989, 1.0, 2.0, 5.0], [119.800507, 3.0, 4.0, 14.0], [0.000307500013, 5.0, 6.0, 11.0], [0.799499989, 7.0, 8.0, 6.0], [129.367493, 9.0, 10.0, 14.0], [3837065470.0, 11.0, 12.0, 4.0], [0.258167088, 0.0, 0.0, 0.0], [0.595499992, 13.0, 14.0, 6.0], [3652268540.0, 15.0, 16.0, 1.0], [0.814999998, 17.0, 18.0, 13.0], [720303744.0, 19.0, 20.0, 1.0], [10.5, 21.0, 22.0, 7.0], [3945333760.0, 23.0, 24.0, 4.0], [0.162817046, 0.0, 0.0, 0.0], [-0.119632274, 0.0, 0.0, 0.0], [0.349634677, 0.0, 0.0, 0.0], [-0.0980794579, 0.0, 0.0, 0.0], [-0.233337075, 0.0, 0.0, 0.0], [0.206356436, 0.0, 0.0, 0.0], [-0.190705538, 0.0, 0.0, 0.0], [0.0661625341, 0.0, 0.0, 0.0], [-0.328629196, 0.0, 0.0, 0.0], [0.0537798963, 0.0, 0.0, 0.0], [0.304725915, 0.0, 0.0, 0.0], [0.0176133849, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_20(xs):
    #Predicts Class 0
    function_dict = np.array([[-3.48250008, 1.0, 2.0, 8.0], [0.244500011, 3.0, 4.0, 12.0], [0.949499965, 5.0, 6.0, 6.0], [0.0639500022, 7.0, 8.0, 12.0], [1.5, 9.0, 10.0, 7.0], [0.717000008, 11.0, 12.0, 13.0], [-2.61100006, 13.0, 14.0, 8.0], [457.0, 15.0, 16.0, 0.0], [3364301060.0, 17.0, 18.0, 3.0], [0.516499996, 19.0, 20.0, 5.0], [0.00509000011, 21.0, 22.0, 11.0], [0.0464000031, 23.0, 24.0, 10.0], [0.0526335686, 0.0, 0.0, 0.0], [-0.23262918, 0.0, 0.0, 0.0], [0.0867190734, 0.0, 0.0, 0.0], [0.0368712433, 0.0, 0.0, 0.0], [-0.266339779, 0.0, 0.0, 0.0], [0.0787820593, 0.0, 0.0, 0.0], [-0.0963516608, 0.0, 0.0, 0.0], [-0.104231216, 0.0, 0.0, 0.0], [0.227584258, 0.0, 0.0, 0.0], [-0.214796498, 0.0, 0.0, 0.0], [0.0980161801, 0.0, 0.0, 0.0], [0.0667278767, 0.0, 0.0, 0.0], [0.387233883, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-3.48250008, 1.0, 2.0, 8.0], [0.244500011, 3.0, 4.0, 12.0], [0.949499965, 5.0, 6.0, 6.0], [0.0639500022, 7.0, 8.0, 12.0], [1.5, 9.0, 10.0, 7.0], [0.717000008, 11.0, 12.0, 13.0], [-2.61100006, 13.0, 14.0, 8.0], [457.0, 15.0, 16.0, 0.0], [3364301060.0, 17.0, 18.0, 3.0], [0.516499996, 19.0, 20.0, 5.0], [0.00509000011, 21.0, 22.0, 11.0], [0.0464000031, 23.0, 24.0, 10.0], [-0.0526335239, 0.0, 0.0, 0.0], [0.23262918, 0.0, 0.0, 0.0], [-0.086719051, 0.0, 0.0, 0.0], [-0.0368712284, 0.0, 0.0, 0.0], [0.266339749, 0.0, 0.0, 0.0], [-0.0787820444, 0.0, 0.0, 0.0], [0.0963516682, 0.0, 0.0, 0.0], [0.104231209, 0.0, 0.0, 0.0], [-0.227584258, 0.0, 0.0, 0.0], [0.214796484, 0.0, 0.0, 0.0], [-0.0980161801, 0.0, 0.0, 0.0], [-0.0667278394, 0.0, 0.0, 0.0], [-0.387233913, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 7, 8, 4, 9, 10, 2, 5, 11, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[4030266880.0, 1.0, 2.0, 1.0], [3935276290.0, 3.0, 4.0, 1.0], [198115.0, 5.0, 6.0, 15.0], [0.0386499986, 7.0, 8.0, 10.0], [-0.299806416, 0.0, 0.0, 0.0], [0.318600714, 0.0, 0.0, 0.0], [-0.0581619628, 0.0, 0.0, 0.0], [2.365e-06, 9.0, 10.0, 11.0], [0.0597499982, 11.0, 12.0, 10.0], [-0.231062189, 0.0, 0.0, 0.0], [0.0128267081, 0.0, 0.0, 0.0], [0.129248574, 0.0, 0.0, 0.0], [-0.0161774717, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_23(xs):
    #Predicts Class 1
    function_dict = np.array([[4030266880.0, 1.0, 2.0, 1.0], [3935276290.0, 3.0, 4.0, 1.0], [198115.0, 5.0, 6.0, 15.0], [0.0386499986, 7.0, 8.0, 10.0], [0.299806446, 0.0, 0.0, 0.0], [-0.318600714, 0.0, 0.0, 0.0], [0.0581619143, 0.0, 0.0, 0.0], [2.365e-06, 9.0, 10.0, 11.0], [0.0597499982, 11.0, 12.0, 10.0], [0.231062189, 0.0, 0.0, 0.0], [-0.0128267361, 0.0, 0.0, 0.0], [-0.129248574, 0.0, 0.0, 0.0], [0.016177468, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_24(xs):
    #Predicts Class 0
    function_dict = np.array([[0.863499999, 1.0, 2.0, 13.0], [-13.1929998, 3.0, 4.0, 8.0], [0.689999998, 5.0, 6.0, 5.0], [-0.245861337, 0.0, 0.0, 0.0], [169.152008, 7.0, 8.0, 14.0], [-0.292765141, 0.0, 0.0, 0.0], [-0.019777976, 0.0, 0.0, 0.0], [143.985504, 9.0, 10.0, 14.0], [435.0, 11.0, 12.0, 0.0], [0.0111190686, 0.0, 0.0, 0.0], [0.180383623, 0.0, 0.0, 0.0], [-0.391426682, 0.0, 0.0, 0.0], [0.0167875495, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_25(xs):
    #Predicts Class 1
    function_dict = np.array([[0.863499999, 1.0, 2.0, 13.0], [-13.1929998, 3.0, 4.0, 8.0], [0.689999998, 5.0, 6.0, 5.0], [0.245861307, 0.0, 0.0, 0.0], [169.152008, 7.0, 8.0, 14.0], [0.292765141, 0.0, 0.0, 0.0], [0.0197779909, 0.0, 0.0, 0.0], [143.985504, 9.0, 10.0, 14.0], [435.0, 11.0, 12.0, 0.0], [-0.0111190733, 0.0, 0.0, 0.0], [-0.180383638, 0.0, 0.0, 0.0], [0.391426742, 0.0, 0.0, 0.0], [-0.0167875364, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_26(xs):
    #Predicts Class 0
    function_dict = np.array([[2805971460.0, 1.0, 2.0, 2.0], [2661524990.0, 3.0, 4.0, 2.0], [2948765440.0, 5.0, 6.0, 2.0], [228640.0, 7.0, 8.0, 15.0], [170326.0, 9.0, 10.0, 15.0], [0.348832935, 0.0, 0.0, 0.0], [0.579999983, 11.0, 12.0, 5.0], [0.947000027, 13.0, 14.0, 6.0], [2863348740.0, 15.0, 16.0, 1.0], [0.105568714, 0.0, 0.0, 0.0], [3227117310.0, 17.0, 18.0, 1.0], [1726434940.0, 19.0, 20.0, 1.0], [211.0, 21.0, 22.0, 0.0], [-0.0125402128, 0.0, 0.0, 0.0], [0.324278474, 0.0, 0.0, 0.0], [-0.245884493, 0.0, 0.0, 0.0], [0.0392293595, 0.0, 0.0, 0.0], [-0.483493418, 0.0, 0.0, 0.0], [-0.0908751711, 0.0, 0.0, 0.0], [0.0469102152, 0.0, 0.0, 0.0], [-0.25686115, 0.0, 0.0, 0.0], [-0.166217014, 0.0, 0.0, 0.0], [0.124557354, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_27(xs):
    #Predicts Class 1
    function_dict = np.array([[2805971460.0, 1.0, 2.0, 2.0], [2661524990.0, 3.0, 4.0, 2.0], [2948765440.0, 5.0, 6.0, 2.0], [228640.0, 7.0, 8.0, 15.0], [170326.0, 9.0, 10.0, 15.0], [-0.348832935, 0.0, 0.0, 0.0], [0.579999983, 11.0, 12.0, 5.0], [0.947000027, 13.0, 14.0, 6.0], [2863348740.0, 15.0, 16.0, 1.0], [-0.105568707, 0.0, 0.0, 0.0], [3227117310.0, 17.0, 18.0, 1.0], [1726434940.0, 19.0, 20.0, 1.0], [211.0, 21.0, 22.0, 0.0], [0.0125402017, 0.0, 0.0, 0.0], [-0.324278474, 0.0, 0.0, 0.0], [0.245884478, 0.0, 0.0, 0.0], [-0.039229311, 0.0, 0.0, 0.0], [0.483493447, 0.0, 0.0, 0.0], [0.0908751413, 0.0, 0.0, 0.0], [-0.0469102524, 0.0, 0.0, 0.0], [0.25686115, 0.0, 0.0, 0.0], [0.166216999, 0.0, 0.0, 0.0], [-0.124557368, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_28(xs):
    #Predicts Class 0
    function_dict = np.array([[1.5, 1.0, 2.0, 7.0], [3886659070.0, 3.0, 4.0, 4.0], [0.207500011, 5.0, 6.0, 10.0], [0.555000007, 7.0, 8.0, 5.0], [-0.15116182, 0.0, 0.0, 0.0], [0.712499976, 9.0, 10.0, 13.0], [0.328500003, 11.0, 12.0, 13.0], [0.298500001, 13.0, 14.0, 13.0], [0.0433499999, 15.0, 16.0, 10.0], [0.0615499988, 17.0, 18.0, 10.0], [0.744499981, 19.0, 20.0, 5.0], [0.600000024, 21.0, 22.0, 6.0], [1646886140.0, 23.0, 24.0, 1.0], [0.0900261477, 0.0, 0.0, 0.0], [-0.174022675, 0.0, 0.0, 0.0], [-0.0026508139, 0.0, 0.0, 0.0], [0.287993699, 0.0, 0.0, 0.0], [0.0524246395, 0.0, 0.0, 0.0], [-0.0914453343, 0.0, 0.0, 0.0], [-0.326086909, 0.0, 0.0, 0.0], [0.0213271659, 0.0, 0.0, 0.0], [0.0187761299, 0.0, 0.0, 0.0], [-0.332401097, 0.0, 0.0, 0.0], [0.0427545682, 0.0, 0.0, 0.0], [0.277424455, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_29(xs):
    #Predicts Class 1
    function_dict = np.array([[1.5, 1.0, 2.0, 7.0], [3886659070.0, 3.0, 4.0, 4.0], [0.207500011, 5.0, 6.0, 10.0], [0.555000007, 7.0, 8.0, 5.0], [0.151161805, 0.0, 0.0, 0.0], [0.712499976, 9.0, 10.0, 13.0], [0.328500003, 11.0, 12.0, 13.0], [0.298500001, 13.0, 14.0, 13.0], [0.0433499999, 15.0, 16.0, 10.0], [0.0615499988, 17.0, 18.0, 10.0], [0.744499981, 19.0, 20.0, 5.0], [0.600000024, 21.0, 22.0, 6.0], [1646886140.0, 23.0, 24.0, 1.0], [-0.0900261477, 0.0, 0.0, 0.0], [0.174022704, 0.0, 0.0, 0.0], [0.00265079504, 0.0, 0.0, 0.0], [-0.287993699, 0.0, 0.0, 0.0], [-0.0524246544, 0.0, 0.0, 0.0], [0.0914453268, 0.0, 0.0, 0.0], [0.326086909, 0.0, 0.0, 0.0], [-0.0213271547, 0.0, 0.0, 0.0], [-0.0187761355, 0.0, 0.0, 0.0], [0.332401037, 0.0, 0.0, 0.0], [-0.0427545644, 0.0, 0.0, 0.0], [-0.277424455, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_30(xs):
    #Predicts Class 0
    function_dict = np.array([[0.851000011, 1.0, 2.0, 11.0], [0.212500006, 3.0, 4.0, 13.0], [-0.240161404, 0.0, 0.0, 0.0], [0.0912500024, 5.0, 6.0, 10.0], [0.0927000046, 7.0, 8.0, 10.0], [0.421999991, 9.0, 10.0, 6.0], [-0.0905496478, 0.0, 0.0, 0.0], [0.0605999976, 11.0, 12.0, 10.0], [0.3935, 13.0, 14.0, 13.0], [-0.0277266391, 0.0, 0.0, 0.0], [0.287085384, 0.0, 0.0, 0.0], [-0.000153261688, 0.0, 0.0, 0.0], [-0.176396593, 0.0, 0.0, 0.0], [-0.0975863934, 0.0, 0.0, 0.0], [0.102596678, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_31(xs):
    #Predicts Class 1
    function_dict = np.array([[0.851000011, 1.0, 2.0, 11.0], [0.212500006, 3.0, 4.0, 13.0], [0.240161404, 0.0, 0.0, 0.0], [0.0912500024, 5.0, 6.0, 10.0], [0.0927000046, 7.0, 8.0, 10.0], [0.421999991, 9.0, 10.0, 6.0], [0.0905496329, 0.0, 0.0, 0.0], [0.0605999976, 11.0, 12.0, 10.0], [0.3935, 13.0, 14.0, 13.0], [0.0277265813, 0.0, 0.0, 0.0], [-0.287085384, 0.0, 0.0, 0.0], [0.000153284345, 0.0, 0.0, 0.0], [0.176396593, 0.0, 0.0, 0.0], [0.0975864157, 0.0, 0.0, 0.0], [-0.102596685, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_32(xs):
    #Predicts Class 0
    function_dict = np.array([[10.5, 1.0, 2.0, 7.0], [0.49849999, 3.0, 4.0, 6.0], [1127.0, 5.0, 6.0, 0.0], [867.0, 7.0, 8.0, 0.0], [0.119000003, 9.0, 10.0, 12.0], [0.148999989, 11.0, 12.0, 12.0], [0.13388443, 0.0, 0.0, 0.0], [0.742500007, 13.0, 14.0, 5.0], [0.214499995, 15.0, 16.0, 12.0], [1954616580.0, 17.0, 18.0, 3.0], [751119616.0, 19.0, 20.0, 3.0], [539.0, 21.0, 22.0, 0.0], [0.0143723832, 0.0, 0.0, 0.0], [-0.254316568, 0.0, 0.0, 0.0], [-0.0615881234, 0.0, 0.0, 0.0], [0.17970857, 0.0, 0.0, 0.0], [-0.122507535, 0.0, 0.0, 0.0], [0.00104158337, 0.0, 0.0, 0.0], [0.19642964, 0.0, 0.0, 0.0], [0.150639668, 0.0, 0.0, 0.0], [-0.0442519151, 0.0, 0.0, 0.0], [-0.0632060394, 0.0, 0.0, 0.0], [-0.32959646, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_33(xs):
    #Predicts Class 1
    function_dict = np.array([[10.5, 1.0, 2.0, 7.0], [0.49849999, 3.0, 4.0, 6.0], [1127.0, 5.0, 6.0, 0.0], [867.0, 7.0, 8.0, 0.0], [0.119000003, 9.0, 10.0, 12.0], [0.148999989, 11.0, 12.0, 12.0], [-0.13388443, 0.0, 0.0, 0.0], [0.742500007, 13.0, 14.0, 5.0], [0.214499995, 15.0, 16.0, 12.0], [1954616580.0, 17.0, 18.0, 3.0], [751119616.0, 19.0, 20.0, 3.0], [539.0, 21.0, 22.0, 0.0], [-0.0143723814, 0.0, 0.0, 0.0], [0.254316539, 0.0, 0.0, 0.0], [0.0615881383, 0.0, 0.0, 0.0], [-0.179708511, 0.0, 0.0, 0.0], [0.122507602, 0.0, 0.0, 0.0], [-0.001041585, 0.0, 0.0, 0.0], [-0.196429655, 0.0, 0.0, 0.0], [-0.150639653, 0.0, 0.0, 0.0], [0.0442519113, 0.0, 0.0, 0.0], [0.0632060394, 0.0, 0.0, 0.0], [0.32959646, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_34(xs):
    #Predicts Class 0
    function_dict = np.array([[0.579999983, 1.0, 2.0, 5.0], [0.0407500006, 3.0, 4.0, 10.0], [-4.43300009, 5.0, 6.0, 8.0], [3.5, 7.0, 8.0, 7.0], [0.903499961, 9.0, 10.0, 6.0], [4087883260.0, 11.0, 12.0, 3.0], [107.135002, 13.0, 14.0, 14.0], [0.00299493968, 0.0, 0.0, 0.0], [-0.254587322, 0.0, 0.0, 0.0], [0.430999994, 15.0, 16.0, 13.0], [0.0857000053, 17.0, 18.0, 12.0], [140.196991, 19.0, 20.0, 14.0], [0.247604117, 0.0, 0.0, 0.0], [-0.0501316264, 0.0, 0.0, 0.0], [0.776499987, 21.0, 22.0, 13.0], [0.186267108, 0.0, 0.0, 0.0], [-0.0815202072, 0.0, 0.0, 0.0], [0.114114381, 0.0, 0.0, 0.0], [-0.22755082, 0.0, 0.0, 0.0], [0.0161894634, 0.0, 0.0, 0.0], [-0.149570569, 0.0, 0.0, 0.0], [0.287828743, 0.0, 0.0, 0.0], [0.0813216642, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 12, 13, 21, 22])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.579999983, 1.0, 2.0, 5.0], [0.0407500006, 3.0, 4.0, 10.0], [-4.43300009, 5.0, 6.0, 8.0], [3.5, 7.0, 8.0, 7.0], [0.903499961, 9.0, 10.0, 6.0], [4087883260.0, 11.0, 12.0, 3.0], [107.135002, 13.0, 14.0, 14.0], [-0.00299479626, 0.0, 0.0, 0.0], [0.254587322, 0.0, 0.0, 0.0], [0.430999994, 15.0, 16.0, 13.0], [0.0857000053, 17.0, 18.0, 12.0], [140.196991, 19.0, 20.0, 14.0], [-0.247604087, 0.0, 0.0, 0.0], [0.050131686, 0.0, 0.0, 0.0], [0.776499987, 21.0, 22.0, 13.0], [-0.186267108, 0.0, 0.0, 0.0], [0.0815201774, 0.0, 0.0, 0.0], [-0.114114404, 0.0, 0.0, 0.0], [0.22755079, 0.0, 0.0, 0.0], [-0.0161894765, 0.0, 0.0, 0.0], [0.149570599, 0.0, 0.0, 0.0], [-0.287828773, 0.0, 0.0, 0.0], [-0.0813216418, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 15, 16, 17, 18, 19, 20, 12, 13, 21, 22])
    branch_indices = np.array([0, 1, 3, 4, 9, 10, 2, 5, 11, 6, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[118.983002, 1.0, 2.0, 14.0], [2987724540.0, 3.0, 4.0, 4.0], [4.19000025e-06, 5.0, 6.0, 11.0], [176826.0, 7.0, 8.0, 15.0], [3.20999993e-06, 9.0, 10.0, 11.0], [0.731000006, 11.0, 12.0, 13.0], [225623.0, 13.0, 14.0, 15.0], [0.5, 15.0, 16.0, 9.0], [1059.0, 17.0, 18.0, 0.0], [-5.56850004, 19.0, 20.0, 8.0], [0.435000002, 21.0, 22.0, 13.0], [1885531900.0, 23.0, 24.0, 2.0], [-5.32700014, 25.0, 26.0, 8.0], [3224365060.0, 27.0, 28.0, 4.0], [0.812999964, 29.0, 30.0, 6.0], [0.18842195, 0.0, 0.0, 0.0], [-0.115796849, 0.0, 0.0, 0.0], [-0.256833285, 0.0, 0.0, 0.0], [0.034028396, 0.0, 0.0, 0.0], [0.269094586, 0.0, 0.0, 0.0], [0.0688519403, 0.0, 0.0, 0.0], [0.0788076818, 0.0, 0.0, 0.0], [-0.173378065, 0.0, 0.0, 0.0], [-0.185785174, 0.0, 0.0, 0.0], [-0.0119427275, 0.0, 0.0, 0.0], [-0.0384053625, 0.0, 0.0, 0.0], [0.201598749, 0.0, 0.0, 0.0], [0.269397914, 0.0, 0.0, 0.0], [-0.0408426896, 0.0, 0.0, 0.0], [0.119307071, 0.0, 0.0, 0.0], [-0.182160676, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_37(xs):
    #Predicts Class 1
    function_dict = np.array([[118.983002, 1.0, 2.0, 14.0], [2987724540.0, 3.0, 4.0, 4.0], [4.19000025e-06, 5.0, 6.0, 11.0], [176826.0, 7.0, 8.0, 15.0], [3.20999993e-06, 9.0, 10.0, 11.0], [0.731000006, 11.0, 12.0, 13.0], [225623.0, 13.0, 14.0, 15.0], [0.5, 15.0, 16.0, 9.0], [1059.0, 17.0, 18.0, 0.0], [-5.56850004, 19.0, 20.0, 8.0], [0.435000002, 21.0, 22.0, 13.0], [1885531900.0, 23.0, 24.0, 2.0], [-5.32700014, 25.0, 26.0, 8.0], [3224365060.0, 27.0, 28.0, 4.0], [0.812999964, 29.0, 30.0, 6.0], [-0.188421965, 0.0, 0.0, 0.0], [0.115796849, 0.0, 0.0, 0.0], [0.256833315, 0.0, 0.0, 0.0], [-0.0340283997, 0.0, 0.0, 0.0], [-0.269094557, 0.0, 0.0, 0.0], [-0.0688518733, 0.0, 0.0, 0.0], [-0.0788075551, 0.0, 0.0, 0.0], [0.173378035, 0.0, 0.0, 0.0], [0.185785145, 0.0, 0.0, 0.0], [0.0119427573, 0.0, 0.0, 0.0], [0.0384053811, 0.0, 0.0, 0.0], [-0.201598749, 0.0, 0.0, 0.0], [-0.269397855, 0.0, 0.0, 0.0], [0.0408427082, 0.0, 0.0, 0.0], [-0.119307145, 0.0, 0.0, 0.0], [0.182160735, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_38(xs):
    #Predicts Class 0
    function_dict = np.array([[0.879999995, 1.0, 2.0, 5.0], [4030266880.0, 3.0, 4.0, 1.0], [0.199748412, 0.0, 0.0, 0.0], [3935276290.0, 5.0, 6.0, 1.0], [194756.5, 7.0, 8.0, 15.0], [151748.0, 9.0, 10.0, 15.0], [-0.233200699, 0.0, 0.0, 0.0], [0.212735564, 0.0, 0.0, 0.0], [0.00257368339, 0.0, 0.0, 0.0], [-0.134348974, 0.0, 0.0, 0.0], [0.0159502383, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_39(xs):
    #Predicts Class 1
    function_dict = np.array([[0.879999995, 1.0, 2.0, 5.0], [4030266880.0, 3.0, 4.0, 1.0], [-0.199748471, 0.0, 0.0, 0.0], [3935276290.0, 5.0, 6.0, 1.0], [194756.5, 7.0, 8.0, 15.0], [151748.0, 9.0, 10.0, 15.0], [0.233200684, 0.0, 0.0, 0.0], [-0.212735578, 0.0, 0.0, 0.0], [-0.0025736331, 0.0, 0.0, 0.0], [0.134348959, 0.0, 0.0, 0.0], [-0.0159502514, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_40(xs):
    #Predicts Class 0
    function_dict = np.array([[-13.1929998, 1.0, 2.0, 8.0], [-0.186835364, 0.0, 0.0, 0.0], [0.212500006, 3.0, 4.0, 13.0], [253755.5, 5.0, 6.0, 15.0], [723268864.0, 7.0, 8.0, 1.0], [0.178822488, 0.0, 0.0, 0.0], [0.0324381813, 0.0, 0.0, 0.0], [0.0578999966, 9.0, 10.0, 10.0], [980076160.0, 11.0, 12.0, 1.0], [-0.0326227918, 0.0, 0.0, 0.0], [0.188249305, 0.0, 0.0, 0.0], [-0.175142273, 0.0, 0.0, 0.0], [0.00307685137, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_41(xs):
    #Predicts Class 1
    function_dict = np.array([[-13.1929998, 1.0, 2.0, 8.0], [0.186835393, 0.0, 0.0, 0.0], [0.212500006, 3.0, 4.0, 13.0], [253755.5, 5.0, 6.0, 15.0], [723268864.0, 7.0, 8.0, 1.0], [-0.178822502, 0.0, 0.0, 0.0], [-0.0324381627, 0.0, 0.0, 0.0], [0.0578999966, 9.0, 10.0, 10.0], [980076160.0, 11.0, 12.0, 1.0], [0.0326227993, 0.0, 0.0, 0.0], [-0.188249305, 0.0, 0.0, 0.0], [0.175142258, 0.0, 0.0, 0.0], [-0.00307683111, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_42(xs):
    #Predicts Class 0
    function_dict = np.array([[0.555500031, 1.0, 2.0, 5.0], [168495.5, 3.0, 4.0, 15.0], [0.930999994, 5.0, 6.0, 6.0], [0.12192665, 0.0, 0.0, 0.0], [0.1215, 7.0, 8.0, 10.0], [129.977509, 9.0, 10.0, 14.0], [127.2155, 11.0, 12.0, 14.0], [208606.5, 13.0, 14.0, 15.0], [2672437250.0, 15.0, 16.0, 2.0], [118.810501, 17.0, 18.0, 14.0], [1117.5, 19.0, 20.0, 0.0], [0.0471248999, 0.0, 0.0, 0.0], [0.24209553, 0.0, 0.0, 0.0], [-0.251958072, 0.0, 0.0, 0.0], [-0.0623747706, 0.0, 0.0, 0.0], [0.108613558, 0.0, 0.0, 0.0], [-0.0801537707, 0.0, 0.0, 0.0], [0.00181937276, 0.0, 0.0, 0.0], [0.169911817, 0.0, 0.0, 0.0], [-0.10979645, 0.0, 0.0, 0.0], [0.174145162, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_43(xs):
    #Predicts Class 1
    function_dict = np.array([[0.555500031, 1.0, 2.0, 5.0], [168495.5, 3.0, 4.0, 15.0], [0.930999994, 5.0, 6.0, 6.0], [-0.12192668, 0.0, 0.0, 0.0], [0.1215, 7.0, 8.0, 10.0], [129.977509, 9.0, 10.0, 14.0], [127.2155, 11.0, 12.0, 14.0], [208606.5, 13.0, 14.0, 15.0], [2672437250.0, 15.0, 16.0, 2.0], [118.810501, 17.0, 18.0, 14.0], [1117.5, 19.0, 20.0, 0.0], [-0.0471249446, 0.0, 0.0, 0.0], [-0.242095515, 0.0, 0.0, 0.0], [0.251958102, 0.0, 0.0, 0.0], [0.0623747744, 0.0, 0.0, 0.0], [-0.108613588, 0.0, 0.0, 0.0], [0.0801537409, 0.0, 0.0, 0.0], [-0.00181934598, 0.0, 0.0, 0.0], [-0.169911817, 0.0, 0.0, 0.0], [0.109796435, 0.0, 0.0, 0.0], [-0.174145147, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_44(xs):
    #Predicts Class 0
    function_dict = np.array([[0.0386499986, 1.0, 2.0, 10.0], [-4.96199989, 3.0, 4.0, 8.0], [419.5, 5.0, 6.0, 0.0], [0.5, 7.0, 8.0, 9.0], [2459746560.0, 9.0, 10.0, 1.0], [0.823500037, 11.0, 12.0, 5.0], [540.0, 13.0, 14.0, 0.0], [-0.00854570139, 0.0, 0.0, 0.0], [-0.231382668, 0.0, 0.0, 0.0], [0.156834573, 0.0, 0.0, 0.0], [-0.0609404258, 0.0, 0.0, 0.0], [0.629999995, 15.0, 16.0, 6.0], [0.187795818, 0.0, 0.0, 0.0], [0.81400001, 17.0, 18.0, 6.0], [2352033280.0, 19.0, 20.0, 4.0], [-0.232967153, 0.0, 0.0, 0.0], [-0.00508505851, 0.0, 0.0, 0.0], [0.246093944, 0.0, 0.0, 0.0], [0.0711940825, 0.0, 0.0, 0.0], [0.079040572, 0.0, 0.0, 0.0], [-0.064914003, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 15, 16, 12, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.0386499986, 1.0, 2.0, 10.0], [-4.96199989, 3.0, 4.0, 8.0], [419.5, 5.0, 6.0, 0.0], [0.5, 7.0, 8.0, 9.0], [2459746560.0, 9.0, 10.0, 1.0], [0.823500037, 11.0, 12.0, 5.0], [540.0, 13.0, 14.0, 0.0], [0.00854567718, 0.0, 0.0, 0.0], [0.231382638, 0.0, 0.0, 0.0], [-0.156834573, 0.0, 0.0, 0.0], [0.0609404221, 0.0, 0.0, 0.0], [0.629999995, 15.0, 16.0, 6.0], [-0.187795818, 0.0, 0.0, 0.0], [0.81400001, 17.0, 18.0, 6.0], [2352033280.0, 19.0, 20.0, 4.0], [0.232967153, 0.0, 0.0, 0.0], [0.00508509204, 0.0, 0.0, 0.0], [-0.246094003, 0.0, 0.0, 0.0], [-0.0711941421, 0.0, 0.0, 0.0], [-0.0790405869, 0.0, 0.0, 0.0], [0.0649140328, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 15, 16, 12, 17, 18, 19, 20])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 11, 6, 13, 14])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.863499999, 1.0, 2.0, 13.0], [0.579999983, 3.0, 4.0, 5.0], [-0.164410308, 0.0, 0.0, 0.0], [3626939140.0, 5.0, 6.0, 2.0], [-4.43300009, 7.0, 8.0, 8.0], [0.141000003, 9.0, 10.0, 12.0], [-0.16693528, 0.0, 0.0, 0.0], [222032.5, 11.0, 12.0, 15.0], [114.725998, 13.0, 14.0, 14.0], [0.0792001635, 0.0, 0.0, 0.0], [-0.102903634, 0.0, 0.0, 0.0], [-0.0149503713, 0.0, 0.0, 0.0], [0.115995444, 0.0, 0.0, 0.0], [0.00799841713, 0.0, 0.0, 0.0], [0.238402516, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_47(xs):
    #Predicts Class 1
    function_dict = np.array([[0.863499999, 1.0, 2.0, 13.0], [0.579999983, 3.0, 4.0, 5.0], [0.164410368, 0.0, 0.0, 0.0], [3626939140.0, 5.0, 6.0, 2.0], [-4.43300009, 7.0, 8.0, 8.0], [0.141000003, 9.0, 10.0, 12.0], [0.166935265, 0.0, 0.0, 0.0], [222032.5, 11.0, 12.0, 15.0], [114.725998, 13.0, 14.0, 14.0], [-0.0792001262, 0.0, 0.0, 0.0], [0.102903649, 0.0, 0.0, 0.0], [0.0149503332, 0.0, 0.0, 0.0], [-0.115995482, 0.0, 0.0, 0.0], [-0.00799841993, 0.0, 0.0, 0.0], [-0.238402501, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_48(xs):
    #Predicts Class 0
    function_dict = np.array([[0.79550004, 1.0, 2.0, 6.0], [-5.48500013, 3.0, 4.0, 8.0], [0.682999969, 5.0, 6.0, 13.0], [0.164000005, 7.0, 8.0, 12.0], [239.0, 9.0, 10.0, 0.0], [136.819489, 11.0, 12.0, 14.0], [802.0, 13.0, 14.0, 0.0], [3.43000011e-06, 15.0, 16.0, 11.0], [1060568320.0, 17.0, 18.0, 1.0], [-0.039967481, 0.0, 0.0, 0.0], [0.0302000009, 19.0, 20.0, 10.0], [171966.0, 21.0, 22.0, 15.0], [156.105499, 23.0, 24.0, 14.0], [-0.203285962, 0.0, 0.0, 0.0], [-0.0453723185, 0.0, 0.0, 0.0], [0.0105896965, 0.0, 0.0, 0.0], [-0.16552496, 0.0, 0.0, 0.0], [-0.072660014, 0.0, 0.0, 0.0], [0.121237941, 0.0, 0.0, 0.0], [-0.0267468188, 0.0, 0.0, 0.0], [0.246843234, 0.0, 0.0, 0.0], [0.0977374613, 0.0, 0.0, 0.0], [-0.142892286, 0.0, 0.0, 0.0], [0.233123004, 0.0, 0.0, 0.0], [-0.158904523, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_49(xs):
    #Predicts Class 1
    function_dict = np.array([[0.79550004, 1.0, 2.0, 6.0], [-5.48500013, 3.0, 4.0, 8.0], [0.682999969, 5.0, 6.0, 13.0], [0.164000005, 7.0, 8.0, 12.0], [239.0, 9.0, 10.0, 0.0], [136.819489, 11.0, 12.0, 14.0], [802.0, 13.0, 14.0, 0.0], [3.43000011e-06, 15.0, 16.0, 11.0], [1060568320.0, 17.0, 18.0, 1.0], [0.0399675146, 0.0, 0.0, 0.0], [0.0302000009, 19.0, 20.0, 10.0], [171966.0, 21.0, 22.0, 15.0], [156.105499, 23.0, 24.0, 14.0], [0.203285903, 0.0, 0.0, 0.0], [0.0453722291, 0.0, 0.0, 0.0], [-0.010589716, 0.0, 0.0, 0.0], [0.165524915, 0.0, 0.0, 0.0], [0.0726600066, 0.0, 0.0, 0.0], [-0.121237926, 0.0, 0.0, 0.0], [0.0267468095, 0.0, 0.0, 0.0], [-0.246843219, 0.0, 0.0, 0.0], [-0.0977374166, 0.0, 0.0, 0.0], [0.142892256, 0.0, 0.0, 0.0], [-0.233123004, 0.0, 0.0, 0.0], [0.158904582, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_50(xs):
    #Predicts Class 0
    function_dict = np.array([[2761012740.0, 1.0, 2.0, 2.0], [188.5, 3.0, 4.0, 0.0], [203899.5, 5.0, 6.0, 15.0], [0.142000005, 7.0, 8.0, 12.0], [0.127499998, 9.0, 10.0, 12.0], [752.0, 11.0, 12.0, 0.0], [0.0931499973, 13.0, 14.0, 12.0], [-0.00754550146, 0.0, 0.0, 0.0], [0.167699516, 0.0, 0.0, 0.0], [224524.0, 15.0, 16.0, 15.0], [0.00291999988, 17.0, 18.0, 11.0], [0.5, 19.0, 20.0, 9.0], [3579914240.0, 21.0, 22.0, 3.0], [-0.0630500019, 0.0, 0.0, 0.0], [0.811499953, 23.0, 24.0, 6.0], [0.0794185922, 0.0, 0.0, 0.0], [-0.140262514, 0.0, 0.0, 0.0], [-0.142392054, 0.0, 0.0, 0.0], [0.0710145906, 0.0, 0.0, 0.0], [0.0356371738, 0.0, 0.0, 0.0], [-0.151111454, 0.0, 0.0, 0.0], [0.154651165, 0.0, 0.0, 0.0], [-0.0424133651, 0.0, 0.0, 0.0], [0.227823123, 0.0, 0.0, 0.0], [0.0172647368, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_51(xs):
    #Predicts Class 1
    function_dict = np.array([[2761012740.0, 1.0, 2.0, 2.0], [188.5, 3.0, 4.0, 0.0], [203899.5, 5.0, 6.0, 15.0], [0.142000005, 7.0, 8.0, 12.0], [0.127499998, 9.0, 10.0, 12.0], [752.0, 11.0, 12.0, 0.0], [0.0931499973, 13.0, 14.0, 12.0], [0.0075455131, 0.0, 0.0, 0.0], [-0.167699471, 0.0, 0.0, 0.0], [224524.0, 15.0, 16.0, 15.0], [0.00291999988, 17.0, 18.0, 11.0], [0.5, 19.0, 20.0, 9.0], [3579914240.0, 21.0, 22.0, 3.0], [0.063049987, 0.0, 0.0, 0.0], [0.811499953, 23.0, 24.0, 6.0], [-0.0794185996, 0.0, 0.0, 0.0], [0.140262485, 0.0, 0.0, 0.0], [0.142392054, 0.0, 0.0, 0.0], [-0.0710146427, 0.0, 0.0, 0.0], [-0.0356371328, 0.0, 0.0, 0.0], [0.151111454, 0.0, 0.0, 0.0], [-0.154651165, 0.0, 0.0, 0.0], [0.0424133129, 0.0, 0.0, 0.0], [-0.227823123, 0.0, 0.0, 0.0], [-0.0172648262, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_52(xs):
    #Predicts Class 0
    function_dict = np.array([[-10.7709999, 1.0, 2.0, 8.0], [-0.133670315, 0.0, 0.0, 0.0], [0.0388999991, 3.0, 4.0, 10.0], [2379760130.0, 5.0, 6.0, 2.0], [0.0615499988, 7.0, 8.0, 10.0], [0.646999955, 9.0, 10.0, 5.0], [3388638980.0, 11.0, 12.0, 2.0], [0.677999973, 13.0, 14.0, 13.0], [951456512.0, 15.0, 16.0, 1.0], [-0.190120071, 0.0, 0.0, 0.0], [-0.0099456273, 0.0, 0.0, 0.0], [0.12008062, 0.0, 0.0, 0.0], [-0.0639622808, 0.0, 0.0, 0.0], [0.166534364, 0.0, 0.0, 0.0], [-0.0481092222, 0.0, 0.0, 0.0], [0.130897999, 0.0, 0.0, 0.0], [-0.0232396387, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 9, 10, 11, 12, 13, 14, 15, 16])
    branch_indices = np.array([0, 2, 3, 5, 6, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-10.7709999, 1.0, 2.0, 8.0], [0.133670315, 0.0, 0.0, 0.0], [0.0388999991, 3.0, 4.0, 10.0], [2379760130.0, 5.0, 6.0, 2.0], [0.0615499988, 7.0, 8.0, 10.0], [0.646999955, 9.0, 10.0, 5.0], [3388638980.0, 11.0, 12.0, 2.0], [0.677999973, 13.0, 14.0, 13.0], [951456512.0, 15.0, 16.0, 1.0], [0.190120026, 0.0, 0.0, 0.0], [0.00994552113, 0.0, 0.0, 0.0], [-0.120080635, 0.0, 0.0, 0.0], [0.0639622882, 0.0, 0.0, 0.0], [-0.166534334, 0.0, 0.0, 0.0], [0.0481092744, 0.0, 0.0, 0.0], [-0.130897984, 0.0, 0.0, 0.0], [0.0232396089, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 9, 10, 11, 12, 13, 14, 15, 16])
    branch_indices = np.array([0, 2, 3, 5, 6, 4, 7, 8])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.555500031, 1.0, 2.0, 5.0], [0.5, 3.0, 4.0, 9.0], [3249240830.0, 5.0, 6.0, 1.0], [0.759500027, 7.0, 8.0, 6.0], [-6.77150011, 9.0, 10.0, 8.0], [146213.5, 11.0, 12.0, 15.0], [3818502660.0, 13.0, 14.0, 1.0], [-0.0983035713, 0.0, 0.0, 0.0], [0.903499961, 15.0, 16.0, 6.0], [-0.00592282275, 0.0, 0.0, 0.0], [3423850240.0, 17.0, 18.0, 1.0], [0.403999984, 19.0, 20.0, 13.0], [0.732499957, 21.0, 22.0, 5.0], [0.214401528, 0.0, 0.0, 0.0], [1969176060.0, 23.0, 24.0, 2.0], [0.174396724, 0.0, 0.0, 0.0], [-0.0652830079, 0.0, 0.0, 0.0], [-0.190678716, 0.0, 0.0, 0.0], [-0.0493732318, 0.0, 0.0, 0.0], [-0.183720961, 0.0, 0.0, 0.0], [0.00722480193, 0.0, 0.0, 0.0], [-0.0214383062, 0.0, 0.0, 0.0], [0.0848760679, 0.0, 0.0, 0.0], [-0.111432463, 0.0, 0.0, 0.0], [0.12438871, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def booster_55(xs):
    #Predicts Class 1
    function_dict = np.array([[0.555500031, 1.0, 2.0, 5.0], [0.5, 3.0, 4.0, 9.0], [3249240830.0, 5.0, 6.0, 1.0], [0.759500027, 7.0, 8.0, 6.0], [-6.77150011, 9.0, 10.0, 8.0], [146213.5, 11.0, 12.0, 15.0], [3818502660.0, 13.0, 14.0, 1.0], [0.0983035043, 0.0, 0.0, 0.0], [0.903499961, 15.0, 16.0, 6.0], [0.00592264021, 0.0, 0.0, 0.0], [3423850240.0, 17.0, 18.0, 1.0], [0.403999984, 19.0, 20.0, 13.0], [0.732499957, 21.0, 22.0, 5.0], [-0.214401558, 0.0, 0.0, 0.0], [1969176060.0, 23.0, 24.0, 2.0], [-0.174396753, 0.0, 0.0, 0.0], [0.065282993, 0.0, 0.0, 0.0], [0.190678716, 0.0, 0.0, 0.0], [0.0493731983, 0.0, 0.0, 0.0], [0.183720931, 0.0, 0.0, 0.0], [-0.00722477725, 0.0, 0.0, 0.0], [0.0214382969, 0.0, 0.0, 0.0], [-0.0848761201, 0.0, 0.0, 0.0], [0.111432366, 0.0, 0.0, 0.0], [-0.12438871, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
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
def logit_class_0(xs):
    try:
        import multiprocessing
        pool = multiprocessing.Pool()
    except:
        pool = -1
    sum_of_leaf_values = np.zeros(xs.shape[0])
    if pool != -1:
        sum_of_leaf_values = np.sum(list(pool.starmap(apply,[(eval('booster_' + str(booster_index)), xs) for booster_index in range(0,56,2)])), axis=0)
    else:
        for booster_index in range(0,56,2):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def logit_class_1(xs):
    try:
        import multiprocessing
        pool = multiprocessing.Pool()
    except:
        pool = -1
    sum_of_leaf_values = np.zeros(xs.shape[0])
    if pool != -1:
        sum_of_leaf_values = np.sum(list(pool.starmap(apply,[(eval('booster_' + str(booster_index)), xs) for booster_index in range(1,56,2)])), axis=0)
    else:
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
