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
# Total compiler execution time: 0:01:46.74. Finished on: Mar-03-2021 08:57:36.
# This source code requires Python 3.
#
"""
Classifier Type:                    Random Forest
System Type:                         5-way classifier
Best-guess accuracy:                 22.82%
Overall Model accuracy:              100.00% (4177/4177 correct)
Overall Improvement over best guess: 77.18% (of possible 77.18%)
Model capacity (MEC):                6 bits
Generalization ratio:                1610.79 bits/bit
Model efficiency:                    12.86%/parameter
Confusion Matrix:
 [20.61% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 18.48% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 16.83% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 21.26% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 22.82%]
Generalization index:                555.38
Percent of Data Memorized:           0.18%
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
Note: Labels have been remapped to '1.0'=0, '2.0'=1, '3.0'=2, '4.0'=3, '5.0'=4.
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
num_attr = 561
n_classes = 5
transform_true = False

# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target=""
important_idxs=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560]

def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[], trim=False):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target=""
    important_idxs=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560]
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
    clean.mapping={'1.0': 0, '2.0': 1, '3.0': 2, '4.0': 3, '5.0': 4}

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
    function_dict = np.array([[-0.871500015, 1.0, 2.0, 537.0], [0.580500007, 3.0, 4.0, 102.0], [-0.304499984, 5.0, 6.0, 69.0], [0.00953000039, 7.0, 8.0, 379.0], [-0.510499954, 9.0, 10.0, 503.0], [-0.0596999973, 11.0, 12.0, 117.0], [-0.240500003, 13.0, 14.0, 503.0], [-0.581937373, 0.0, 0.0, 0.0], [0.473382682, 0.0, 0.0, 0.0], [1.80719662, 0.0, 0.0, 0.0], [-0.35498625, 0.0, 0.0, 0.0], [-0.452214777, 0.0, 0.0, 0.0], [0.691841722, 0.0, 0.0, 0.0], [1.80492508, 0.0, 0.0, 0.0], [-0.497078687, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.821500003, 1.0, 2.0, 330.0], [-0.585000038, 3.0, 4.0, 434.0], [-0.75849998, 5.0, 6.0, 450.0], [0.705500007, 7.0, 8.0, 184.0], [-0.359499991, 9.0, 10.0, 69.0], [0.0624499992, 11.0, 12.0, 502.0], [-0.644999981, 13.0, 14.0, 352.0], [-0.585311234, 0.0, 0.0, 0.0], [-0.0520036109, 0.0, 0.0, 0.0], [1.80330682, 0.0, 0.0, 0.0], [-0.433459312, 0.0, 0.0, 0.0], [1.86110759, 0.0, 0.0, 0.0], [-0.380396783, 0.0, 0.0, 0.0], [1.66912758, 0.0, 0.0, 0.0], [-0.558121562, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.0617500022, 1.0, 2.0, 201.0], [-0.13350001, 3.0, 4.0, 504.0], [-0.294499993, 5.0, 6.0, 474.0], [0.247000009, 7.0, 8.0, 132.0], [-0.203500003, 9.0, 10.0, 53.0], [-0.466499984, 11.0, 12.0, 89.0], [0.0164999999, 13.0, 14.0, 1.0], [-0.0148390085, 0.0, 0.0, 0.0], [-0.557585299, 0.0, 0.0, 0.0], [-0.282901227, 0.0, 0.0, 0.0], [1.37809575, 0.0, 0.0, 0.0], [-0.0687990412, 0.0, 0.0, 0.0], [2.2224853, 0.0, 0.0, 0.0], [-0.551271558, 0.0, 0.0, 0.0], [0.084087722, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.0538500026, 1.0, 2.0, 41.0], [-0.996500015, 3.0, 4.0, 426.0], [-0.907999992, 5.0, 6.0, 126.0], [-0.609499991, 7.0, 8.0, 539.0], [-0.892500043, 9.0, 10.0, 429.0], [-0.799999952, 11.0, 12.0, 558.0], [-0.81400001, 13.0, 14.0, 44.0], [-0.254495561, 0.0, 0.0, 0.0], [1.80442345, 0.0, 0.0, 0.0], [-0.211511418, 0.0, 0.0, 0.0], [-0.597042143, 0.0, 0.0, 0.0], [1.63957977, 0.0, 0.0, 0.0], [2.37415051, 0.0, 0.0, 0.0], [-0.53296113, 0.0, 0.0, 0.0], [0.919205427, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.972499967, 1.0, 2.0, 138.0], [-0.111500002, 3.0, 4.0, 50.0], [0.601999998, 5.0, 6.0, 12.0], [-0.996500015, 7.0, 8.0, 426.0], [-0.976500034, 9.0, 10.0, 432.0], [-0.598326981, 0.0, 0.0, 0.0], [0.168500006, 11.0, 12.0, 559.0], [0.560220718, 0.0, 0.0, 0.0], [2.20543504, 0.0, 0.0, 0.0], [-0.373636305, 0.0, 0.0, 0.0], [0.875551343, 0.0, 0.0, 0.0], [-0.368692279, 0.0, 0.0, 0.0], [1.68402696, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 4, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.911499977, 1.0, 2.0, 396.0], [-0.513499975, 3.0, 4.0, 355.0], [-0.966500044, 5.0, 6.0, 448.0], [-0.871500015, 7.0, 8.0, 537.0], [-0.327499986, 9.0, 10.0, 3.0], [-0.422500014, 11.0, 12.0, 268.0], [-0.280499995, 13.0, 14.0, 209.0], [-0.510906875, 0.0, 0.0, 0.0], [-0.680520535, 0.0, 0.0, 0.0], [0.795010507, 0.0, 0.0, 0.0], [-0.0425713547, 0.0, 0.0, 0.0], [0.673920453, 0.0, 0.0, 0.0], [-0.541936278, 0.0, 0.0, 0.0], [-0.235908672, 0.0, 0.0, 0.0], [0.717335343, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[-0.666000009, 1.0, 2.0, 62.0], [-0.418500006, 3.0, 4.0, 434.0], [-0.536499977, 5.0, 6.0, 73.0], [0.306500018, 7.0, 8.0, 456.0], [-0.827499986, 9.0, 10.0, 450.0], [-0.874000013, 11.0, 12.0, 310.0], [-0.290499985, 13.0, 14.0, 53.0], [-0.548377335, 0.0, 0.0, 0.0], [-0.0995496139, 0.0, 0.0, 0.0], [0.624823093, 0.0, 0.0, 0.0], [-0.450004905, 0.0, 0.0, 0.0], [-0.562970996, 0.0, 0.0, 0.0], [0.629103839, 0.0, 0.0, 0.0], [0.457721144, 0.0, 0.0, 0.0], [-0.34363091, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[0.129500002, 1.0, 2.0, 9.0], [-0.333499998, 3.0, 4.0, 37.0], [0.422500014, 5.0, 6.0, 22.0], [-0.818500042, 7.0, 8.0, 91.0], [-0.0333499983, 9.0, 10.0, 432.0], [-0.965499997, 11.0, 12.0, 418.0], [0.0760499984, 13.0, 14.0, 201.0], [-0.469181836, 0.0, 0.0, 0.0], [0.320427477, 0.0, 0.0, 0.0], [-0.515626073, 0.0, 0.0, 0.0], [0.291895539, 0.0, 0.0, 0.0], [-0.262545466, 0.0, 0.0, 0.0], [0.638122022, 0.0, 0.0, 0.0], [-0.516212106, 0.0, 0.0, 0.0], [0.505262673, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.103, 1.0, 2.0, 182.0], [-0.16049999, 3.0, 4.0, 50.0], [0.170000002, 5.0, 6.0, 50.0], [0.26700002, 7.0, 8.0, 42.0], [-0.101500005, 9.0, 10.0, 53.0], [0.601500034, 11.0, 12.0, 23.0], [0.289842218, 0.0, 0.0, 0.0], [-0.321805567, 0.0, 0.0, 0.0], [1.31376815, 0.0, 0.0, 0.0], [1.79500628, 0.0, 0.0, 0.0], [0.475544989, 0.0, 0.0, 0.0], [-0.524036109, 0.0, 0.0, 0.0], [0.0764282495, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 6])
    branch_indices = np.array([0, 1, 3, 4, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.817499995, 1.0, 2.0, 268.0], [0.186000004, 3.0, 4.0, 54.0], [-0.725000024, 5.0, 6.0, 454.0], [-0.888499975, 7.0, 8.0, 129.0], [-0.395999998, 9.0, 10.0, 451.0], [-0.000964276027, 0.0, 0.0, 0.0], [-0.524103522, 0.0, 0.0, 0.0], [-0.630319715, 0.0, 0.0, 0.0], [0.518279672, 0.0, 0.0, 0.0], [0.635058224, 0.0, 0.0, 0.0], [-0.455272913, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.803499997, 1.0, 2.0, 544.0], [0.906499982, 3.0, 4.0, 247.0], [-0.24849999, 5.0, 6.0, 37.0], [0.540499985, 7.0, 8.0, 172.0], [-0.238000005, 9.0, 10.0, 209.0], [-0.4745, 11.0, 12.0, 65.0], [-0.966500044, 13.0, 14.0, 448.0], [0.178219423, 0.0, 0.0, 0.0], [-0.517096162, 0.0, 0.0, 0.0], [-0.478430867, 0.0, 0.0, 0.0], [0.533150256, 0.0, 0.0, 0.0], [-0.466677725, 0.0, 0.0, 0.0], [0.396836668, 0.0, 0.0, 0.0], [-0.189659908, 0.0, 0.0, 0.0], [0.634603679, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.642500043, 1.0, 2.0, 316.0], [-0.659999967, 3.0, 4.0, 371.0], [-0.75849998, 5.0, 6.0, 450.0], [0.318500012, 7.0, 8.0, 102.0], [-0.0703499988, 9.0, 10.0, 457.0], [-0.800500035, 11.0, 12.0, 302.0], [0.0846000016, 13.0, 14.0, 249.0], [-0.547162354, 0.0, 0.0, 0.0], [0.413198709, 0.0, 0.0, 0.0], [-0.513525605, 0.0, 0.0, 0.0], [-0.0518050231, 0.0, 0.0, 0.0], [-0.218660697, 0.0, 0.0, 0.0], [0.513106346, 0.0, 0.0, 0.0], [-0.43742609, 0.0, 0.0, 0.0], [0.302271903, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.286500007, 1.0, 2.0, 37.0], [-0.877499998, 3.0, 4.0, 57.0], [-0.114, 5.0, 6.0, 202.0], [-0.984500051, 7.0, 8.0, 411.0], [0.140499994, 9.0, 10.0, 201.0], [0.408500016, 11.0, 12.0, 0.0], [-0.485000014, 13.0, 14.0, 71.0], [-0.474704713, 0.0, 0.0, 0.0], [0.622084618, 0.0, 0.0, 0.0], [-0.476201445, 0.0, 0.0, 0.0], [0.486741245, 0.0, 0.0, 0.0], [-0.520201743, 0.0, 0.0, 0.0], [0.283736974, 0.0, 0.0, 0.0], [-0.294940084, 0.0, 0.0, 0.0], [0.531687081, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.852499962, 1.0, 2.0, 129.0], [-0.980499983, 3.0, 4.0, 101.0], [-0.955500007, 5.0, 6.0, 139.0], [-0.992499948, 7.0, 8.0, 179.0], [-0.0586500019, 9.0, 10.0, 41.0], [-0.949499965, 11.0, 12.0, 286.0], [-0.966500044, 13.0, 14.0, 79.0], [0.415245831, 0.0, 0.0, 0.0], [-0.305112571, 0.0, 0.0, 0.0], [1.35483348, 0.0, 0.0, 0.0], [0.441443533, 0.0, 0.0, 0.0], [-0.362613976, 0.0, 0.0, 0.0], [0.785150886, 0.0, 0.0, 0.0], [0.59102577, 0.0, 0.0, 0.0], [-0.499093562, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.9375, 1.0, 2.0, 140.0], [-0.09155, 3.0, 4.0, 451.0], [0.00638499996, 5.0, 6.0, 105.0], [0.431500018, 7.0, 8.0, 159.0], [-0.956499994, 9.0, 10.0, 558.0], [-0.924499989, 11.0, 12.0, 423.0], [-0.155499995, 13.0, 14.0, 445.0], [0.470617801, 0.0, 0.0, 0.0], [1.30965173, 0.0, 0.0, 0.0], [2.36322737, 0.0, 0.0, 0.0], [-0.0450524017, 0.0, 0.0, 0.0], [-1.0308677, 0.0, 0.0, 0.0], [-0.439581066, 0.0, 0.0, 0.0], [-0.551119804, 0.0, 0.0, 0.0], [0.473683536, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.87650001, 1.0, 2.0, 396.0], [-0.586500049, 3.0, 4.0, 85.0], [-0.67750001, 5.0, 6.0, 302.0], [-0.840499997, 7.0, 8.0, 403.0], [-0.453500003, 9.0, 10.0, 62.0], [-0.157499999, 11.0, 12.0, 141.0], [-0.620499969, 13.0, 14.0, 450.0], [-0.508895099, 0.0, 0.0, 0.0], [-0.145976067, 0.0, 0.0, 0.0], [0.335977435, 0.0, 0.0, 0.0], [-0.453345776, 0.0, 0.0, 0.0], [0.517209172, 0.0, 0.0, 0.0], [-0.523105085, 0.0, 0.0, 0.0], [-0.508894265, 0.0, 0.0, 0.0], [0.154466256, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[-0.955500007, 1.0, 2.0, 57.0], [-0.0659499988, 3.0, 4.0, 243.0], [-0.804499984, 5.0, 6.0, 302.0], [-0.278500021, 7.0, 8.0, 434.0], [-0.73300004, 9.0, 10.0, 544.0], [-0.901499987, 11.0, 12.0, 460.0], [-0.218499988, 13.0, 14.0, 518.0], [-0.525741756, 0.0, 0.0, 0.0], [0.182152659, 0.0, 0.0, 0.0], [0.527827084, 0.0, 0.0, 0.0], [-0.325525522, 0.0, 0.0, 0.0], [-0.489009976, 0.0, 0.0, 0.0], [0.276040107, 0.0, 0.0, 0.0], [0.413963348, 0.0, 0.0, 0.0], [-0.305202514, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.700000048, 1.0, 2.0, 369.0], [0.0977499932, 3.0, 4.0, 159.0], [-0.0519500002, 5.0, 6.0, 89.0], [-0.124500006, 7.0, 8.0, 298.0], [-0.404500008, 9.0, 10.0, 298.0], [-0.577499986, 11.0, 12.0, 37.0], [-0.300500005, 13.0, 14.0, 503.0], [0.611189425, 0.0, 0.0, 0.0], [-0.100678779, 0.0, 0.0, 0.0], [0.414614916, 0.0, 0.0, 0.0], [-0.403733939, 0.0, 0.0, 0.0], [0.425756156, 0.0, 0.0, 0.0], [-0.437771142, 0.0, 0.0, 0.0], [-0.337593764, 0.0, 0.0, 0.0], [0.525056779, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.997500002, 1.0, 2.0, 387.0], [0.50849998, 3.0, 4.0, 557.0], [-0.941500008, 5.0, 6.0, 91.0], [-0.989500046, 7.0, 8.0, 433.0], [0.0229000002, 9.0, 10.0, 186.0], [0.193358272, 0.0, 0.0, 0.0], [-0.499426335, 0.0, 0.0, 0.0], [-0.0787138417, 0.0, 0.0, 0.0], [0.339187652, 0.0, 0.0, 0.0], [-0.61084348, 0.0, 0.0, 0.0], [0.14482899, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.817499995, 1.0, 2.0, 268.0], [-0.985499978, 3.0, 4.0, 427.0], [0.520500004, 5.0, 6.0, 23.0], [0.318499982, 7.0, 8.0, 185.0], [-0.985499978, 9.0, 10.0, 432.0], [-0.494063735, 0.0, 0.0, 0.0], [-0.134070709, 0.0, 0.0, 0.0], [0.55517292, 0.0, 0.0, 0.0], [-0.445482433, 0.0, 0.0, 0.0], [-0.45665729, 0.0, 0.0, 0.0], [0.212407634, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.596500039, 1.0, 2.0, 252.0], [-0.476500005, 3.0, 4.0, 179.0], [-0.33950001, 5.0, 6.0, 503.0], [-0.912500024, 7.0, 8.0, 332.0], [-0.760499954, 9.0, 10.0, 302.0], [-0.403999984, 11.0, 12.0, 526.0], [-0.888999999, 13.0, 14.0, 524.0], [-0.479906887, 0.0, 0.0, 0.0], [0.0566666014, 0.0, 0.0, 0.0], [0.51043427, 0.0, 0.0, 0.0], [-0.424180627, 0.0, 0.0, 0.0], [-0.180883259, 0.0, 0.0, 0.0], [0.516208708, 0.0, 0.0, 0.0], [-0.497301877, 0.0, 0.0, 0.0], [0.129611537, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.282499999, 1.0, 2.0, 53.0], [0.428499997, 3.0, 4.0, 39.0], [0.695500016, 5.0, 6.0, 76.0], [-0.846000016, 7.0, 8.0, 302.0], [-0.597000003, 9.0, 10.0, 75.0], [-0.99849999, 11.0, 12.0, 58.0], [-0.597000003, 13.0, 14.0, 140.0], [-0.338413596, 0.0, 0.0, 0.0], [0.593831539, 0.0, 0.0, 0.0], [0.264308393, 0.0, 0.0, 0.0], [-0.428753912, 0.0, 0.0, 0.0], [0.170531526, 0.0, 0.0, 0.0], [-0.469959646, 0.0, 0.0, 0.0], [-0.345866919, 0.0, 0.0, 0.0], [0.443422794, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.3495, 1.0, 2.0, 37.0], [-0.185499996, 3.0, 4.0, 458.0], [-0.114500001, 5.0, 6.0, 202.0], [-0.690500021, 7.0, 8.0, 469.0], [0.494000018, 9.0, 10.0, 198.0], [0.400000006, 11.0, 12.0, 296.0], [0.292999983, 13.0, 14.0, 559.0], [0.548065841, 0.0, 0.0, 0.0], [-0.177123383, 0.0, 0.0, 0.0], [-0.353321552, 0.0, 0.0, 0.0], [0.365861088, 0.0, 0.0, 0.0], [-0.435294062, 0.0, 0.0, 0.0], [0.205003306, 0.0, 0.0, 0.0], [0.44524017, 0.0, 0.0, 0.0], [-0.499226242, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.0945499986, 1.0, 2.0, 53.0], [-0.0142000001, 3.0, 4.0, 37.0], [-0.361499995, 5.0, 6.0, 119.0], [-0.993499994, 7.0, 8.0, 426.0], [0.256500006, 9.0, 10.0, 197.0], [-0.107500002, 11.0, 12.0, 454.0], [0.574999988, 13.0, 14.0, 107.0], [0.223084539, 0.0, 0.0, 0.0], [-0.340269089, 0.0, 0.0, 0.0], [-0.00513014756, 0.0, 0.0, 0.0], [0.570183814, 0.0, 0.0, 0.0], [0.291177481, 0.0, 0.0, 0.0], [-0.466040015, 0.0, 0.0, 0.0], [0.52307415, 0.0, 0.0, 0.0], [-0.0884962976, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.0684999973, 1.0, 2.0, 105.0], [0.0406499989, 3.0, 4.0, 194.0], [0.0945499986, 5.0, 6.0, 53.0], [0.0953999981, 7.0, 8.0, 196.0], [0.815500021, 9.0, 10.0, 132.0], [0.193000004, 11.0, 12.0, 451.0], [-0.361499995, 13.0, 14.0, 119.0], [-0.280001342, 0.0, 0.0, 0.0], [0.360801548, 0.0, 0.0, 0.0], [-0.374743789, 0.0, 0.0, 0.0], [-1.03764725, 0.0, 0.0, 0.0], [0.327625692, 0.0, 0.0, 0.0], [-0.203624398, 0.0, 0.0, 0.0], [0.24615939, 0.0, 0.0, 0.0], [-0.469750345, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.581499994, 1.0, 2.0, 175.0], [-0.0751999989, 3.0, 4.0, 526.0], [-0.277500004, 5.0, 6.0, 207.0], [-0.709499955, 7.0, 8.0, 449.0], [-0.478999972, 9.0, 10.0, 503.0], [-0.57949996, 11.0, 12.0, 38.0], [0.30250001, 13.0, 14.0, 159.0], [-0.490507483, 0.0, 0.0, 0.0], [-0.0413148329, 0.0, 0.0, 0.0], [0.277187616, 0.0, 0.0, 0.0], [-0.399251968, 0.0, 0.0, 0.0], [-0.505664408, 0.0, 0.0, 0.0], [0.341647118, 0.0, 0.0, 0.0], [-0.453757614, 0.0, 0.0, 0.0], [0.175198421, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[-0.480499983, 1.0, 2.0, 275.0], [-0.654500008, 3.0, 4.0, 330.0], [-0.903499961, 5.0, 6.0, 419.0], [-0.801499963, 7.0, 8.0, 488.0], [-0.625500023, 9.0, 10.0, 543.0], [-0.96449995, 11.0, 12.0, 62.0], [-0.265999973, 13.0, 14.0, 141.0], [-0.511583388, 0.0, 0.0, 0.0], [0.0199868344, 0.0, 0.0, 0.0], [0.389570057, 0.0, 0.0, 0.0], [-0.27687788, 0.0, 0.0, 0.0], [-0.307099104, 0.0, 0.0, 0.0], [0.348412305, 0.0, 0.0, 0.0], [-0.519601822, 0.0, 0.0, 0.0], [0.236531734, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.805500031, 1.0, 2.0, 302.0], [0.708000004, 3.0, 4.0, 555.0], [-0.636500001, 5.0, 6.0, 434.0], [-0.470103592, 0.0, 0.0, 0.0], [-0.0798485875, 0.0, 0.0, 0.0], [-0.453500003, 7.0, 8.0, 433.0], [0.0648000017, 9.0, 10.0, 201.0], [0.581663132, 0.0, 0.0, 0.0], [-0.0957209989, 0.0, 0.0, 0.0], [-0.252954543, 0.0, 0.0, 0.0], [0.422203213, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.140000001, 1.0, 2.0, 182.0], [0.109499998, 3.0, 4.0, 196.0], [-0.994500041, 5.0, 6.0, 435.0], [-0.916999996, 7.0, 8.0, 558.0], [-0.503499985, 9.0, 10.0, 454.0], [0.0234273467, 0.0, 0.0, 0.0], [-0.477140188, 0.0, 0.0, 0.0], [0.704439044, 0.0, 0.0, 0.0], [0.167313933, 0.0, 0.0, 0.0], [0.31663835, 0.0, 0.0, 0.0], [-0.271313012, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.993499994, 1.0, 2.0, 434.0], [-0.993499994, 3.0, 4.0, 423.0], [-0.710999966, 5.0, 6.0, 64.0], [-0.101549998, 7.0, 8.0, 41.0], [-0.995499969, 9.0, 10.0, 57.0], [-0.164499998, 11.0, 12.0, 54.0], [-0.990499973, 13.0, 14.0, 138.0], [0.498075336, 0.0, 0.0, 0.0], [-0.363672137, 0.0, 0.0, 0.0], [1.02476418, 0.0, 0.0, 0.0], [0.365739197, 0.0, 0.0, 0.0], [0.111727819, 0.0, 0.0, 0.0], [-0.420901299, 0.0, 0.0, 0.0], [0.225873709, 0.0, 0.0, 0.0], [-0.271716535, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.532500029, 1.0, 2.0, 354.0], [-0.925500035, 3.0, 4.0, 474.0], [-0.232499987, 5.0, 6.0, 37.0], [-0.46899581, 0.0, 0.0, 0.0], [-0.42049998, 7.0, 8.0, 504.0], [-0.75849998, 9.0, 10.0, 450.0], [-0.0854500011, 11.0, 12.0, 457.0], [0.208519652, 0.0, 0.0, 0.0], [-0.404271007, 0.0, 0.0, 0.0], [-0.456910789, 0.0, 0.0, 0.0], [0.100245744, 0.0, 0.0, 0.0], [0.375202805, 0.0, 0.0, 0.0], [-0.253977001, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.113000005, 1.0, 2.0, 22.0], [-0.334999979, 3.0, 4.0, 29.0], [0.291500002, 5.0, 6.0, 559.0], [-0.788499951, 7.0, 8.0, 138.0], [-0.225500003, 9.0, 10.0, 275.0], [-0.0854500011, 11.0, 12.0, 457.0], [0.150999993, 13.0, 14.0, 512.0], [-0.271622568, 0.0, 0.0, 0.0], [0.29782629, 0.0, 0.0, 0.0], [-0.534160733, 0.0, 0.0, 0.0], [0.0255534984, 0.0, 0.0, 0.0], [-0.13172242, 0.0, 0.0, 0.0], [0.459848106, 0.0, 0.0, 0.0], [0.507736921, 0.0, 0.0, 0.0], [-0.0259417426, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.324499995, 1.0, 2.0, 504.0], [-0.860000014, 3.0, 4.0, 369.0], [0.678499997, 5.0, 6.0, 184.0], [-0.870499969, 7.0, 8.0, 18.0], [-0.450834274, 0.0, 0.0, 0.0], [-0.306999981, 9.0, 10.0, 433.0], [-0.475499988, 11.0, 12.0, 381.0], [0.274193257, 0.0, 0.0, 0.0], [-0.342628062, 0.0, 0.0, 0.0], [0.375854164, 0.0, 0.0, 0.0], [-0.26104784, 0.0, 0.0, 0.0], [-0.431922376, 0.0, 0.0, 0.0], [0.362876892, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 3, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.869500041, 1.0, 2.0, 129.0], [0.105499998, 3.0, 4.0, 373.0], [-0.917500019, 5.0, 6.0, 558.0], [-0.943500042, 7.0, 8.0, 9.0], [0.689499974, 9.0, 10.0, 13.0], [-0.99150002, 11.0, 12.0, 168.0], [0.223500013, 13.0, 14.0, 451.0], [-0.409792155, 0.0, 0.0, 0.0], [0.368814081, 0.0, 0.0, 0.0], [-0.539260983, 0.0, 0.0, 0.0], [0.0948395804, 0.0, 0.0, 0.0], [-0.439583659, 0.0, 0.0, 0.0], [0.555402696, 0.0, 0.0, 0.0], [-0.317468733, 0.0, 0.0, 0.0], [0.329540163, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[0.103500001, 1.0, 2.0, 122.0], [-0.647500038, 3.0, 4.0, 23.0], [-0.984500051, 5.0, 6.0, 169.0], [0.175000012, 7.0, 8.0, 54.0], [-0.499500006, 9.0, 10.0, 142.0], [-0.923500001, 11.0, 12.0, 57.0], [-0.973500013, 13.0, 14.0, 230.0], [0.550582588, 0.0, 0.0, 0.0], [-0.274697036, 0.0, 0.0, 0.0], [-0.22447817, 0.0, 0.0, 0.0], [0.112358421, 0.0, 0.0, 0.0], [-0.834229767, 0.0, 0.0, 0.0], [0.0111551397, 0.0, 0.0, 0.0], [0.445621967, 0.0, 0.0, 0.0], [-0.314276189, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.67750001, 1.0, 2.0, 179.0], [-0.868499994, 3.0, 4.0, 304.0], [-0.720499992, 5.0, 6.0, 302.0], [-0.448563397, 0.0, 0.0, 0.0], [-0.0827696249, 0.0, 0.0, 0.0], [-0.271499991, 7.0, 8.0, 141.0], [-0.888999999, 9.0, 10.0, 550.0], [0.365491331, 0.0, 0.0, 0.0], [-0.304183155, 0.0, 0.0, 0.0], [-0.445378542, 0.0, 0.0, 0.0], [0.0769177973, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[-0.418500006, 1.0, 2.0, 428.0], [-0.430000007, 3.0, 4.0, 433.0], [-0.00707999989, 5.0, 6.0, 117.0], [0.192499995, 7.0, 8.0, 55.0], [-0.627499998, 9.0, 10.0, 403.0], [-0.895500004, 11.0, 12.0, 487.0], [-0.9745, 13.0, 14.0, 477.0], [-0.417312235, 0.0, 0.0, 0.0], [0.142282635, 0.0, 0.0, 0.0], [0.367261142, 0.0, 0.0, 0.0], [-0.261048079, 0.0, 0.0, 0.0], [0.370241016, 0.0, 0.0, 0.0], [-0.215349019, 0.0, 0.0, 0.0], [0.113323249, 0.0, 0.0, 0.0], [-0.435396612, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.663499951, 1.0, 2.0, 171.0], [-0.269500017, 3.0, 4.0, 121.0], [0.351500005, 5.0, 6.0, 22.0], [0.128378496, 0.0, 0.0, 0.0], [-0.425500005, 7.0, 8.0, 126.0], [-0.99150002, 9.0, 10.0, 58.0], [-0.195999995, 11.0, 12.0, 207.0], [-0.443502098, 0.0, 0.0, 0.0], [-0.126515076, 0.0, 0.0, 0.0], [-0.0676156431, 0.0, 0.0, 0.0], [0.42763415, 0.0, 0.0, 0.0], [-0.284770817, 0.0, 0.0, 0.0], [0.286024779, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.5625, 1.0, 2.0, 556.0], [-0.527999997, 3.0, 4.0, 556.0], [-0.997500002, 5.0, 6.0, 278.0], [-0.0600999966, 7.0, 8.0, 152.0], [0.151499987, 9.0, 10.0, 113.0], [-0.0324999988, 11.0, 12.0, 35.0], [0.158500001, 13.0, 14.0, 112.0], [0.249362007, 0.0, 0.0, 0.0], [-0.465716183, 0.0, 0.0, 0.0], [0.260256261, 0.0, 0.0, 0.0], [-0.0549764708, 0.0, 0.0, 0.0], [0.658949256, 0.0, 0.0, 0.0], [-0.169449061, 0.0, 0.0, 0.0], [0.0811984316, 0.0, 0.0, 0.0], [-0.606030703, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.984500051, 1.0, 2.0, 352.0], [-0.738000035, 3.0, 4.0, 445.0], [-0.982499957, 5.0, 6.0, 179.0], [-0.142500013, 7.0, 8.0, 53.0], [0.769000053, 9.0, 10.0, 158.0], [-0.312999994, 11.0, 12.0, 75.0], [-0.996500015, 13.0, 14.0, 381.0], [0.39360261, 0.0, 0.0, 0.0], [-0.245860443, 0.0, 0.0, 0.0], [0.471053749, 0.0, 0.0, 0.0], [-0.39387235, 0.0, 0.0, 0.0], [-0.535028875, 0.0, 0.0, 0.0], [0.157213137, 0.0, 0.0, 0.0], [0.116475612, 0.0, 0.0, 0.0], [-0.460329324, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.620499969, 1.0, 2.0, 450.0], [-0.0752499998, 3.0, 4.0, 526.0], [-0.122500002, 5.0, 6.0, 202.0], [-0.709499955, 7.0, 8.0, 449.0], [-0.157999992, 9.0, 10.0, 37.0], [-0.74849999, 11.0, 12.0, 434.0], [-0.314788729, 0.0, 0.0, 0.0], [-0.391686261, 0.0, 0.0, 0.0], [-0.0460254997, 0.0, 0.0, 0.0], [-0.147572026, 0.0, 0.0, 0.0], [0.313021749, 0.0, 0.0, 0.0], [-0.113130897, 0.0, 0.0, 0.0], [0.401765913, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 6])
    branch_indices = np.array([0, 1, 3, 4, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.430500001, 1.0, 2.0, 92.0], [-0.372500002, 3.0, 4.0, 233.0], [-0.346000016, 5.0, 6.0, 141.0], [-0.335500002, 7.0, 8.0, 157.0], [-0.0336000025, 9.0, 10.0, 131.0], [0.738999963, 11.0, 12.0, 247.0], [0.201437518, 0.0, 0.0, 0.0], [-0.00161346281, 0.0, 0.0, 0.0], [0.366595834, 0.0, 0.0, 0.0], [-0.311987638, 0.0, 0.0, 0.0], [0.216846392, 0.0, 0.0, 0.0], [-0.404854894, 0.0, 0.0, 0.0], [-0.0362937637, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 6])
    branch_indices = np.array([0, 1, 3, 4, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.435499996, 1.0, 2.0, 89.0], [0.0874999985, 3.0, 4.0, 42.0], [-0.0659500062, 5.0, 6.0, 92.0], [-0.60650003, 7.0, 8.0, 37.0], [0.354499996, 9.0, 10.0, 510.0], [-0.0377499983, 11.0, 12.0, 452.0], [-0.618499994, 13.0, 14.0, 527.0], [0.0223461986, 0.0, 0.0, 0.0], [-0.415022194, 0.0, 0.0, 0.0], [-0.319176108, 0.0, 0.0, 0.0], [0.315750927, 0.0, 0.0, 0.0], [-0.283560842, 0.0, 0.0, 0.0], [0.197933391, 0.0, 0.0, 0.0], [-0.0123768793, 0.0, 0.0, 0.0], [0.411783457, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.844500005, 1.0, 2.0, 40.0], [-0.109999999, 3.0, 4.0, 22.0], [-0.166500002, 5.0, 6.0, 41.0], [0.596258342, 0.0, 0.0, 0.0], [-0.0253915526, 0.0, 0.0, 0.0], [0.190499991, 7.0, 8.0, 198.0], [-0.02905, 9.0, 10.0, 560.0], [-0.469734162, 0.0, 0.0, 0.0], [0.135282546, 0.0, 0.0, 0.0], [-0.135858729, 0.0, 0.0, 0.0], [0.198400766, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[0.0148999998, 1.0, 2.0, 186.0], [0.0167500004, 3.0, 4.0, 196.0], [0.137499988, 5.0, 6.0, 187.0], [-0.855499983, 7.0, 8.0, 558.0], [0.0142000001, 9.0, 10.0, 50.0], [0.0495999977, 11.0, 12.0, 36.0], [0.2685, 13.0, 14.0, 111.0], [-0.370837361, 0.0, 0.0, 0.0], [0.111434445, 0.0, 0.0, 0.0], [0.363758385, 0.0, 0.0, 0.0], [-0.134450614, 0.0, 0.0, 0.0], [0.215610772, 0.0, 0.0, 0.0], [-0.389674753, 0.0, 0.0, 0.0], [-0.15409337, 0.0, 0.0, 0.0], [-0.609832466, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.632500052, 1.0, 2.0, 166.0], [-0.620499969, 3.0, 4.0, 450.0], [-0.612499952, 5.0, 6.0, 302.0], [-0.404130489, 0.0, 0.0, 0.0], [0.0148582077, 0.0, 0.0, 0.0], [-0.210000008, 7.0, 8.0, 295.0], [0.0596000031, 9.0, 10.0, 198.0], [-0.0727155358, 0.0, 0.0, 0.0], [0.302723557, 0.0, 0.0, 0.0], [-0.369822979, 0.0, 0.0, 0.0], [0.0793716386, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[0.538499951, 1.0, 2.0, 66.0], [-0.411500007, 3.0, 4.0, 459.0], [0.157000005, 5.0, 6.0, 22.0], [-0.247500002, 7.0, 8.0, 198.0], [-0.00295533473, 0.0, 0.0, 0.0], [-0.528499961, 9.0, 10.0, 374.0], [-0.32099998, 11.0, 12.0, 298.0], [-0.0695747286, 0.0, 0.0, 0.0], [-0.404820114, 0.0, 0.0, 0.0], [0.102950722, 0.0, 0.0, 0.0], [-0.293745548, 0.0, 0.0, 0.0], [-0.0120111601, 0.0, 0.0, 0.0], [0.293705434, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 3, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.756500006, 1.0, 2.0, 91.0], [0.237499997, 3.0, 4.0, 9.0], [0.645500004, 5.0, 6.0, 102.0], [0.1105, 7.0, 8.0, 262.0], [0.0688775927, 0.0, 0.0, 0.0], [-0.591500044, 9.0, 10.0, 558.0], [-0.348999977, 11.0, 12.0, 249.0], [-0.3846865, 0.0, 0.0, 0.0], [-0.0408651792, 0.0, 0.0, 0.0], [0.339881241, 0.0, 0.0, 0.0], [-0.247353181, 0.0, 0.0, 0.0], [0.255843937, 0.0, 0.0, 0.0], [-0.231591031, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 3, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.982499957, 1.0, 2.0, 139.0], [-0.993499994, 3.0, 4.0, 89.0], [-0.955500007, 5.0, 6.0, 140.0], [0.451999992, 7.0, 8.0, 158.0], [0.369499981, 9.0, 10.0, 107.0], [0.749000013, 11.0, 12.0, 158.0], [-0.907500029, 13.0, 14.0, 423.0], [-0.220356077, 0.0, 0.0, 0.0], [0.401698917, 0.0, 0.0, 0.0], [0.432978123, 0.0, 0.0, 0.0], [-0.0527564958, 0.0, 0.0, 0.0], [-0.347128212, 0.0, 0.0, 0.0], [0.345790356, 0.0, 0.0, 0.0], [0.335938096, 0.0, 0.0, 0.0], [-0.273013443, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[0.490500003, 1.0, 2.0, 159.0], [-0.326499999, 3.0, 4.0, 157.0], [-0.17050001, 5.0, 6.0, 55.0], [0.0185500011, 7.0, 8.0, 261.0], [0.00463499967, 9.0, 10.0, 50.0], [-0.882499993, 11.0, 12.0, 240.0], [-0.309295267, 0.0, 0.0, 0.0], [0.121545054, 0.0, 0.0, 0.0], [-0.357544661, 0.0, 0.0, 0.0], [0.12726824, 0.0, 0.0, 0.0], [-0.382587969, 0.0, 0.0, 0.0], [0.592845976, 0.0, 0.0, 0.0], [-0.0176912174, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 6])
    branch_indices = np.array([0, 1, 3, 4, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.966500044, 1.0, 2.0, 448.0], [-0.0557999983, 3.0, 4.0, 37.0], [0.654999971, 5.0, 6.0, 132.0], [-0.364912242, 0.0, 0.0, 0.0], [-0.0277282707, 0.0, 0.0, 0.0], [-0.591500044, 7.0, 8.0, 65.0], [-0.330150187, 0.0, 0.0, 0.0], [-0.114661239, 0.0, 0.0, 0.0], [0.261516362, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 6])
    branch_indices = np.array([0, 1, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.6875, 1.0, 2.0, 316.0], [-0.300499976, 3.0, 4.0, 42.0], [-0.483500004, 5.0, 6.0, 518.0], [0.104912907, 0.0, 0.0, 0.0], [0.192000002, 7.0, 8.0, 92.0], [-0.486000001, 9.0, 10.0, 274.0], [-0.87650001, 11.0, 12.0, 57.0], [-0.00560679613, 0.0, 0.0, 0.0], [-0.341190308, 0.0, 0.0, 0.0], [0.010070079, 0.0, 0.0, 0.0], [0.375210345, 0.0, 0.0, 0.0], [-0.210796848, 0.0, 0.0, 0.0], [0.130676717, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.359499991, 1.0, 2.0, 89.0], [-0.860000014, 3.0, 4.0, 369.0], [0.415499985, 5.0, 6.0, 22.0], [-0.406000018, 7.0, 8.0, 298.0], [-0.507500052, 9.0, 10.0, 37.0], [-0.316500008, 11.0, 12.0, 41.0], [-0.935500026, 13.0, 14.0, 45.0], [0.292223364, 0.0, 0.0, 0.0], [-0.198026165, 0.0, 0.0, 0.0], [-0.00378956506, 0.0, 0.0, 0.0], [-0.357219636, 0.0, 0.0, 0.0], [-0.131589368, 0.0, 0.0, 0.0], [0.32890901, 0.0, 0.0, 0.0], [-0.313034862, 0.0, 0.0, 0.0], [0.117487088, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.1505, 1.0, 2.0, 36.0], [0.233500004, 3.0, 4.0, 451.0], [-0.929499984, 5.0, 6.0, 429.0], [-0.628499985, 7.0, 8.0, 143.0], [-0.35799998, 9.0, 10.0, 69.0], [-0.986500025, 11.0, 12.0, 444.0], [-0.418500006, 13.0, 14.0, 457.0], [0.255628109, 0.0, 0.0, 0.0], [-0.278262734, 0.0, 0.0, 0.0], [-0.172765076, 0.0, 0.0, 0.0], [0.452114224, 0.0, 0.0, 0.0], [-0.0150118731, 0.0, 0.0, 0.0], [0.359830588, 0.0, 0.0, 0.0], [-0.330944031, 0.0, 0.0, 0.0], [0.15258956, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.899999976, 1.0, 2.0, 291.0], [0.187999994, 3.0, 4.0, 105.0], [0.331499994, 5.0, 6.0, 192.0], [0.136500001, 7.0, 8.0, 195.0], [-0.115500003, 9.0, 10.0, 560.0], [-0.0302500017, 11.0, 12.0, 120.0], [0.886500001, 13.0, 14.0, 49.0], [0.113768801, 0.0, 0.0, 0.0], [-0.195668578, 0.0, 0.0, 0.0], [-0.338899642, 0.0, 0.0, 0.0], [0.316579431, 0.0, 0.0, 0.0], [-0.0477858633, 0.0, 0.0, 0.0], [-0.407329679, 0.0, 0.0, 0.0], [-0.0631357282, 0.0, 0.0, 0.0], [0.377745152, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.851500034, 1.0, 2.0, 317.0], [-0.917500019, 3.0, 4.0, 400.0], [0.0294499993, 5.0, 6.0, 198.0], [-0.340529978, 0.0, 0.0, 0.0], [-0.0746626854, 0.0, 0.0, 0.0], [0.316500008, 7.0, 8.0, 72.0], [-0.465499997, 9.0, 10.0, 505.0], [0.189427137, 0.0, 0.0, 0.0], [-0.212216765, 0.0, 0.0, 0.0], [0.296223074, 0.0, 0.0, 0.0], [-0.120627828, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[0.316500008, 1.0, 2.0, 92.0], [-0.689499974, 3.0, 4.0, 450.0], [-0.417500019, 5.0, 6.0, 113.0], [0.1215, 7.0, 8.0, 295.0], [0.0877500027, 9.0, 10.0, 24.0], [-0.234499991, 11.0, 12.0, 54.0], [0.0491500013, 13.0, 14.0, 458.0], [0.283494502, 0.0, 0.0, 0.0], [-0.0988968834, 0.0, 0.0, 0.0], [0.084646523, 0.0, 0.0, 0.0], [-0.308428377, 0.0, 0.0, 0.0], [0.203111306, 0.0, 0.0, 0.0], [-0.0361252986, 0.0, 0.0, 0.0], [-0.284528136, 0.0, 0.0, 0.0], [0.0444103666, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.0525500029, 1.0, 2.0, 504.0], [-0.167499989, 3.0, 4.0, 269.0], [-0.410000026, 5.0, 6.0, 149.0], [-0.485500008, 7.0, 8.0, 274.0], [-0.583000004, 9.0, 10.0, 381.0], [-0.0826959908, 0.0, 0.0, 0.0], [-0.102499999, 11.0, 12.0, 202.0], [-0.275899738, 0.0, 0.0, 0.0], [0.249453083, 0.0, 0.0, 0.0], [-0.318156719, 0.0, 0.0, 0.0], [-0.0129594905, 0.0, 0.0, 0.0], [0.0558453538, 0.0, 0.0, 0.0], [0.350468516, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 4, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.994500041, 1.0, 2.0, 179.0], [-0.136500001, 3.0, 4.0, 53.0], [0.00832999963, 5.0, 6.0, 374.0], [-0.0479499996, 7.0, 8.0, 159.0], [0.184, 9.0, 10.0, 149.0], [-0.983500004, 11.0, 12.0, 432.0], [-0.0420500003, 13.0, 14.0, 114.0], [-0.395769328, 0.0, 0.0, 0.0], [0.158520535, 0.0, 0.0, 0.0], [0.429184198, 0.0, 0.0, 0.0], [-0.299871325, 0.0, 0.0, 0.0], [0.353595495, 0.0, 0.0, 0.0], [-0.02450414, 0.0, 0.0, 0.0], [0.0616992004, 0.0, 0.0, 0.0], [-0.305985719, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.446500003, 1.0, 2.0, 199.0], [-0.0126, 3.0, 4.0, 50.0], [0.0106499996, 5.0, 6.0, 374.0], [-0.0599999987, 7.0, 8.0, 371.0], [-0.270717263, 0.0, 0.0, 0.0], [-0.148499995, 9.0, 10.0, 118.0], [-0.993499994, 11.0, 12.0, 179.0], [0.406671077, 0.0, 0.0, 0.0], [-0.125589982, 0.0, 0.0, 0.0], [0.0410734899, 0.0, 0.0, 0.0], [-0.307199627, 0.0, 0.0, 0.0], [-0.189670712, 0.0, 0.0, 0.0], [0.221382156, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 3, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.460500002, 1.0, 2.0, 354.0], [-0.347499996, 3.0, 4.0, 157.0], [-0.2755, 5.0, 6.0, 207.0], [0.0135685951, 0.0, 0.0, 0.0], [-0.380500019, 7.0, 8.0, 193.0], [-0.567499995, 9.0, 10.0, 65.0], [-0.186142981, 0.0, 0.0, 0.0], [-0.0670371205, 0.0, 0.0, 0.0], [-0.302993894, 0.0, 0.0, 0.0], [-0.0425365344, 0.0, 0.0, 0.0], [0.286839455, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 10, 6])
    branch_indices = np.array([0, 1, 4, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.331499994, 1.0, 2.0, 9.0], [-0.282022297, 0.0, 0.0, 0.0], [-0.653499961, 3.0, 4.0, 171.0], [0.490500003, 5.0, 6.0, 72.0], [-0.301499993, 7.0, 8.0, 41.0], [0.0523259267, 0.0, 0.0, 0.0], [0.336753309, 0.0, 0.0, 0.0], [0.175054893, 0.0, 0.0, 0.0], [-0.146186963, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 5, 6, 7, 8])
    branch_indices = np.array([0, 2, 3, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[0.032250002, 1.0, 2.0, 159.0], [-0.958500028, 3.0, 4.0, 489.0], [0.208499998, 5.0, 6.0, 132.0], [-0.197217032, 0.0, 0.0, 0.0], [-0.995499969, 7.0, 8.0, 58.0], [0.0666685104, 0.0, 0.0, 0.0], [-0.0752499998, 9.0, 10.0, 504.0], [-0.0434133559, 0.0, 0.0, 0.0], [0.287517369, 0.0, 0.0, 0.0], [-0.307633966, 0.0, 0.0, 0.0], [-0.0576702654, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 5, 9, 10])
    branch_indices = np.array([0, 1, 4, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.00731500005, 1.0, 2.0, 186.0], [-0.202500001, 3.0, 4.0, 149.0], [0.137499988, 5.0, 6.0, 187.0], [0.152500004, 7.0, 8.0, 197.0], [-0.699999988, 9.0, 10.0, 448.0], [0.28549999, 11.0, 12.0, 190.0], [-0.813500047, 13.0, 14.0, 104.0], [-0.104863875, 0.0, 0.0, 0.0], [0.265471369, 0.0, 0.0, 0.0], [-0.311341792, 0.0, 0.0, 0.0], [0.106038168, 0.0, 0.0, 0.0], [0.0765535235, 0.0, 0.0, 0.0], [-0.34189105, 0.0, 0.0, 0.0], [-0.231792539, 0.0, 0.0, 0.0], [0.333872437, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.00774000026, 1.0, 2.0, 186.0], [-0.202500001, 3.0, 4.0, 149.0], [0.137499988, 5.0, 6.0, 187.0], [0.114500001, 7.0, 8.0, 197.0], [-0.699999988, 9.0, 10.0, 448.0], [0.122500002, 11.0, 12.0, 144.0], [-0.813500047, 13.0, 14.0, 104.0], [0.139938027, 0.0, 0.0, 0.0], [-0.241410524, 0.0, 0.0, 0.0], [0.323894083, 0.0, 0.0, 0.0], [-0.102653548, 0.0, 0.0, 0.0], [0.113428943, 0.0, 0.0, 0.0], [-0.346171468, 0.0, 0.0, 0.0], [0.245157197, 0.0, 0.0, 0.0], [-0.344382942, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.620499969, 1.0, 2.0, 450.0], [-0.142500013, 3.0, 4.0, 37.0], [0.297500014, 5.0, 6.0, 22.0], [-0.444499999, 7.0, 8.0, 65.0], [0.289499998, 9.0, 10.0, 13.0], [-0.0694961846, 0.0, 0.0, 0.0], [0.265451699, 0.0, 0.0, 0.0], [-0.281390935, 0.0, 0.0, 0.0], [0.034012951, 0.0, 0.0, 0.0], [0.206351116, 0.0, 0.0, 0.0], [-0.178720012, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[-0.327499986, 1.0, 2.0, 141.0], [0.720499992, 3.0, 4.0, 184.0], [-0.827499986, 5.0, 6.0, 450.0], [-0.270500004, 7.0, 8.0, 295.0], [0.208424687, 0.0, 0.0, 0.0], [0.308169216, 0.0, 0.0, 0.0], [-0.043009147, 0.0, 0.0, 0.0], [0.104767755, 0.0, 0.0, 0.0], [-0.220349893, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 5, 6])
    branch_indices = np.array([0, 1, 3, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.255500019, 1.0, 2.0, 37.0], [-0.25849998, 3.0, 4.0, 50.0], [-0.591500044, 5.0, 6.0, 508.0], [-0.0414500013, 7.0, 8.0, 504.0], [0.132499993, 9.0, 10.0, 23.0], [-0.293013781, 0.0, 0.0, 0.0], [-0.00224629813, 0.0, 0.0, 0.0], [-0.237265825, 0.0, 0.0, 0.0], [0.0416888818, 0.0, 0.0, 0.0], [-0.14288424, 0.0, 0.0, 0.0], [0.26750043, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.397500008, 1.0, 2.0, 112.0], [-0.978500009, 3.0, 4.0, 163.0], [0.176499993, 5.0, 6.0, 451.0], [0.0909000039, 7.0, 8.0, 158.0], [-0.0815999955, 9.0, 10.0, 452.0], [0.0107000005, 11.0, 12.0, 50.0], [-0.0194500014, 13.0, 14.0, 1.0], [0.0174996406, 0.0, 0.0, 0.0], [0.362174511, 0.0, 0.0, 0.0], [0.0645342246, 0.0, 0.0, 0.0], [-0.366993248, 0.0, 0.0, 0.0], [-0.286744535, 0.0, 0.0, 0.0], [0.0647846609, 0.0, 0.0, 0.0], [-0.164056212, 0.0, 0.0, 0.0], [0.272235692, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[0.397500008, 1.0, 2.0, 112.0], [0.871500015, 3.0, 4.0, 78.0], [0.176499993, 5.0, 6.0, 451.0], [-0.69749999, 7.0, 8.0, 64.0], [-0.972499967, 9.0, 10.0, 444.0], [0.790499985, 11.0, 12.0, 79.0], [0.276000023, 13.0, 14.0, 556.0], [-0.28291139, 0.0, 0.0, 0.0], [-0.00855019409, 0.0, 0.0, 0.0], [0.299755275, 0.0, 0.0, 0.0], [-0.163197517, 0.0, 0.0, 0.0], [0.0725870803, 0.0, 0.0, 0.0], [0.368040591, 0.0, 0.0, 0.0], [-0.266066104, 0.0, 0.0, 0.0], [0.168203324, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.668500006, 1.0, 2.0, 259.0], [0.832499981, 3.0, 4.0, 247.0], [-0.292500019, 5.0, 6.0, 503.0], [-0.23254548, 0.0, 0.0, 0.0], [0.0107457107, 0.0, 0.0, 0.0], [-0.328500003, 7.0, 8.0, 526.0], [-0.100797236, 0.0, 0.0, 0.0], [-0.043713633, 0.0, 0.0, 0.0], [0.261945903, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 6])
    branch_indices = np.array([0, 1, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.912500024, 1.0, 2.0, 136.0], [-0.332499981, 3.0, 4.0, 9.0], [-0.261500001, 5.0, 6.0, 233.0], [-0.249317184, 0.0, 0.0, 0.0], [-0.0346639492, 0.0, 0.0, 0.0], [0.953500032, 7.0, 8.0, 52.0], [-0.104499996, 9.0, 10.0, 148.0], [0.219282508, 0.0, 0.0, 0.0], [-0.113852285, 0.0, 0.0, 0.0], [0.00604219036, 0.0, 0.0, 0.0], [-0.222140461, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[0.0488499999, 1.0, 2.0, 502.0], [-0.60650003, 3.0, 4.0, 316.0], [-0.0961499959, 5.0, 6.0, 92.0], [-0.438499987, 7.0, 8.0, 274.0], [-0.522500038, 9.0, 10.0, 145.0], [0.00773567753, 0.0, 0.0, 0.0], [0.224448964, 0.0, 0.0, 0.0], [-0.182132214, 0.0, 0.0, 0.0], [0.146746948, 0.0, 0.0, 0.0], [-0.0244997963, 0.0, 0.0, 0.0], [-0.291622907, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.146499991, 1.0, 2.0, 53.0], [-0.265500009, 3.0, 4.0, 159.0], [0.56099999, 5.0, 6.0, 556.0], [-0.0066750003, 7.0, 8.0, 144.0], [-0.999500036, 9.0, 10.0, 467.0], [0.614500046, 11.0, 12.0, 159.0], [-0.975499988, 13.0, 14.0, 140.0], [-0.44748491, 0.0, 0.0, 0.0], [-0.0539182872, 0.0, 0.0, 0.0], [0.21380575, 0.0, 0.0, 0.0], [-0.179717183, 0.0, 0.0, 0.0], [0.139493674, 0.0, 0.0, 0.0], [-0.274473697, 0.0, 0.0, 0.0], [-0.397150189, 0.0, 0.0, 0.0], [0.0736602768, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[0.105499998, 1.0, 2.0, 41.0], [-0.917500019, 3.0, 4.0, 558.0], [-0.327000022, 5.0, 6.0, 119.0], [-0.984500051, 7.0, 8.0, 230.0], [0.219500005, 9.0, 10.0, 199.0], [0.0547436103, 0.0, 0.0, 0.0], [-0.397783369, 0.0, 0.0, 0.0], [0.110473879, 0.0, 0.0, 0.0], [-0.359050363, 0.0, 0.0, 0.0], [0.0333827958, 0.0, 0.0, 0.0], [0.352490664, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[0.75999999, 1.0, 2.0, 94.0], [-0.379999995, 3.0, 4.0, 37.0], [-0.207676411, 0.0, 0.0, 0.0], [-0.156016856, 0.0, 0.0, 0.0], [-0.966500044, 5.0, 6.0, 448.0], [-0.0727247521, 0.0, 0.0, 0.0], [0.203464374, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 5, 6, 2])
    branch_indices = np.array([0, 1, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[-0.17050001, 1.0, 2.0, 526.0], [-0.80250001, 3.0, 4.0, 338.0], [-0.362500012, 5.0, 6.0, 459.0], [-0.0751312375, 0.0, 0.0, 0.0], [0.459500015, 7.0, 8.0, 72.0], [-0.103500001, 9.0, 10.0, 198.0], [0.137032717, 0.0, 0.0, 0.0], [0.0075966469, 0.0, 0.0, 0.0], [0.258250087, 0.0, 0.0, 0.0], [0.0182255488, 0.0, 0.0, 0.0], [-0.258302957, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 10, 6])
    branch_indices = np.array([0, 1, 4, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.186500013, 1.0, 2.0, 458.0], [-0.539999962, 3.0, 4.0, 369.0], [-0.179914668, 0.0, 0.0, 0.0], [-0.208499998, 5.0, 6.0, 300.0], [-0.943500042, 7.0, 8.0, 61.0], [0.222788274, 0.0, 0.0, 0.0], [0.0241412073, 0.0, 0.0, 0.0], [-0.162902847, 0.0, 0.0, 0.0], [0.0404373519, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([5, 6, 7, 8, 2])
    branch_indices = np.array([0, 1, 3, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.0418500006, 1.0, 2.0, 538.0], [-0.459500015, 3.0, 4.0, 556.0], [0.0533500016, 5.0, 6.0, 53.0], [-0.982499957, 7.0, 8.0, 45.0], [-0.988499999, 9.0, 10.0, 179.0], [-0.925500035, 11.0, 12.0, 540.0], [0.230720311, 0.0, 0.0, 0.0], [0.0566845126, 0.0, 0.0, 0.0], [-0.372111261, 0.0, 0.0, 0.0], [0.245144323, 0.0, 0.0, 0.0], [-0.0170167852, 0.0, 0.0, 0.0], [0.153502733, 0.0, 0.0, 0.0], [-0.303626627, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 6])
    branch_indices = np.array([0, 1, 3, 4, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.994500041, 1.0, 2.0, 347.0], [-0.518999994, 3.0, 4.0, 65.0], [-0.983500004, 5.0, 6.0, 432.0], [-0.245499998, 7.0, 8.0, 455.0], [-0.997500002, 9.0, 10.0, 166.0], [-0.372000009, 11.0, 12.0, 262.0], [-0.926499963, 13.0, 14.0, 434.0], [-0.0585686862, 0.0, 0.0, 0.0], [-0.253100067, 0.0, 0.0, 0.0], [-0.224797845, 0.0, 0.0, 0.0], [0.312561035, 0.0, 0.0, 0.0], [0.0318541117, 0.0, 0.0, 0.0], [-0.317286909, 0.0, 0.0, 0.0], [0.155800343, 0.0, 0.0, 0.0], [-0.19583565, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.75849998, 1.0, 2.0, 450.0], [-0.12650001, 3.0, 4.0, 37.0], [-0.781499982, 5.0, 6.0, 459.0], [-0.217871696, 0.0, 0.0, 0.0], [0.054470513, 0.0, 0.0, 0.0], [-0.0702835843, 0.0, 0.0, 0.0], [0.198773459, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 6])
    branch_indices = np.array([0, 1, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.476500005, 1.0, 2.0, 102.0], [-0.158669084, 0.0, 0.0, 0.0], [-0.75849998, 3.0, 4.0, 450.0], [-0.0882499963, 5.0, 6.0, 117.0], [-0.089347735, 0.0, 0.0, 0.0], [0.224287495, 0.0, 0.0, 0.0], [-0.0628341958, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 5, 6, 4])
    branch_indices = np.array([0, 2, 3])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.756500006, 1.0, 2.0, 91.0], [-0.162093461, 0.0, 0.0, 0.0], [-0.838500023, 3.0, 4.0, 470.0], [-0.515499949, 5.0, 6.0, 434.0], [-0.107106335, 0.0, 0.0, 0.0], [0.220712885, 0.0, 0.0, 0.0], [-0.00155732408, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 5, 6, 4])
    branch_indices = np.array([0, 2, 3])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.124499999, 1.0, 2.0, 50.0], [-0.00565499999, 3.0, 4.0, 159.0], [-0.176499993, 5.0, 6.0, 28.0], [-0.000850000011, 7.0, 8.0, 145.0], [-0.0088500008, 9.0, 10.0, 118.0], [-0.00615500007, 11.0, 12.0, 1.0], [0.487500012, 13.0, 14.0, 557.0], [-0.326099992, 0.0, 0.0, 0.0], [0.00220697769, 0.0, 0.0, 0.0], [0.233833238, 0.0, 0.0, 0.0], [-0.0839293376, 0.0, 0.0, 0.0], [-0.266833782, 0.0, 0.0, 0.0], [0.183231473, 0.0, 0.0, 0.0], [0.186653346, 0.0, 0.0, 0.0], [-0.153953791, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[0.202500001, 1.0, 2.0, 39.0], [-0.138500005, 3.0, 4.0, 28.0], [-0.0294499993, 5.0, 6.0, 50.0], [-0.362999976, 7.0, 8.0, 301.0], [-0.997500002, 9.0, 10.0, 170.0], [-6.00000021e-05, 11.0, 12.0, 192.0], [-0.319000006, 13.0, 14.0, 538.0], [-0.114034541, 0.0, 0.0, 0.0], [0.296460479, 0.0, 0.0, 0.0], [0.149205431, 0.0, 0.0, 0.0], [-0.207589626, 0.0, 0.0, 0.0], [-0.0265873671, 0.0, 0.0, 0.0], [0.281899124, 0.0, 0.0, 0.0], [0.126309395, 0.0, 0.0, 0.0], [-0.314106345, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.905499995, 1.0, 2.0, 461.0], [-0.155932203, 0.0, 0.0, 0.0], [-0.345499992, 3.0, 4.0, 274.0], [-0.45450002, 5.0, 6.0, 62.0], [0.151499987, 7.0, 8.0, 295.0], [0.244361565, 0.0, 0.0, 0.0], [0.0319264233, 0.0, 0.0, 0.0], [-0.190368935, 0.0, 0.0, 0.0], [0.088557072, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 5, 6, 7, 8])
    branch_indices = np.array([0, 2, 3, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[0.695500016, 1.0, 2.0, 76.0], [-0.995499969, 3.0, 4.0, 58.0], [-0.772500038, 5.0, 6.0, 256.0], [0.0849394128, 0.0, 0.0, 0.0], [-0.0861500055, 7.0, 8.0, 458.0], [-0.0302983578, 0.0, 0.0, 0.0], [0.197584212, 0.0, 0.0, 0.0], [-0.234555438, 0.0, 0.0, 0.0], [0.0299224518, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 5, 6])
    branch_indices = np.array([0, 1, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[0.12650001, 1.0, 2.0, 9.0], [0.543500006, 3.0, 4.0, 102.0], [0.665500045, 5.0, 6.0, 102.0], [-0.00821538456, 0.0, 0.0, 0.0], [-0.170882165, 0.0, 0.0, 0.0], [0.191524327, 0.0, 0.0, 0.0], [-0.0498304032, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 6])
    branch_indices = np.array([0, 1, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.842499971, 1.0, 2.0, 132.0], [-0.986500025, 3.0, 4.0, 444.0], [-0.735000014, 5.0, 6.0, 22.0], [-0.00652500009, 7.0, 8.0, 155.0], [-0.973500013, 9.0, 10.0, 179.0], [-0.102926672, 0.0, 0.0, 0.0], [-0.113499999, 11.0, 12.0, 2.0], [0.0700308606, 0.0, 0.0, 0.0], [-0.299284637, 0.0, 0.0, 0.0], [0.139544502, 0.0, 0.0, 0.0], [-0.136250153, 0.0, 0.0, 0.0], [-0.0276048407, 0.0, 0.0, 0.0], [0.361699373, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 4, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[0.842499971, 1.0, 2.0, 132.0], [-0.986500025, 3.0, 4.0, 444.0], [-0.735000014, 5.0, 6.0, 22.0], [-0.00652500009, 7.0, 8.0, 155.0], [-0.100500003, 9.0, 10.0, 551.0], [0.108295456, 0.0, 0.0, 0.0], [-0.113499999, 11.0, 12.0, 2.0], [-0.0656347871, 0.0, 0.0, 0.0], [0.303445369, 0.0, 0.0, 0.0], [0.116942227, 0.0, 0.0, 0.0], [-0.138935015, 0.0, 0.0, 0.0], [0.0317435935, 0.0, 0.0, 0.0], [-0.357007831, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 4, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.510499954, 1.0, 2.0, 72.0], [-0.25849998, 3.0, 4.0, 37.0], [-0.116563283, 0.0, 0.0, 0.0], [-0.0628177151, 0.0, 0.0, 0.0], [0.177845746, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 2])
    branch_indices = np.array([0, 1])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.607499957, 1.0, 2.0, 527.0], [-0.206499994, 3.0, 4.0, 53.0], [-0.261500001, 5.0, 6.0, 298.0], [0.178862557, 0.0, 0.0, 0.0], [-0.0331517495, 0.0, 0.0, 0.0], [-0.163814157, 0.0, 0.0, 0.0], [0.0406458452, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 6])
    branch_indices = np.array([0, 1, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.280499995, 1.0, 2.0, 89.0], [-0.404500008, 3.0, 4.0, 298.0], [0.121348046, 0.0, 0.0, 0.0], [0.040187154, 0.0, 0.0, 0.0], [-0.150063574, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 2])
    branch_indices = np.array([0, 1])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.1505, 1.0, 2.0, 36.0], [0.0725999996, 3.0, 4.0, 41.0], [-0.1505, 5.0, 6.0, 42.0], [-0.917500019, 7.0, 8.0, 558.0], [0.210452124, 0.0, 0.0, 0.0], [-0.0864000022, 9.0, 10.0, 113.0], [-0.948500037, 11.0, 12.0, 130.0], [0.149524331, 0.0, 0.0, 0.0], [-0.201918274, 0.0, 0.0, 0.0], [-0.268064499, 0.0, 0.0, 0.0], [-0.0088571161, 0.0, 0.0, 0.0], [-0.0580828711, 0.0, 0.0, 0.0], [0.197648495, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 3, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.355000019, 1.0, 2.0, 301.0], [-0.596500039, 3.0, 4.0, 104.0], [-0.0377999991, 5.0, 6.0, 118.0], [-0.988499999, 7.0, 8.0, 166.0], [0.0130000003, 9.0, 10.0, 109.0], [-0.0164499991, 11.0, 12.0, 50.0], [0.274500012, 13.0, 14.0, 197.0], [-0.113772973, 0.0, 0.0, 0.0], [0.220455348, 0.0, 0.0, 0.0], [-0.303487688, 0.0, 0.0, 0.0], [0.00373756466, 0.0, 0.0, 0.0], [0.325628728, 0.0, 0.0, 0.0], [-0.0794839561, 0.0, 0.0, 0.0], [0.0498109423, 0.0, 0.0, 0.0], [-0.262749285, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.46450001, 1.0, 2.0, 354.0], [-0.119766459, 0.0, 0.0, 0.0], [-0.222000003, 3.0, 4.0, 202.0], [0.148272857, 0.0, 0.0, 0.0], [-0.0491786823, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 4])
    branch_indices = np.array([0, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[-0.188499987, 1.0, 2.0, 374.0], [0.321500003, 3.0, 4.0, 92.0], [-0.137264013, 0.0, 0.0, 0.0], [0.190499991, 5.0, 6.0, 24.0], [0.699499965, 7.0, 8.0, 76.0], [0.1943326, 0.0, 0.0, 0.0], [0.011677104, 0.0, 0.0, 0.0], [-0.12269935, 0.0, 0.0, 0.0], [0.034616854, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([5, 6, 7, 8, 2])
    branch_indices = np.array([0, 1, 3, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.111500002, 1.0, 2.0, 202.0], [-0.3495, 3.0, 4.0, 37.0], [0.130513266, 0.0, 0.0, 0.0], [0.0383916423, 0.0, 0.0, 0.0], [-0.162940338, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 2])
    branch_indices = np.array([0, 1])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.0501499996, 1.0, 2.0, 185.0], [-0.984500051, 3.0, 4.0, 101.0], [-0.168500006, 5.0, 6.0, 41.0], [-0.114999995, 7.0, 8.0, 249.0], [-0.98150003, 9.0, 10.0, 166.0], [-0.161205366, 0.0, 0.0, 0.0], [-0.831499994, 11.0, 12.0, 380.0], [-0.00931583438, 0.0, 0.0, 0.0], [-0.303646654, 0.0, 0.0, 0.0], [0.178650573, 0.0, 0.0, 0.0], [-0.105757616, 0.0, 0.0, 0.0], [0.267485499, 0.0, 0.0, 0.0], [-0.0111316927, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 11, 12])
    branch_indices = np.array([0, 1, 3, 4, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[0.0501499996, 1.0, 2.0, 185.0], [-0.984500051, 3.0, 4.0, 101.0], [-0.147, 5.0, 6.0, 53.0], [-0.114999995, 7.0, 8.0, 249.0], [-0.98150003, 9.0, 10.0, 166.0], [-0.400999993, 11.0, 12.0, 159.0], [-0.989500046, 13.0, 14.0, 364.0], [0.014538344, 0.0, 0.0, 0.0], [0.307329267, 0.0, 0.0, 0.0], [-0.174497962, 0.0, 0.0, 0.0], [0.0774256662, 0.0, 0.0, 0.0], [0.23717235, 0.0, 0.0, 0.0], [0.00783868134, 0.0, 0.0, 0.0], [-0.0342046954, 0.0, 0.0, 0.0], [-0.313031346, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.709499955, 1.0, 2.0, 449.0], [0.556499958, 3.0, 4.0, 104.0], [0.109003328, 0.0, 0.0, 0.0], [-0.157782331, 0.0, 0.0, 0.0], [0.0506744683, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 2])
    branch_indices = np.array([0, 1])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.458499998, 1.0, 2.0, 141.0], [-0.245499998, 3.0, 4.0, 295.0], [-0.451499999, 5.0, 6.0, 88.0], [0.0488311686, 0.0, 0.0, 0.0], [-0.144027978, 0.0, 0.0, 0.0], [0.174940243, 0.0, 0.0, 0.0], [-0.0280808769, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 6])
    branch_indices = np.array([0, 1, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.700000048, 1.0, 2.0, 369.0], [0.102620661, 0.0, 0.0, 0.0], [-0.0846615881, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.182500005, 1.0, 2.0, 54.0], [0.0734000057, 3.0, 4.0, 54.0], [-0.14199999, 5.0, 6.0, 117.0], [-0.124499999, 7.0, 8.0, 50.0], [-0.0108000003, 9.0, 10.0, 50.0], [-0.0554477982, 0.0, 0.0, 0.0], [0.306753069, 0.0, 0.0, 0.0], [-0.126984358, 0.0, 0.0, 0.0], [0.11744494, 0.0, 0.0, 0.0], [-0.286509156, 0.0, 0.0, 0.0], [0.134595394, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.0996499956, 1.0, 2.0, 121.0], [0.00877499953, 3.0, 4.0, 187.0], [-0.966500044, 5.0, 6.0, 291.0], [0.0145628694, 0.0, 0.0, 0.0], [-0.263995558, 0.0, 0.0, 0.0], [-0.00693000015, 7.0, 8.0, 458.0], [-0.120499998, 9.0, 10.0, 41.0], [0.171497807, 0.0, 0.0, 0.0], [-0.0534835085, 0.0, 0.0, 0.0], [0.0708545148, 0.0, 0.0, 0.0], [-0.17240037, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[0.254500002, 1.0, 2.0, 198.0], [0.0118000004, 3.0, 4.0, 42.0], [0.1081606, 0.0, 0.0, 0.0], [-0.134065881, 0.0, 0.0, 0.0], [0.0500638932, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 2])
    branch_indices = np.array([0, 1])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[-0.499500006, 1.0, 2.0, 274.0], [-0.109142475, 0.0, 0.0, 0.0], [-0.946500003, 3.0, 4.0, 418.0], [0.15758808, 0.0, 0.0, 0.0], [0.102499999, 5.0, 6.0, 197.0], [0.064111948, 0.0, 0.0, 0.0], [-0.108250849, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 5, 6])
    branch_indices = np.array([0, 2, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[0.032250002, 1.0, 2.0, 159.0], [0.255500019, 3.0, 4.0, 28.0], [-0.0994062871, 0.0, 0.0, 0.0], [-0.00651425589, 0.0, 0.0, 0.0], [0.143007815, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 2])
    branch_indices = np.array([0, 1])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.188499987, 1.0, 2.0, 199.0], [-0.0106499996, 3.0, 4.0, 41.0], [0.0700500011, 5.0, 6.0, 372.0], [0.00797499996, 7.0, 8.0, 196.0], [0.267499983, 9.0, 10.0, 188.0], [0.0474360473, 0.0, 0.0, 0.0], [-0.244348913, 0.0, 0.0, 0.0], [0.106779993, 0.0, 0.0, 0.0], [-0.119707383, 0.0, 0.0, 0.0], [0.270339817, 0.0, 0.0, 0.0], [-0.021213159, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[0.115500003, 1.0, 2.0, 196.0], [0.469500005, 3.0, 4.0, 159.0], [-0.793500006, 5.0, 6.0, 455.0], [0.949499965, 7.0, 8.0, 172.0], [0.144801363, 0.0, 0.0, 0.0], [0.904500008, 9.0, 10.0, 56.0], [-0.994500041, 11.0, 12.0, 179.0], [0.0921619684, 0.0, 0.0, 0.0], [-0.176760584, 0.0, 0.0, 0.0], [-0.205612913, 0.0, 0.0, 0.0], [0.0889329389, 0.0, 0.0, 0.0], [-0.0711270943, 0.0, 0.0, 0.0], [0.23138465, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 3, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.871500015, 1.0, 2.0, 537.0], [-0.0965043604, 0.0, 0.0, 0.0], [-0.956499994, 3.0, 4.0, 44.0], [0.151769295, 0.0, 0.0, 0.0], [-0.0416374356, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 4])
    branch_indices = np.array([0, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.607499957, 1.0, 2.0, 527.0], [0.0961374715, 0.0, 0.0, 0.0], [0.0495500006, 3.0, 4.0, 114.0], [0.0479085594, 0.0, 0.0, 0.0], [-0.124852441, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 4])
    branch_indices = np.array([0, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.306500018, 1.0, 2.0, 298.0], [0.573500037, 3.0, 4.0, 103.0], [-0.0755297393, 0.0, 0.0, 0.0], [0.127784654, 0.0, 0.0, 0.0], [-0.00128818897, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 2])
    branch_indices = np.array([0, 1])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.993499994, 1.0, 2.0, 179.0], [-0.993499994, 3.0, 4.0, 430.0], [0.989500046, 5.0, 6.0, 174.0], [-0.081882976, 0.0, 0.0, 0.0], [0.254554302, 0.0, 0.0, 0.0], [-0.982499957, 7.0, 8.0, 432.0], [-0.992499948, 9.0, 10.0, 47.0], [0.253222018, 0.0, 0.0, 0.0], [-0.0471566133, 0.0, 0.0, 0.0], [-0.0360237099, 0.0, 0.0, 0.0], [-0.257357776, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.967499971, 1.0, 2.0, 444.0], [-0.986500025, 3.0, 4.0, 423.0], [0.073300004, 5.0, 6.0, 51.0], [-0.993499994, 7.0, 8.0, 424.0], [-0.9745, 9.0, 10.0, 140.0], [-0.192499995, 11.0, 12.0, 294.0], [-0.968500018, 13.0, 14.0, 99.0], [0.108182572, 0.0, 0.0, 0.0], [-0.204665169, 0.0, 0.0, 0.0], [0.23653397, 0.0, 0.0, 0.0], [0.0134685989, 0.0, 0.0, 0.0], [-0.00856729504, 0.0, 0.0, 0.0], [-0.268378377, 0.0, 0.0, 0.0], [0.137925759, 0.0, 0.0, 0.0], [-0.0892137811, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.439500004, 1.0, 2.0, 379.0], [-0.0866549239, 0.0, 0.0, 0.0], [0.0765303746, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[-0.109499998, 1.0, 2.0, 157.0], [-0.257499993, 3.0, 4.0, 117.0], [0.0798727646, 0.0, 0.0, 0.0], [0.0482672825, 0.0, 0.0, 0.0], [-0.135368153, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 2])
    branch_indices = np.array([0, 1])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.435499996, 1.0, 2.0, 89.0], [-0.0796915516, 0.0, 0.0, 0.0], [0.0839186758, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.13350001, 1.0, 2.0, 50.0], [-0.0243499987, 3.0, 4.0, 159.0], [-0.905499995, 5.0, 6.0, 423.0], [0.29549998, 7.0, 8.0, 187.0], [0.0600421913, 0.0, 0.0, 0.0], [0.630500019, 9.0, 10.0, 72.0], [-0.13653101, 0.0, 0.0, 0.0], [-0.248121753, 0.0, 0.0, 0.0], [-0.00673344964, 0.0, 0.0, 0.0], [0.0300899148, 0.0, 0.0, 0.0], [0.239237502, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 6])
    branch_indices = np.array([0, 1, 3, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[0.53550005, 1.0, 2.0, 13.0], [-0.204366446, 0.0, 0.0, 0.0], [-0.166500002, 3.0, 4.0, 41.0], [-0.26699999, 5.0, 6.0, 159.0], [-0.976500034, 7.0, 8.0, 139.0], [0.28376478, 0.0, 0.0, 0.0], [-0.0206758734, 0.0, 0.0, 0.0], [-0.102842599, 0.0, 0.0, 0.0], [0.0924657807, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 5, 6, 7, 8])
    branch_indices = np.array([0, 2, 3, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[0.3495, 1.0, 2.0, 22.0], [-0.0801077336, 0.0, 0.0, 0.0], [0.0823493078, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.0757500008, 1.0, 2.0, 197.0], [0.0952440426, 0.0, 0.0, 0.0], [-0.265500009, 3.0, 4.0, 53.0], [0.0487864614, 0.0, 0.0, 0.0], [-0.113254473, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 4])
    branch_indices = np.array([0, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[0.109499998, 1.0, 2.0, 24.0], [-0.0758588985, 0.0, 0.0, 0.0], [0.077233687, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.115500003, 1.0, 2.0, 41.0], [-0.700000048, 3.0, 4.0, 370.0], [0.193183243, 0.0, 0.0, 0.0], [-0.958500028, 5.0, 6.0, 432.0], [-0.584499955, 7.0, 8.0, 142.0], [0.184771165, 0.0, 0.0, 0.0], [0.0191072505, 0.0, 0.0, 0.0], [0.0766017959, 0.0, 0.0, 0.0], [-0.119978793, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([5, 6, 7, 8, 2])
    branch_indices = np.array([0, 1, 3, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.979499996, 1.0, 2.0, 230.0], [-0.986500025, 3.0, 4.0, 429.0], [0.149500012, 5.0, 6.0, 295.0], [-0.992499948, 7.0, 8.0, 359.0], [-0.0815500021, 9.0, 10.0, 114.0], [-0.316500008, 11.0, 12.0, 264.0], [0.0909563303, 0.0, 0.0, 0.0], [0.113641471, 0.0, 0.0, 0.0], [-0.16555573, 0.0, 0.0, 0.0], [-0.0135812629, 0.0, 0.0, 0.0], [0.223386005, 0.0, 0.0, 0.0], [0.0640829876, 0.0, 0.0, 0.0], [-0.196353704, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 6])
    branch_indices = np.array([0, 1, 3, 4, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.75849998, 1.0, 2.0, 450.0], [-0.0681040511, 0.0, 0.0, 0.0], [0.0776754543, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[-0.445500016, 1.0, 2.0, 141.0], [-0.0610519759, 0.0, 0.0, 0.0], [0.0776064321, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[0.159500003, 1.0, 2.0, 28.0], [-0.0768986493, 0.0, 0.0, 0.0], [0.0707452744, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[0.12650001, 1.0, 2.0, 192.0], [-0.999500036, 3.0, 4.0, 416.0], [-0.789499998, 5.0, 6.0, 455.0], [0.324999988, 7.0, 8.0, 159.0], [-0.0298500005, 9.0, 10.0, 117.0], [0.888499975, 11.0, 12.0, 49.0], [-0.244500011, 13.0, 14.0, 189.0], [0.174357787, 0.0, 0.0, 0.0], [-0.0847218782, 0.0, 0.0, 0.0], [-0.160260022, 0.0, 0.0, 0.0], [0.0462901294, 0.0, 0.0, 0.0], [0.145926565, 0.0, 0.0, 0.0], [-0.0440268889, 0.0, 0.0, 0.0], [0.00329674338, 0.0, 0.0, 0.0], [-0.207635328, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[0.125499994, 1.0, 2.0, 192.0], [0.431500018, 3.0, 4.0, 159.0], [-0.787500024, 5.0, 6.0, 455.0], [-0.715499997, 7.0, 8.0, 457.0], [0.135561883, 0.0, 0.0, 0.0], [0.897500038, 9.0, 10.0, 56.0], [-0.978500009, 11.0, 12.0, 506.0], [0.036285121, 0.0, 0.0, 0.0], [-0.179578185, 0.0, 0.0, 0.0], [-0.138938233, 0.0, 0.0, 0.0], [0.0496284701, 0.0, 0.0, 0.0], [0.194167778, 0.0, 0.0, 0.0], [-0.0389182344, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 3, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.941499949, 1.0, 2.0, 61.0], [0.069016248, 0.0, 0.0, 0.0], [-0.0774583742, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.611500025, 1.0, 2.0, 73.0], [0.0840372071, 0.0, 0.0, 0.0], [0.145500004, 3.0, 4.0, 210.0], [-0.112776175, 0.0, 0.0, 0.0], [0.0222000834, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 3, 4])
    branch_indices = np.array([0, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.280499995, 1.0, 2.0, 89.0], [-0.0603219792, 0.0, 0.0, 0.0], [0.0898606256, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.0599500015, 1.0, 2.0, 53.0], [0.234499991, 3.0, 4.0, 187.0], [-0.0230999999, 5.0, 6.0, 300.0], [0.0445000008, 7.0, 8.0, 159.0], [-0.9745, 9.0, 10.0, 58.0], [0.95449996, 11.0, 12.0, 172.0], [-0.98150003, 13.0, 14.0, 58.0], [-0.235297605, 0.0, 0.0, 0.0], [0.0236584302, 0.0, 0.0, 0.0], [-0.0906165987, 0.0, 0.0, 0.0], [0.102263935, 0.0, 0.0, 0.0], [-0.0309239868, 0.0, 0.0, 0.0], [0.202147931, 0.0, 0.0, 0.0], [0.09433157, 0.0, 0.0, 0.0], [-0.139703557, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 13, 14])
    branch_indices = np.array([0, 1, 3, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.362500012, 1.0, 2.0, 301.0], [-0.318499982, 3.0, 4.0, 24.0], [0.115500003, 5.0, 6.0, 110.0], [-0.995499969, 7.0, 8.0, 139.0], [-0.959499955, 9.0, 10.0, 179.0], [-0.0923999995, 11.0, 12.0, 121.0], [-0.0713040978, 0.0, 0.0, 0.0], [-0.142279401, 0.0, 0.0, 0.0], [0.0863964856, 0.0, 0.0, 0.0], [-0.21989809, 0.0, 0.0, 0.0], [-0.048592709, 0.0, 0.0, 0.0], [-0.0293135606, 0.0, 0.0, 0.0], [0.187037304, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 6])
    branch_indices = np.array([0, 1, 3, 4, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.134499997, 1.0, 2.0, 37.0], [-0.0611581653, 0.0, 0.0, 0.0], [0.0746474117, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[0.299499989, 1.0, 2.0, 92.0], [0.0698652491, 0.0, 0.0, 0.0], [-0.0762526542, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.3495, 1.0, 2.0, 37.0], [0.0684687793, 0.0, 0.0, 0.0], [-0.0654521659, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.136500001, 1.0, 2.0, 53.0], [-0.265500009, 3.0, 4.0, 159.0], [0.467500001, 5.0, 6.0, 556.0], [-0.209198624, 0.0, 0.0, 0.0], [0.00768825365, 0.0, 0.0, 0.0], [-0.976500034, 7.0, 8.0, 139.0], [-0.0949979499, 0.0, 0.0, 0.0], [0.139992073, 0.0, 0.0, 0.0], [-0.0364196077, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 6])
    branch_indices = np.array([0, 1, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.267499983, 1.0, 2.0, 251.0], [0.271499991, 3.0, 4.0, 108.0], [-0.95449996, 5.0, 6.0, 85.0], [-0.194459155, 0.0, 0.0, 0.0], [-0.0108359568, 0.0, 0.0, 0.0], [-0.988499999, 7.0, 8.0, 179.0], [-0.110159285, 0.0, 0.0, 0.0], [-0.044198323, 0.0, 0.0, 0.0], [0.160900638, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 6])
    branch_indices = np.array([0, 1, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.598500013, 1.0, 2.0, 432.0], [-0.0780448392, 0.0, 0.0, 0.0], [0.0635976344, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    function_dict = np.array([[-0.0194500014, 1.0, 2.0, 1.0], [0.0646485239, 0.0, 0.0, 0.0], [-0.0637731701, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[0.286500007, 1.0, 2.0, 0.0], [-0.0597981811, 0.0, 0.0, 0.0], [0.0727693737, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.244000003, 1.0, 2.0, 540.0], [0.271499991, 3.0, 4.0, 556.0], [-0.0920000002, 5.0, 6.0, 160.0], [-0.709499955, 7.0, 8.0, 290.0], [-0.03895, 9.0, 10.0, 161.0], [0.00844349433, 0.0, 0.0, 0.0], [0.189939529, 0.0, 0.0, 0.0], [-0.0761144608, 0.0, 0.0, 0.0], [0.0916316137, 0.0, 0.0, 0.0], [-0.0385047086, 0.0, 0.0, 0.0], [-0.210666463, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.0834500045, 1.0, 2.0, 154.0], [-0.995499969, 3.0, 4.0, 166.0], [0.0719500035, 5.0, 6.0, 42.0], [-0.092816852, 0.0, 0.0, 0.0], [-0.980499983, 7.0, 8.0, 140.0], [0.891499996, 9.0, 10.0, 49.0], [0.882500052, 11.0, 12.0, 56.0], [0.200250626, 0.0, 0.0, 0.0], [-0.0151225422, 0.0, 0.0, 0.0], [-0.0257395208, 0.0, 0.0, 0.0], [-0.234252751, 0.0, 0.0, 0.0], [-0.0627428517, 0.0, 0.0, 0.0], [0.146693483, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 0
    function_dict = np.array([[-0.907500029, 1.0, 2.0, 327.0], [-0.0760909468, 0.0, 0.0, 0.0], [0.0598060526, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 1
    function_dict = np.array([[-0.163500011, 1.0, 2.0, 526.0], [0.0586062111, 0.0, 0.0, 0.0], [-0.0652891397, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 2
    function_dict = np.array([[-0.271499991, 1.0, 2.0, 509.0], [-0.0517857671, 0.0, 0.0, 0.0], [0.0731932223, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 3
    function_dict = np.array([[-0.879500031, 1.0, 2.0, 429.0], [-0.475499988, 3.0, 4.0, 446.0], [-0.154338121, 0.0, 0.0, 0.0], [0.441500008, 5.0, 6.0, 158.0], [-0.29549998, 7.0, 8.0, 118.0], [-0.0920627639, 0.0, 0.0, 0.0], [0.116660416, 0.0, 0.0, 0.0], [-0.0425254367, 0.0, 0.0, 0.0], [0.200158, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([5, 6, 7, 8, 2])
    branch_indices = np.array([0, 1, 3, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    #Predicts Class 4
    function_dict = np.array([[-0.966500044, 1.0, 2.0, 448.0], [-0.255500019, 3.0, 4.0, 451.0], [-0.37650001, 5.0, 6.0, 454.0], [0.0985040665, 0.0, 0.0, 0.0], [0.17050001, 7.0, 8.0, 27.0], [0.0633499995, 9.0, 10.0, 373.0], [-0.461000025, 11.0, 12.0, 289.0], [-0.203403518, 0.0, 0.0, 0.0], [0.0150597701, 0.0, 0.0, 0.0], [-0.109944284, 0.0, 0.0, 0.0], [0.0813009143, 0.0, 0.0, 0.0], [0.208686635, 0.0, 0.0, 0.0], [-0.0341796093, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 10, 11, 12])
    branch_indices = np.array([0, 1, 4, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_150(xs):
    #Predicts Class 0
    function_dict = np.array([[-0.961500049, 1.0, 2.0, 413.0], [-0.0744211003, 0.0, 0.0, 0.0], [0.0570644923, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_151(xs):
    #Predicts Class 1
    function_dict = np.array([[-0.417500019, 1.0, 2.0, 274.0], [-0.0739292353, 0.0, 0.0, 0.0], [0.0516208969, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_152(xs):
    #Predicts Class 2
    function_dict = np.array([[-0.317499995, 1.0, 2.0, 275.0], [0.0626853108, 0.0, 0.0, 0.0], [-0.06083671, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_153(xs):
    #Predicts Class 3
    function_dict = np.array([[-0.855499983, 1.0, 2.0, 129.0], [0.845499992, 3.0, 4.0, 12.0], [-0.539999962, 5.0, 6.0, 371.0], [-0.979499996, 7.0, 8.0, 345.0], [0.109499998, 9.0, 10.0, 145.0], [0.0384841599, 0.0, 0.0, 0.0], [-0.172545478, 0.0, 0.0, 0.0], [0.0199306849, 0.0, 0.0, 0.0], [0.185261115, 0.0, 0.0, 0.0], [-0.093516238, 0.0, 0.0, 0.0], [0.0852984041, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_154(xs):
    #Predicts Class 4
    function_dict = np.array([[-0.130499989, 1.0, 2.0, 142.0], [-0.0111499997, 3.0, 4.0, 50.0], [-0.1175, 5.0, 6.0, 25.0], [-0.984500051, 7.0, 8.0, 278.0], [-0.163412571, 0.0, 0.0, 0.0], [-0.026314266, 0.0, 0.0, 0.0], [0.155811995, 0.0, 0.0, 0.0], [0.0823622495, 0.0, 0.0, 0.0], [-0.106923774, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 5, 6])
    branch_indices = np.array([0, 1, 3, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_155(xs):
    #Predicts Class 0
    function_dict = np.array([[-0.507500052, 1.0, 2.0, 62.0], [0.0614084303, 0.0, 0.0, 0.0], [-0.0693434626, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_156(xs):
    #Predicts Class 1
    function_dict = np.array([[-0.403500021, 1.0, 2.0, 374.0], [0.0619641095, 0.0, 0.0, 0.0], [-0.0605184659, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_157(xs):
    #Predicts Class 2
    function_dict = np.array([[0.190499991, 1.0, 2.0, 9.0], [-0.0558866784, 0.0, 0.0, 0.0], [0.0748102143, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_158(xs):
    #Predicts Class 3
    function_dict = np.array([[-0.979499996, 1.0, 2.0, 179.0], [0.132499993, 3.0, 4.0, 188.0], [-0.0199499987, 5.0, 6.0, 50.0], [-0.150999993, 7.0, 8.0, 236.0], [-0.861500025, 9.0, 10.0, 234.0], [0.109999999, 11.0, 12.0, 37.0], [0.0490542315, 0.0, 0.0, 0.0], [-0.130916685, 0.0, 0.0, 0.0], [0.0849391595, 0.0, 0.0, 0.0], [2.25323256e-05, 0.0, 0.0, 0.0], [0.168152645, 0.0, 0.0, 0.0], [-0.185763285, 0.0, 0.0, 0.0], [-0.0247836299, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 11, 12, 6])
    branch_indices = np.array([0, 1, 3, 4, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_159(xs):
    #Predicts Class 4
    function_dict = np.array([[-0.921000004, 1.0, 2.0, 524.0], [-0.603500009, 3.0, 4.0, 459.0], [-0.827000022, 5.0, 6.0, 553.0], [0.00894500036, 7.0, 8.0, 212.0], [0.0946000069, 9.0, 10.0, 115.0], [0.0112695536, 0.0, 0.0, 0.0], [-0.159392595, 0.0, 0.0, 0.0], [0.150391519, 0.0, 0.0, 0.0], [-0.00554232486, 0.0, 0.0, 0.0], [-0.124320127, 0.0, 0.0, 0.0], [0.0947931483, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 9, 10, 5, 6])
    branch_indices = np.array([0, 1, 3, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_160(xs):
    #Predicts Class 0
    function_dict = np.array([[-0.323500007, 1.0, 2.0, 377.0], [-0.0662255138, 0.0, 0.0, 0.0], [0.0538618788, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_161(xs):
    #Predicts Class 1
    function_dict = np.array([[0.129999995, 1.0, 2.0, 92.0], [0.0667842403, 0.0, 0.0, 0.0], [-0.056343589, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_162(xs):
    #Predicts Class 2
    function_dict = np.array([[-0.74000001, 1.0, 2.0, 369.0], [0.070476979, 0.0, 0.0, 0.0], [-0.0538656302, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_163(xs):
    #Predicts Class 3
    function_dict = np.array([[-0.113499999, 1.0, 2.0, 50.0], [-0.000850000011, 3.0, 4.0, 145.0], [0.188499987, 5.0, 6.0, 199.0], [-0.150427252, 0.0, 0.0, 0.0], [0.0336291678, 0.0, 0.0, 0.0], [-0.519500017, 7.0, 8.0, 296.0], [-0.0660200268, 0.0, 0.0, 0.0], [0.162461266, 0.0, 0.0, 0.0], [-0.0110556334, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 6])
    branch_indices = np.array([0, 1, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_164(xs):
    #Predicts Class 4
    function_dict = np.array([[0.173500001, 1.0, 2.0, 54.0], [-0.02905, 3.0, 4.0, 560.0], [-0.119754858, 0.0, 0.0, 0.0], [0.343500018, 5.0, 6.0, 112.0], [-0.820500016, 7.0, 8.0, 558.0], [0.0291373264, 0.0, 0.0, 0.0], [0.172827154, 0.0, 0.0, 0.0], [-0.111513756, 0.0, 0.0, 0.0], [0.0882367119, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([5, 6, 7, 8, 2])
    branch_indices = np.array([0, 1, 3, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_165(xs):
    #Predicts Class 0
    function_dict = np.array([[-0.154500008, 1.0, 2.0, 209.0], [-0.0677094087, 0.0, 0.0, 0.0], [0.0521571226, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_166(xs):
    #Predicts Class 1
    function_dict = np.array([[-0.459999979, 1.0, 2.0, 371.0], [0.0586574934, 0.0, 0.0, 0.0], [-0.0541425273, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_167(xs):
    #Predicts Class 2
    function_dict = np.array([[-0.359499991, 1.0, 2.0, 456.0], [0.0628856495, 0.0, 0.0, 0.0], [-0.0550006814, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_168(xs):
    #Predicts Class 3
    function_dict = np.array([[-0.0921999961, 1.0, 2.0, 451.0], [-0.412499994, 3.0, 4.0, 38.0], [0.912500024, 5.0, 6.0, 56.0], [0.0551874712, 0.0, 0.0, 0.0], [0.108999997, 7.0, 8.0, 197.0], [-0.0181500018, 9.0, 10.0, 1.0], [-0.0566500723, 0.0, 0.0, 0.0], [-0.177468449, 0.0, 0.0, 0.0], [-0.00548628392, 0.0, 0.0, 0.0], [-0.0142720779, 0.0, 0.0, 0.0], [0.136548713, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 9, 10, 6])
    branch_indices = np.array([0, 1, 4, 2, 5])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_169(xs):
    #Predicts Class 4
    function_dict = np.array([[-0.899999976, 1.0, 2.0, 291.0], [-0.335500002, 3.0, 4.0, 157.0], [-0.0820469931, 0.0, 0.0, 0.0], [-0.0512216352, 0.0, 0.0, 0.0], [0.0335500017, 5.0, 6.0, 30.0], [0.124616876, 0.0, 0.0, 0.0], [-0.0376964472, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 5, 6, 2])
    branch_indices = np.array([0, 1, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_170(xs):
    #Predicts Class 0
    function_dict = np.array([[-0.75849998, 1.0, 2.0, 450.0], [-0.0628712103, 0.0, 0.0, 0.0], [0.0561601333, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_171(xs):
    #Predicts Class 1
    function_dict = np.array([[0.113499999, 1.0, 2.0, 525.0], [-0.0521629862, 0.0, 0.0, 0.0], [0.0683423355, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_172(xs):
    #Predicts Class 2
    function_dict = np.array([[-0.403500021, 1.0, 2.0, 298.0], [0.0600573421, 0.0, 0.0, 0.0], [-0.0508954115, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_173(xs):
    #Predicts Class 3
    function_dict = np.array([[0.478500009, 1.0, 2.0, 557.0], [-0.979499996, 3.0, 4.0, 179.0], [-0.101797245, 0.0, 0.0, 0.0], [0.911499977, 5.0, 6.0, 133.0], [-0.984500051, 7.0, 8.0, 19.0], [0.118967295, 0.0, 0.0, 0.0], [-0.0441919528, 0.0, 0.0, 0.0], [-0.121330343, 0.0, 0.0, 0.0], [0.0429089889, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([5, 6, 7, 8, 2])
    branch_indices = np.array([0, 1, 3, 4])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_174(xs):
    #Predicts Class 4
    function_dict = np.array([[-0.0985499993, 1.0, 2.0, 551.0], [-0.0225000009, 3.0, 4.0, 372.0], [0.819499969, 5.0, 6.0, 134.0], [-0.0140987551, 0.0, 0.0, 0.0], [0.127442181, 0.0, 0.0, 0.0], [0.625999987, 7.0, 8.0, 79.0], [0.059299998, 9.0, 10.0, 145.0], [-0.164825901, 0.0, 0.0, 0.0], [0.0186190549, 0.0, 0.0, 0.0], [0.113592386, 0.0, 0.0, 0.0], [-0.0439939052, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_175(xs):
    #Predicts Class 0
    function_dict = np.array([[0.546499968, 1.0, 2.0, 104.0], [-0.058658842, 0.0, 0.0, 0.0], [0.054872673, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_176(xs):
    #Predicts Class 1
    function_dict = np.array([[-0.416500002, 1.0, 2.0, 284.0], [-0.0653547049, 0.0, 0.0, 0.0], [0.0444165915, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_177(xs):
    #Predicts Class 2
    function_dict = np.array([[-0.109499998, 1.0, 2.0, 504.0], [-0.0517684072, 0.0, 0.0, 0.0], [0.0561563261, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_178(xs):
    #Predicts Class 3
    function_dict = np.array([[-0.113499999, 1.0, 2.0, 50.0], [0.29549998, 3.0, 4.0, 187.0], [0.172499999, 5.0, 6.0, 106.0], [-0.119634986, 0.0, 0.0, 0.0], [0.0190658327, 0.0, 0.0, 0.0], [-0.977499962, 7.0, 8.0, 123.0], [0.347499996, 9.0, 10.0, 112.0], [0.0491995551, 0.0, 0.0, 0.0], [-0.103751339, 0.0, 0.0, 0.0], [0.1477108, 0.0, 0.0, 0.0], [0.0168345999, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 7, 8, 9, 10])
    branch_indices = np.array([0, 1, 2, 5, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_179(xs):
    #Predicts Class 4
    function_dict = np.array([[0.0190000013, 1.0, 2.0, 186.0], [-0.0486500002, 3.0, 4.0, 162.0], [0.0449500009, 5.0, 6.0, 187.0], [-0.667999983, 7.0, 8.0, 182.0], [-0.0390099846, 0.0, 0.0, 0.0], [0.0433335938, 0.0, 0.0, 0.0], [-0.115933768, 0.0, 0.0, 0.0], [-0.00968006998, 0.0, 0.0, 0.0], [0.137223274, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([7, 8, 4, 5, 6])
    branch_indices = np.array([0, 1, 3, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_180(xs):
    #Predicts Class 0
    function_dict = np.array([[-0.374500006, 1.0, 2.0, 124.0], [-0.0551735759, 0.0, 0.0, 0.0], [0.05066007, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_181(xs):
    #Predicts Class 1
    function_dict = np.array([[-0.445500016, 1.0, 2.0, 141.0], [-0.0575261377, 0.0, 0.0, 0.0], [0.052357696, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_182(xs):
    #Predicts Class 2
    function_dict = np.array([[0.304499984, 1.0, 2.0, 23.0], [-0.0466253459, 0.0, 0.0, 0.0], [0.0559132621, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_183(xs):
    #Predicts Class 3
    function_dict = np.array([[-0.374000013, 1.0, 2.0, 77.0], [-0.249500006, 3.0, 4.0, 145.0], [-0.00196500006, 5.0, 6.0, 114.0], [-0.0153109599, 0.0, 0.0, 0.0], [0.108224168, 0.0, 0.0, 0.0], [0.0488069132, 0.0, 0.0, 0.0], [-0.975499988, 7.0, 8.0, 58.0], [-0.151046485, 0.0, 0.0, 0.0], [-0.00401603477, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 4, 5, 7, 8])
    branch_indices = np.array([0, 1, 2, 6])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_184(xs):
    #Predicts Class 4
    function_dict = np.array([[0.037250001, 1.0, 2.0, 374.0], [-0.440500021, 3.0, 4.0, 145.0], [-0.993499994, 5.0, 6.0, 179.0], [0.0493107997, 0.0, 0.0, 0.0], [0.0473000035, 7.0, 8.0, 294.0], [-0.0497879796, 0.0, 0.0, 0.0], [0.124224193, 0.0, 0.0, 0.0], [-0.145289153, 0.0, 0.0, 0.0], [0.0228824988, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 5, 6])
    branch_indices = np.array([0, 1, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_185(xs):
    #Predicts Class 0
    function_dict = np.array([[-0.473500013, 1.0, 2.0, 62.0], [0.0463530272, 0.0, 0.0, 0.0], [-0.0494785905, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_186(xs):
    #Predicts Class 1
    function_dict = np.array([[0.695500016, 1.0, 2.0, 76.0], [-0.0485051163, 0.0, 0.0, 0.0], [0.0606728345, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_187(xs):
    #Predicts Class 2
    function_dict = np.array([[-0.32249999, 1.0, 2.0, 89.0], [-0.0481399521, 0.0, 0.0, 0.0], [0.0556875616, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 2])
    branch_indices = np.array([0])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_188(xs):
    #Predicts Class 3
    function_dict = np.array([[-0.342500001, 1.0, 2.0, 556.0], [-0.0828337967, 0.0, 0.0, 0.0], [0.467500001, 3.0, 4.0, 556.0], [-0.0770500004, 5.0, 6.0, 53.0], [-0.0885413289, 0.0, 0.0, 0.0], [-0.010711425, 0.0, 0.0, 0.0], [0.134702832, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([1, 5, 6, 4])
    branch_indices = np.array([0, 2, 3])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
            yes_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row < thres_pointed_at).reshape(-1)]
            pointers[yes_indices] = yes_pointed_at[yes_indices]
            no_indices = pointed_at_branch_rowindices[np.argwhere(feature_val_for_each_row >= thres_pointed_at).reshape(-1)]
            pointers[no_indices] = no_pointed_at[no_indices]
        if pointed_at_leaf_rowindices.shape[0]>0:
            pointers[pointed_at_leaf_rowindices] = -1 * np.ones(pointed_at_leaf_rowindices.shape[0])
        if (dones==1).all() or (pointers == -1).all():
            break
    return values
def booster_189(xs):
    #Predicts Class 4
    function_dict = np.array([[0.845499992, 1.0, 2.0, 12.0], [-0.980499983, 3.0, 4.0, 139.0], [-0.993499994, 5.0, 6.0, 179.0], [-0.131689727, 0.0, 0.0, 0.0], [-0.984500051, 7.0, 8.0, 19.0], [-0.0237131212, 0.0, 0.0, 0.0], [0.116495967, 0.0, 0.0, 0.0], [0.12108361, 0.0, 0.0, 0.0], [-0.0668580011, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]])
    leaf_indices = np.array([3, 7, 8, 5, 6])
    branch_indices = np.array([0, 1, 4, 2])
    pointers = np.zeros(xs.shape[0]).astype('int')
    values = np.zeros(xs.shape[0])
    dones = np.zeros(xs.shape[0])
    while True:
        pointed_at_leaf_rowindices = np.argwhere(np.isin(pointers,leaf_indices).reshape(-1)).reshape(-1)
        if pointed_at_leaf_rowindices.shape[0]>0:
            leaf_pointers = pointers[pointed_at_leaf_rowindices]
            dones[pointed_at_leaf_rowindices] = np.ones(pointed_at_leaf_rowindices.shape[0])
            values[pointed_at_leaf_rowindices] = function_dict[leaf_pointers,0].reshape(-1)
        pointed_at_branch_rowindices = np.argwhere(np.isin(pointers, branch_indices)).reshape(-1)
        if pointed_at_branch_rowindices.shape[0]>0:
            features_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 3].reshape(-1).astype('int')
            branching_rows = xs[pointed_at_branch_rowindices, :].reshape(pointed_at_branch_rowindices.shape[0],-1)
            no_pointed_at = function_dict[:-1][pointers, 2].reshape(-1).astype('int')
            yes_pointed_at = function_dict[:-1][pointers, 1].reshape(-1).astype('int')
            thres_pointed_at = function_dict[:-1][pointers[pointed_at_branch_rowindices], 0].reshape(-1).astype('float64')
            feature_val_for_each_row = branching_rows[np.arange(len(branching_rows)),features_pointed_at].reshape(-1).astype('float64')
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
    for booster_index in range(0,190,5):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def logit_class_1(xs):
    sum_of_leaf_values = np.zeros(xs.shape[0])
    for booster_index in range(1,190,5):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def logit_class_2(xs):
    sum_of_leaf_values = np.zeros(xs.shape[0])
    for booster_index in range(2,190,5):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def logit_class_3(xs):
    sum_of_leaf_values = np.zeros(xs.shape[0])
    for booster_index in range(3,190,5):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def logit_class_4(xs):
    sum_of_leaf_values = np.zeros(xs.shape[0])
    for booster_index in range(4,190,5):
            sum_of_leaf_values += eval('booster_' + str(booster_index) + '(xs)')
    return sum_of_leaf_values


def classify(rows, return_probabilities=False):
    logits = []
    logits.append(logit_class_0)
    logits.append(logit_class_1)
    logits.append(logit_class_2)
    logits.append(logit_class_3)
    logits.append(logit_class_4)
    o = np.array([logits[class_index](rows) for class_index in range(5)]).T
    if not return_probabilities:
        return np.argmax(o,axis=1)
    else:
        exps = np.exp(o)
        Z = np.sum(exps, axis=1).reshape(-1, 1)
        return exps/Z





def Validate(cleanarr):
    #validation
    outputs = classify(cleanarr[:, :-1])
    #metrics
    count, correct_count = 0, 0
    numeachclass = {}
    for k, o in enumerate(outputs):
        if int(o) == int(float(cleanarr[k, -1])):
            correct_count += 1
        if int(float(cleanarr[k, -1])) in numeachclass.keys():
            numeachclass[int(float(cleanarr[k, -1]))] += 1
        else:
            numeachclass[int(float(cleanarr[k, -1]))] = 1
        count += 1
    return count, correct_count, numeachclass, outputs


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
        count, correct_count, numeachclass, preds = Validate(cleanarr)
        #Correct Labels
        true_labels = cleanarr[:, -1]

        #Report Metrics
        model_cap=6
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
