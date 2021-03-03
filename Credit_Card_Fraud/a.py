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
# Invocation: btc creditcard.csv -f NN -O 1 --yes
# Total compiler execution time: 0:41:15.58. Finished on: Mar-03-2021 05:06:17.
# This source code requires Python 3.
#
"""
Classifier Type:                    Neural Network
System Type:                         Binary classifier
Training/Validation Split:           50:50%
Best-guess accuracy:                 99.82%
Training accuracy:                   1.44% (2064/142403 correct)
Validation accuracy:                 0.80% (1152/142404 correct)
Overall Model accuracy:              1.12% (3216/284807 correct)
Overall Improvement over best guess: -98.70% (of possible 0.18%)
Model capacity (MEC):                1 bits
Generalization ratio:                37.46 bits/bit
Model efficiency:                    -98.69%/parameter
System behavior
True Negatives:                      0.96% (2724/284807)
True Positives:                      0.17% (492/284807)
False Negatives:                     0.00% (0/284807)
False Positives:                     98.87% (281591/284807)
True Pos. Rate/Sensitivity/Recall:   1.00
True Neg. Rate/Specificity:          0.01
Precision:                           0.00
F-1 Measure:                         0.00
False Negative Rate/Miss Rate:       0.00
Critical Success Index:              0.00
Confusion Matrix:
 [0.96% 98.87%]
 [0.00% 0.17%]
Warning: The prediction model is less accurate overall than the best guess.
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

# Imports -- external
try:
    import numpy as np # For numpy see: http://numpy.org
    from numpy import array
except:
    print("This predictor requires the Numpy library. For installation instructions please refer to: http://numpy.org")

# Magic constants follow
# I/O buffer for clean. Reduce this constant for low memory devices. 
IOBUF = 100000000

# Ugly workaround for large classifiers
sys.setrecursionlimit(1000000)

# Training file given to compiler
TRAINFILE = "creditcard.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 11
num_attr_before_transform = 30
n_classes = 2

list_of_cols_to_normalize = [0]

transform_true = True

def column_norm(column,mappings):
    listy = []
    for i,val in enumerate(column.reshape(-1)):
        if not (val in mappings):
            mappings[val] = int(max(mappings.values())) + 1
        listy.append(mappings[val])
    return np.array(listy)

def Normalize(data_arr):
    if list_of_cols_to_normalize:
        for i,mapping in zip(list_of_cols_to_normalize, mappings):
            if i >= data_arr.shape[1]:
                break
            col = data_arr[:, i]
            normcol = column_norm(col,mapping)
            data_arr[:, i] = normcol
        return data_arr
    else:
        return data_arr

def transform(X):
    mean = None
    components = None
    whiten = None
    explained_variance = None
    if (transform_true):
        mean = np.array([45247.71079260971, 0.004100310956093994, 0.001788483597460661, 0.0021182189821326762, -0.0003979971785323599, 0.0021453389375219473, -0.00125209721023333, 0.0016904947089810799, 0.00413332014029374, 4.258012506781533e-05, -0.0012089510726554366, -6.4200911248103905e-06, 0.0041796202208396385, -0.0004741902645041533, -0.002289093328727914, -0.00040540684908417935, -6.022292894374407e-05, -0.0004726861157545393, -0.00020496679730063502, -0.0020736214445004853, 0.0004441211743153723, 0.0015165435106965614, -0.00038923093520300413, 0.0025898203733420583, 0.0006338842493684018, -0.0007479439551485727, 0.001626644459013125, 0.0005763476884823245, 0.00011099775311737537, 88.01273301828398])
        components = np.array([array([-9.99999989e-01, -8.61565289e-06,  2.22633669e-07,  2.43320169e-05,
        5.76828181e-06, -9.24573338e-06,  3.12068259e-06, -3.93790789e-06,
        1.66907250e-06,  1.20039452e-07, -1.20143381e-06,  9.65963902e-06,
       -4.72668744e-06,  2.26473628e-06,  3.47832325e-06,  6.32072155e-06,
       -3.12388941e-07,  2.38859731e-06, -2.90654782e-06, -9.26108245e-07,
        1.51399694e-06, -1.17709403e-06, -3.86532550e-06, -1.27653517e-06,
        4.52553416e-07,  4.67398545e-06,  7.31105347e-07,  9.19957957e-08,
        1.57203130e-07,  1.47772591e-04]), array([ 1.47773239e-04, -1.74729648e-03, -3.43876584e-03, -1.31684880e-03,
        5.65303447e-04, -2.15908785e-03,  1.18086299e-03,  2.05500220e-03,
       -5.02024323e-04, -1.99011043e-04, -4.41929731e-04, -1.19063027e-05,
       -4.19587738e-05,  2.72143324e-05,  1.20938751e-04, -9.40273408e-06,
        2.07932275e-06,  2.00652566e-05,  1.23041211e-04, -1.75754812e-04,
        9.81671752e-04,  2.85400070e-04, -1.74002020e-04, -2.31646417e-04,
        1.56707800e-05, -9.48334589e-05,  5.46433013e-06,  7.55721409e-05,
       -7.29806081e-07,  9.99985536e-01]), array([ 6.59936956e-06, -9.75414312e-01,  1.83411984e-01, -5.71505737e-03,
       -4.32112074e-02,  9.95262785e-02, -3.49310909e-02, -2.02101937e-02,
        1.65117396e-02,  1.74586765e-03,  1.17251041e-02, -1.77971144e-02,
        8.58865567e-03, -4.89250720e-03, -5.73947305e-03, -9.94182100e-03,
        1.15882133e-03,  2.51654670e-03,  6.21005474e-03,  2.90269636e-03,
       -1.48659326e-02,  2.87235704e-04,  8.42640723e-03,  1.27468280e-02,
       -1.50580638e-03, -4.19350255e-03, -4.98871835e-04, -1.50582535e-03,
        1.21201414e-03, -7.26002596e-04]), array([-3.81968861e-06,  1.01463327e-01,  8.21012471e-01, -3.14944062e-01,
        1.60908223e-01, -3.53052841e-01,  1.45866939e-01,  2.02156434e-01,
       -2.52898331e-02, -6.40237781e-03, -1.62488905e-02,  2.23498201e-02,
       -1.15649217e-02,  8.49304404e-03,  1.06205237e-02,  1.19804643e-02,
        6.58085199e-03,  1.20355887e-02, -1.53592522e-04, -6.12976714e-03,
        2.77646352e-02,  8.49742143e-03, -1.33001779e-02, -2.17533976e-02,
        3.92902397e-04,  1.13968032e-04,  8.12584956e-04,  6.35669963e-03,
       -1.02045405e-02,  1.08331511e-03]), array([-2.49791935e-06, -2.17436030e-02, -9.80161178e-02, -2.24142654e-01,
        9.10452858e-01,  3.19236558e-01, -3.79294826e-02, -7.73788448e-02,
        2.44183055e-02,  2.69246386e-03,  8.92526689e-03,  1.61110366e-02,
       -9.12856085e-03,  5.01887892e-03,  5.69228063e-03,  8.20869203e-03,
       -4.33635679e-03,  1.71403386e-03, -2.82254712e-03,  2.14188639e-03,
       -3.33350892e-03, -9.54071887e-04, -1.52609254e-03,  1.21410220e-02,
       -9.77310444e-04,  6.95831005e-03,  1.07448899e-03, -2.46199697e-03,
        3.72329353e-03, -2.68383862e-04]), array([-1.90751641e-05,  2.01689621e-04, -1.59560813e-01, -8.55015066e-01,
       -2.90656725e-01,  7.77370897e-02, -1.39571902e-02, -3.21283701e-01,
        9.69226734e-02,  2.78195749e-02,  1.76361027e-02,  1.43881842e-01,
       -6.32397711e-02,  3.15135282e-02,  4.31633562e-02,  8.00256172e-02,
       -3.38205638e-04,  3.59354076e-02, -3.17557659e-02, -5.98769591e-03,
       -1.38822596e-02, -1.66829314e-02, -2.82568670e-02,  4.75335840e-03,
        1.20937670e-03,  3.78717869e-02,  5.31841139e-03, -3.01532957e-03,
        2.31282298e-03, -5.89197127e-04]), array([ 1.77152690e-06, -3.39300774e-02, -6.73674231e-02, -3.34892038e-02,
        1.12955407e-01, -4.87368104e-01, -8.57545009e-01, -7.71844958e-02,
       -1.49943570e-02,  7.77041232e-03,  1.65538943e-02, -1.71560564e-02,
        1.35202549e-02, -7.28815876e-04, -5.99549866e-03, -9.49524080e-03,
        1.78148822e-03,  3.58256300e-03,  4.43410264e-03,  3.29862529e-03,
       -1.18518863e-02, -5.85967236e-03,  9.40632707e-03,  8.20781766e-03,
        1.64737550e-04, -3.44676256e-03,  9.29563670e-05, -2.36476467e-03,
        1.89452819e-03, -2.60911158e-04]), array([ 1.10076876e-05, -7.94819530e-02, -1.18822762e-01,  1.24096196e-01,
        1.73298754e-01, -5.56358680e-01,  4.08130986e-01, -6.19869421e-01,
        1.65933478e-01,  4.26350038e-02,  1.07303153e-01, -1.24899848e-01,
        5.58556177e-02, -2.77515498e-02, -5.29140760e-02, -6.48161993e-02,
        5.28388464e-03, -4.95762167e-03,  2.66114505e-02,  1.62433111e-02,
       -5.82562267e-02, -1.27576088e-03,  3.12911187e-02,  2.12622931e-02,
       -2.02952146e-03, -2.34358210e-02, -3.08305576e-03, -4.33758157e-04,
       -4.02087623e-03, -6.79327833e-04]), array([-6.52246710e-07, -7.62996899e-03,  2.04608493e-02, -1.56189580e-02,
        1.36260404e-03,  6.85493515e-03,  3.46077224e-02, -2.33370240e-01,
       -9.62918112e-01,  6.13881425e-02,  9.11451293e-02,  1.54296729e-02,
       -3.11729417e-02,  2.06981309e-02, -2.37058266e-02,  1.82132591e-02,
       -9.67852700e-03, -3.56100442e-03, -3.81080561e-03,  4.31432242e-03,
       -1.34748068e-02,  3.55279358e-02, -1.00431780e-02, -5.94812744e-03,
       -8.20751365e-04,  4.97956276e-03,  1.89170295e-03,  1.55648477e-03,
        8.95257295e-04,  6.12792029e-05]), array([ 4.32633568e-08, -2.04730168e-03, -1.61437821e-03,  5.43510777e-03,
        4.20551977e-03,  2.67167844e-04,  8.37135964e-04,  2.44251881e-02,
        1.94553787e-02,  9.35794122e-01, -3.47808297e-01, -3.91914539e-02,
        1.12285069e-02, -9.11484224e-03, -4.48237154e-03, -1.39474316e-02,
        3.23999922e-03, -1.00459713e-02,  1.06742440e-02, -1.42619253e-03,
       -3.74032278e-03,  2.73848144e-04, -6.07261076e-03,  6.89033670e-03,
       -6.62869727e-04,  1.46987848e-03, -5.81303184e-04,  2.52061365e-04,
        5.35692864e-04, -9.09923290e-06]), array([ 1.62677571e-06,  1.18345970e-02,  7.20409646e-02,  3.57870735e-02,
       -5.33652830e-03,  3.77016207e-02, -2.68492460e-02, -2.41256326e-01,
       -4.72027530e-02, -3.30732644e-01, -9.02762708e-01, -4.39088146e-02,
       -7.34228835e-04,  2.36858616e-03, -4.48545349e-02, -1.37330537e-02,
        8.64060686e-04,  3.87322252e-04,  2.62841309e-02, -6.20667054e-03,
        2.46945631e-02,  2.09232300e-02, -1.74965159e-02, -1.13770755e-02,
        1.92731927e-03, -8.06810243e-03, -1.01474272e-03,  1.73036832e-03,
        3.90532854e-04,  4.02503087e-04])])
        whiten = False
        explained_variance = np.array([646519409.7655505, 65703.60206085254, 3.6397645611561207, 2.366660031716787, 1.9947470518532424, 1.85661597286119, 1.8036544043765586, 1.583166817727125, 1.319033892147702, 1.1983017051409022, 1.167261574862545])
        X = X - mean

    X_transformed = np.dot(X, components.T)
    if whiten:
        X_transformed /= np.sqrt(explained_variance)
    return X_transformed

# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target=""
important_idxs=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[], trim=False):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target=""
    important_idxs=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
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

def single_classify(row, return_probabilities=False):
    #inits
    x = row
    o = [0] * num_output_logits


    #Nueron Equations
    h_0 = max((((-2797.6982 * float(x[0]))+ (-165.17856 * float(x[1]))+ (-0.29879606 * float(x[2]))+ (-0.4480171 * float(x[3]))+ (-0.31042856 * float(x[4]))+ (-0.3270369 * float(x[5]))+ (0.06304116 * float(x[6]))+ (-1.1889466 * float(x[7]))+ (-0.17902394 * float(x[8]))+ (-0.35620245 * float(x[9]))+ (-0.7551529 * float(x[10]))) + -1.7606287), 0)
    h_1 = max((((69.53225 * float(x[0]))+ (-31.679314 * float(x[1]))+ (-1.0545781 * float(x[2]))+ (-0.30978638 * float(x[3]))+ (0.9353187 * float(x[4]))+ (-1.4680208 * float(x[5]))+ (-0.8635088 * float(x[6]))+ (0.9590522 * float(x[7]))+ (0.9702756 * float(x[8]))+ (-0.022806121 * float(x[9]))+ (-0.056109577 * float(x[10]))) + -1.4315711), 0)
    h_2 = max((((-0.12288545 * float(x[0]))+ (-2.048171 * float(x[1]))+ (-5.582683 * float(x[2]))+ (-5.473965 * float(x[3]))+ (-5.49833 * float(x[4]))+ (-5.0820074 * float(x[5]))+ (-4.4425564 * float(x[6]))+ (-4.283728 * float(x[7]))+ (-4.179075 * float(x[8]))+ (4.6308546 * float(x[9]))+ (-5.90226 * float(x[10]))) + -0.2095006), 0)
    o[0] = (8.8140434e-05 * h_0)+ (6.2481064e-05 * h_1)+ (-0.4260788 * h_2) + -0.56212574



    #Output Decision Rule
    if num_output_logits==1:
        if return_probabilities:
            exp_o = 1./(1. + np.exp(-o[0]))
            return np.array([1.-exp_o, exp_o])
        else:
            return o[0]>=0
    else:
        if return_probabilities:
            exps = np.exp(o)
            Z = sum(exps).reshape(-1, 1)
            return exps/Z
        else:
            return argmax(o)


def classify(arr, transform_true=False, return_probabilities=False):
    #apply transformation if necessary
    if transform_true:
        arr = transform(arr)
    #init
    w_h = np.array([[-2797.6982421875, -165.17855834960938, -0.29879605770111084, -0.44801709055900574, -0.31042855978012085, -0.32703688740730286, 0.06304115802049637, -1.1889466047286987, -0.17902393639087677, -0.35620245337486267, -0.7551528811454773], [69.5322494506836, -31.67931365966797, -1.0545780658721924, -0.3097863793373108, 0.9353187084197998, -1.4680207967758179, -0.8635088205337524, 0.9590522050857544, 0.9702755808830261, -0.02280612103641033, -0.05610957741737366], [-0.12288545072078705, -2.048171043395996, -5.582683086395264, -5.473965167999268, -5.498330116271973, -5.08200740814209, -4.442556381225586, -4.283728122711182, -4.179074764251709, 4.630854606628418, -5.902259826660156]])
    b_h = np.array([-1.7606287002563477, -1.431571125984192, -0.20950059592723846])
    w_o = np.array([[8.814043394522741e-05, 6.248106365092099e-05, -0.42607879638671875]])
    b_o = np.array(-0.5621257424354553)

    #Hidden Layer
    h = np.dot(arr, w_h.T) + b_h
    
    relu = np.maximum(h, np.zeros_like(h))


    #Output
    out = np.dot(relu, w_o.T) + b_o
    if num_output_logits == 1:
        if return_probabilities:
            exp_o = 1./(1. + np.exp(-out))
            return np.concatenate((1.-exp_o, exp_o), axis=1)
        else:
            return (out >= 0).astype('int').reshape(-1)
    else:
        if return_probabilities:
            exps = np.exp(out)
            Z = np.sum(exps, axis=1).reshape(-1, 1)
            return exps/Z
        else:
            return (np.argmax(out, axis=1)).reshape(-1)



def Predict(arr,headerless,csvfile, get_key, classmapping):
    with open(csvfile, 'r', encoding='utf-8') as csvinput:
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


def Validate(cleanarr):
    if n_classes == 2:
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


    else:
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
    


# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile',action='store_true',help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    parser.add_argument('-json', action="store_true", default=False, help="report measurements as json")
    parser.add_argument('-trim', action="store_true", help="If true, the prediction will not output ignored columns.")
    args = parser.parse_args()
    faulthandler.enable()

    if args.validate:
        args.trim = True


    #clean if not already clean
    if not args.cleanfile:
        cleanfile = tempfile.NamedTemporaryFile().name
        preprocessedfile = tempfile.NamedTemporaryFile().name
        output = preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate), trim=args.trim)
        get_key, classmapping = clean(preprocessedfile if output!=-1 else args.csvfile, cleanfile, -1, args.headerless, (not args.validate), trim=args.trim)
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x, y: x
        classmapping = {}
        output = None


    #load file
    cleanarr = np.loadtxt(cleanfile, delimiter=',', dtype='float64')
    if not args.trim and ignorecolumns != []:
        cleanarr = cleanarr[:, important_idxs]


    #Normalize
    cleanarr = Normalize(cleanarr)


    #Transform
    if transform_true:
        if args.validate:
            trans = transform(cleanarr[:, :-1])
            cleanarr = np.concatenate((trans, cleanarr[:, -1].reshape(-1, 1)), axis = 1)
        else:
            cleanarr = transform(cleanarr)


    #Predict
    if not args.validate:
        Predict(cleanarr, args.headerless, preprocessedfile if output!=-1 else args.csvfile, get_key, classmapping)


    #Validate
    else:
        classifier_type = 'NN'
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds = Validate(cleanarr)
        else:
            count, correct_count, numeachclass, preds = Validate(cleanarr)
        #Correct Labels
        true_labels = cleanarr[:, -1]


        #Report Metrics
        model_cap = 1
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






