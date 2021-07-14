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
# Invocation: btc titanic_test.csv
# Total compiler execution time: 0:00:15.07. Finished on: Feb-23-2021 22:24:06.
# This source code requires Python 3.
#
"""
Classifier Type:                    Decision Tree
System Type:                         3-way classifier
Best-guess accuracy:                 64.59%
Overall Model accuracy:              100.00% (418/418 correct)
Overall Improvement over best guess: 35.41% (of possible 35.41%)
Model capacity (MEC):                213 bits
Generalization ratio:                2.46 bits/bit
Model efficiency:                    0.16%/parameter
Confusion Matrix:
 [11.00% 0.00% 0.00%]
 [0.00% 64.59% 0.00%]
 [0.00% 0.00% 24.40%]
Generalization index:           1.00
Overfitting:                         No
Note: Unable to split dataset. The predictor was trained and evaluated on the same data.
Note: Labels have been remapped to 'Q'=0, 'S'=1, 'C'=2.
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
TRAINFILE = "titanic_test.csv"


#Number of attributes
num_attr = 10
n_classes = 3
transform_true = False

# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=[]
target=""
important_idxs=[0,1,2,3,4,5,6,7,8,9]

def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[], trim=False):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=[]
    target=""
    important_idxs=[0,1,2,3,4,5,6,7,8,9]
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
    clean.mapping={'Q': 0, 'S': 1, 'C': 2}

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


# Calculate energy

# Imports -- external
import numpy as np # For numpy see: http://numpy.org
from numpy import array
energy_thresholds = array([1403061367.4479, 1455119232.3458, 1470084905.3479, 1489828377.4875, 1502707156.84585, 1524616470.8895998, 1583919670.7020998, 1611221454.3625, 1793090654.0996, 1812278937.94585, 1818412266.45415, 1829953896.6229, 1922510504.1646, 1956001522.9292, 1961917058.5124998, 1970916628.2582998, 1983922410.05415, 2012433156.63335, 2033595394.77085, 2080287052.6104, 2089485309.1479, 2131921499.6125, 2140703512.0604, 2154528989.5604, 2168153584.5625, 2234632462.2, 2267985943.39585, 2288385242.49585, 2293741920.1896, 2306798445.02085, 2376154564.7, 2410619582.45, 2476657552.9771, 2482380345.42295, 2492138842.8917, 2510794130.6146, 2538804828.3374996, 2553700838.6021, 2568202706.1021, 2586224279.60415, 2605352436.65625, 2630372429.1271, 2664303688.6146, 2710294296.625, 2726173793.7625, 2729573854.0125, 2740729655.375, 2768052893.43125, 2782728452.3021, 2806594031.2479, 2816024164.2354, 2835751430.7625, 2847182395.4646, 2852951955.8896, 2866927207.7146, 2869465327.4396, 2892584672.8646, 2906649318.1146, 2916036761.375, 2919695537.4979, 2937977509.6479, 2957853434.78335, 2972186832.25835, 3087544535.8875, 3093218434.0521, 3103556985.98335, 3120685367.8042, 3132117605.80835, 3138233795.375, 3185329393.21875, 3185800864.88125, 3241057427.45625, 3280725808.1479, 3319843942.95205, 3349034799.02705, 3365353138.32915, 3378520502.0896, 3381886750.6146, 3390050686.39585, 3404375561.62085, 3426047988.12085, 3469090466.62085, 3575691214.63545, 3585510696.58335, 3589116414.81875, 3609852997.2396, 3626470830.9396, 3743984832.7354, 3745557933.03955, 3759011579.15, 3797289624.83335, 3827581427.7, 3903482419.36665, 3924542835.86665, 3951354102.8604, 3969450063.5417, 4012771201.375, 4043520550.125, 4054104623.625, 4060478553.15, 4075333379.65, 4083326398.8229, 4133515763.825, 4136857308.8104, 4149357830.35415, 4155974968.60415, 4163799850.625, 4176826674.4979, 4191733300.6229, 4259955693.125, 4262626384.875, 4316076773.6125, 4331366087.7375, 4335202528.96875, 4341265183.03125, 4371443105.979151, 4398664227.36665, 4428736197.2, 4490858133.4, 4521795386.375, 4551422568.89165, 4567446805.14165, 4592158811.3125, 4600762791.6125, 4621563005.4771, 4624619855.4771, 4632148572.46875, 4665769175.46875, 4702701239.1229, 4712934801.28125, 4737398460.3896, 4764062984.8271, 4860529809.375, 4891324357.73125, 4899415877.2271, 4907350439.87085, 4927894886.358351, 4928653092.262501, 4953339719.0104, 4968720113.5708, 4999438002.170799, 5022123580.0729, 5026661082.225, 5033451613.2, 5055252150.925, 5060197724.9896, 5066364660.077101, 5197194138.06875, 5203699665.875, 5218457596.2896, 5223717859.989599, 5276640866.725, 5284603563.487499, 5311891968.3604, 5326345000.50415, 5334236105.70625, 5345423481.0625, 5380423488.5229, 5413846579.8979, 5422253422.15, 5509643669.20415, 5520146445.6375, 5595377651.7896, 5603256634.4146, 5610175086.35, 5629299930.487499, 5649181827.0021, 5669298230.61875, 5703309496.8604, 5744076215.135401, 5781604493.043751, 5801652346.47085, 5947655599.275, 5971599061.0854, 6002468503.5021, 6006194567.5021, 6088108174.15, 6108092908.5771, 6168391072.75, 6264797300.175, 6306374783.39165, 6332264779.69165, 6360410896.5229, 6387832852.0104, 6407586810.5396, 6444107578.3146, 6720605615.25, 6762534032.9375, 6822443605.8896, 6854773256.6646, 6951103788.525, 7064380588.9646, 7098149140.225, 7127549533.85, 7393990922.6396, 7421915697.51665, 7438238372.2625, 7466568883.6521, 7660820944.25, 7667415352.2, 7681069972.075, 7804835887.60205, 7874765278.99585, 7899061613.74585, 7934736465.275, 7974815518.0, 8243262072.9375, 8297989367.1875, 8374430777.45, 8420331775.85, 8991300919.40415, 9282834828.81665, 9433063586.1875])
labels = array([2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
def eqenergy(rows):
    return np.sum(rows, axis=1, dtype=np.float128)
def classify(rows):
    energys = eqenergy(rows)

    def thresh_search(input_energys):
        numers = np.searchsorted(energy_thresholds, input_energys, side='left')-1
        indys = np.argwhere(np.logical_and(numers<=len(energy_thresholds), numers>=0)).reshape(-1)
        defaultindys = np.argwhere(np.logical_not(np.logical_and(numers<=len(energy_thresholds), numers>=0))).reshape(-1)
        outputs = np.zeros(input_energys.shape[0])
        outputs[indys] = labels[numers[indys]]
        if list(defaultindys):
            outputs[defaultindys] = 1
        return outputs
    return thresh_search(energys)

numthresholds = 213

# Main method
model_cap = numthresholds


def Validate(file):
    #Load Array
    cleanarr = np.loadtxt(file, delimiter=',', dtype='float64')


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
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, outputs, cleanarr[:, -1]


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
        return count, correct_count, numeachclass, outputs, cleanarr[:, -1]


#Predict on unlabeled data
def Predict(file, get_key, headerless, preprocessedfile, classmapping, trim=False):
    cleanarr = np.loadtxt(file, delimiter=',', dtype='float64')
    cleanarr = cleanarr.reshape(cleanarr.shape[0], -1)
    if not trim and ignorecolumns != []:
        cleanarr = cleanarr[:, important_idxs]
    with open(preprocessedfile, 'r', encoding='utf-8') as csvinput:
        dirtyreader = csv.reader(csvinput)
        if not headerless:
            header = next(dirtyreader, None)
        #print original header
        if (not headerless):
            print(','.join(header + ["Prediction"]))

        outputs = classify(cleanarr)

        for k, row in enumerate(dirtyreader):
            if k >= outputs.shape[0]:
                continue
            print(str(','.join(str(j) for j in (['"' + i + '"' if ',' in i else i for i in row]))) + str(',' if len(important_idxs) != 1 else '') + str(get_key(int(outputs[k]), classmapping)))
                




#Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile', action='store_true', help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    parser.add_argument('-json', action="store_true", default=False, help="report measurements as json")
    parser.add_argument('-trim', action="store_true", default=False, help="Output predictor file with only the selected important attributes (to be used with predictors built with -rank)")
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

    #Predict or Validate?
    if not args.validate:
        Predict(cleanfile, get_key, args.headerless, preprocessedfile if output!=-1 else args.csvfile, classmapping, trim=args.trim)


    else:
        classifier_type = 'DT'
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds, true_labels = Validate(cleanfile)
        else:
            count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)


        #validation report
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

        def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
            #check for numpy/scipy is imported
            try:
                from scipy.sparse import coo_matrix #required for multiclass metrics
            except:
                if not args.json:
                    print("Note: If you install scipy (https://www.scipy.org) this predictor generates a confusion matrix")
                else:
                    print(json.dumps(json_dict))
                sys.exit()

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
            if labels is None:
                labels = np.array(list(set(list(y_true.astype('int')))))
            else:
                labels = np.asarray(labels)
                if np.all([l not in y_true for l in labels]):
                    raise ValueError("At least one label specified must be in y_true")


            if sample_weight is None:
                sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
            else:
                sample_weight = np.asarray(sample_weight)
            if y_true.shape[0]!=y_pred.shape[0]:
                raise ValueError("y_true and y_pred must be of the same length")

            if normalize not in ['true', 'pred', 'all', None]:
                raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")


            n_labels = labels.size
            label_to_ind = {y: x for x, y in enumerate(labels)}
            # convert yt, yp into index
            y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
            y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
            # intersect y_pred, y_true with labels, eliminate items not in labels
            ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
            y_pred = y_pred[ind]
            y_true = y_true[ind]
            stats = {}
            for class_i in range(n_labels):
                stats[class_i] = {'TP':{},'FP':{},'FN':{},'TN':{}}
                class_i_indices = np.argwhere(y_true==class_i)
                not_class_i_indices = np.argwhere(y_true!=class_i)
                stats[int(class_i)]['TP'] = int(np.sum(y_pred[class_i_indices]==y_true[class_i_indices]))
                stats[int(class_i)]['FP'] = int(np.sum(y_pred[class_i_indices]!=y_true[class_i_indices]))
                stats[int(class_i)]['TN'] = int(np.sum(y_pred[not_class_i_indices]==y_true[not_class_i_indices]))
                stats[int(class_i)]['FN'] = int(np.sum(y_pred[not_class_i_indices]!=y_true[not_class_i_indices]))
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
        mtrx, stats = confusion_matrix(np.array(true_labels).reshape(-1), np.array(preds).reshape(-1))
        if args.json:
            json_dict['confusion_matrix'] = mtrx.tolist()
            if n_classes > 2:
                json_dict['multiclass_stats'] = stats
            print(json.dumps(json_dict))
        else:
            mtrx = mtrx / np.sum(mtrx) * 100.0
            print("Confusion Matrix:")
            print(' ' + np.array2string(mtrx, formatter={'float': (lambda x: '{:.2f}%'.format(round(float(x), 2)))})[1:-1])


    #remove tempfile if created
    if not args.cleanfile: 
        os.remove(cleanfile)
        if output!=-1:
            os.remove(preprocessedfile)
