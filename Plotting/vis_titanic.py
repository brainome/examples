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
# Output of Brainome Daimensions(tm) 0.99 Table Compiler v0.99.
# Invocation: btc -f NN titanic_train.csv -o vis_titanic.py -target Survived -rank -e 5
# Total compiler execution time: 0:05:02.91. Finished on: Feb-25-2021 05:37:55.
# This source code requires Python 3.
#
"""
Classifier Type:                     Neural Network
System Type:                         Binary classifier
Training/Validation Split:           70:30%
Best-guess accuracy:                 61.61%
Training accuracy:                   83.30% (519/623 correct)
Validation accuracy:                 75.74% (203/268 correct)
Overall Model accuracy:              81.03% (722/891 correct)
Overall Improvement over best guess: 19.42% (of possible 38.39%)
Model capacity (MEC):                25 bits
Generalization ratio:                27.75 bits/bit
Model efficiency:                    0.77%/parameter
System behavior
True Negatives:                      55.22% (492/891)
True Positives:                      25.81% (230/891)
False Negatives:                     12.57% (112/891)
False Positives:                     6.40% (57/891)
True Pos. Rate/Sensitivity/Recall:   0.67
True Neg. Rate/Specificity:          0.90
Precision:                           0.80
F-1 Measure:                         0.73
False Negative Rate/Miss Rate:       0.33
Critical Success Index:              0.58
Confusion Matrix:
 [55.22% 6.40%]
 [12.57% 25.81%]
Generalization efficiency:           13.66
Overfitting:                         No
Using only the important columns: Sex SibSp Parch Pclass 
Risk of coincidental column correlation: <0.001%

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
TRAINFILE = "titanic_train.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 4
num_attr_before_transform = 0
n_classes = 2

mappings = [{1249151596.0: 0, 1435361449.0: 1}]
list_of_cols_to_normalize = [1]

transform_true = False

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
        mean = np.array([])
        components = np.array([])
        whiten = None
        explained_variance = np.array([])
        X = X - mean

    X_transformed = np.dot(X, components.T)
    if whiten:
        X_transformed /= np.sqrt(explained_variance)
    return X_transformed

# Preprocessor for CSV files

ignorelabels=[]
ignorecolumns=["PassengerId","Name","Age","Ticket","Fare","Cabin","Embarked",]
target="Survived"
important_idxs=[1,3,5,6]

def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[], trim=False):
    #This function streams in a csv and outputs a csv with the correct columns and target column on the right hand side. 
    #Precursor to clean

    il=[]

    ignorelabels=[]
    ignorecolumns=["PassengerId","Name","Age","Ticket","Fare","Cabin","Embarked",]
    target="Survived"
    important_idxs=[1,3,5,6]
    if ignorelabels == [] and ignorecolumns == [] and target == "":
        return -1
    if not trim:
        ignorecolumns = []
    if (testfile):
        target = ''
        hc = -1 
    with open(outputcsvfile, "w+") as outputfile:
        with open(inputcsvfile) as csvfile:
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

                for row in csv.DictReader(open(inputcsvfile)):
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
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        f = open(outfile, "w+")
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
    h_0 = max((((-4.474127 * float(x[0]))+ (-16.530014 * float(x[1]))+ (0.3632827 * float(x[2]))+ (0.32136515 * float(x[3]))) + 12.100474), 0)
    h_1 = max((((0.65849805 * float(x[0]))+ (2.2165382 * float(x[1]))+ (-0.6220546 * float(x[2]))+ (-4.196301 * float(x[3]))) + -0.8042341), 0)
    h_2 = max((((-1.2425824 * float(x[0]))+ (-0.665531 * float(x[1]))+ (-1.277177 * float(x[2]))+ (-0.11120979 * float(x[3]))) + -0.6508854), 0)
    h_3 = max((((0.19137733 * float(x[0]))+ (2.001466 * float(x[1]))+ (0.7650189 * float(x[2]))+ (0.7440585 * float(x[3]))) + -0.6627988), 0)
    o[0] = (0.48395234 * h_0)+ (-0.7045365 * h_1)+ (0.74831176 * h_2)+ (-0.6090292 * h_3) + 1.3535782



    #Output Decision Rule
    if num_output_logits==1:
        if return_probabilities:
            exp_o = 1./(1. + np.exp(-o[0]))
            return np.array([exp_o, 1.-exp_o])
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
        trans = transform(arr[:, :-1])
        arr = np.concatenate((trans, cleanarr[:, -1].reshape(-1, 1)), axis = 1)
    #init
    w_h = np.array([[-4.474126815795898, -16.530014038085938, 0.363282710313797, 0.3213651478290558], [0.6584980487823486, 2.216538190841675, -0.6220545768737793, -4.196300983428955], [-1.2425824403762817, -0.6655309796333313, -1.2771769762039185, -0.11120978742837906], [0.19137732684612274, 2.0014660358428955, 0.765018880367279, 0.7440584897994995]])
    b_h = np.array([12.10047435760498, -0.8042340874671936, -0.6508854031562805, -0.6627988219261169])
    w_o = np.array([[0.4839523434638977, -0.704536497592926, 0.7483117580413818, -0.6090291738510132]])
    b_o = np.array(1.3535782098770142)

    #Hidden Layer
    h = np.dot(arr, w_h.T) + b_h
    
    relu = np.maximum(h, np.zeros_like(h))


    #Output
    out = np.dot(relu, w_o.T) + b_o
    if return_probabilities:
        return np.array(np.exp(out)) / np.sum(np.exp(out), axis=1).reshape(-1,1)
    if num_output_logits == 1:
        if return_probabilities:
            exp_o = 1./(1. + np.exp(-out))
            return np.concatenate((exp_o, 1.-exp_o), axis=1)
        else:
            return (out >= 0).astype('int').reshape(-1)
    else:
        if return_probabilities:
            exps = np.exp(out)
            Z = sum(exps, axis=1).reshape(-1, 1)
            return exps/Z
        else:
            return (np.argmax(out, axis=1)).reshape(-1)



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
        model_cap = 25
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
                print("Note: If you install scipy (https://www.scipy.org) this predictor generates a confusion matrix")
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
            return cm
        mtrx = confusion_matrix(np.array(true_labels).reshape(-1), np.array(preds).reshape(-1))
        if args.json:
            json_dict['confusion_matrix'] = mtrx.tolist()
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
