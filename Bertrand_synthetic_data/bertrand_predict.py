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
# Invocation: btc -f NN bertrandtrain.csv -o bertrand_predict.py -headerless -e 10 --yes
# Total compiler execution time: 0:03:47.51. Finished on: Feb-23-2021 21:41:45.
# This source code requires Python 3.
#
"""
Classifier Type:                    Neural Network
System Type:                         Binary classifier
Training/Validation Split:           50:50%
Best-guess accuracy:                 62.65%
Training accuracy:                   100.00% (6593/6593 correct)
Validation accuracy:                 100.00% (6594/6594 correct)
Overall Model accuracy:              100.00% (13187/13187 correct)
Overall Improvement over best guess: 37.35% (of possible 37.35%)
Model capacity (MEC):                61 bits
Generalization ratio:                102.69 bits/bit
Model efficiency:                    0.61%/parameter
System behavior
True Negatives:                      37.35% (4925/13187)
True Positives:                      62.65% (8262/13187)
False Negatives:                     0.00% (0/13187)
False Positives:                     0.00% (0/13187)
True Pos. Rate/Sensitivity/Recall:   1.00
True Neg. Rate/Specificity:          1.00
Precision:                           1.00
F-1 Measure:                         1.00
False Negative Rate/Miss Rate:       0.00
Critical Success Index:              1.00
Confusion Matrix:
 [37.35% 0.00%]
 [0.00% 62.65%]
Generalization index:           50.41
Overfitting:                         No
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
TRAINFILE = "bertrandtrain.csv"


#Number of output logits
num_output_logits = 1

#Number of attributes
num_attr = 10
n_classes = 2
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


# Helper (save an import)
def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)
# Classifier
def single_classify(row, return_probabilities=False):
    x = row
    o = [0] * num_output_logits
    h_0 = max((((-5.685367 * float(x[0]))+ (-0.02004818 * float(x[1]))+ (-5.6867394 * float(x[2]))+ (-0.02039341 * float(x[3]))+ (5.6459394 * float(x[4]))+ (-0.083726525 * float(x[5]))+ (19.112246 * float(x[6]))+ (-0.07199509 * float(x[7]))+ (8.992511 * float(x[8]))+ (-0.051320046 * float(x[9]))) + -7.7231684), 0)
    h_1 = max((((-1.472554 * float(x[0]))+ (0.0005602278 * float(x[1]))+ (-1.4773116 * float(x[2]))+ (0.0002606527 * float(x[3]))+ (1.636749 * float(x[4]))+ (0.0006741732 * float(x[5]))+ (-6.8559494 * float(x[6]))+ (2.3902525e-05 * float(x[7]))+ (0.0011643022 * float(x[8]))+ (0.00078504335 * float(x[9]))) + 2.946482), 0)
    h_2 = max((((-0.32789797 * float(x[0]))+ (-0.9710082 * float(x[1]))+ (0.07482189 * float(x[2]))+ (-0.12774573 * float(x[3]))+ (-1.4239761 * float(x[4]))+ (-0.38576716 * float(x[5]))+ (-0.17901486 * float(x[6]))+ (-0.9842609 * float(x[7]))+ (-0.30547693 * float(x[8]))+ (-1.3153666 * float(x[9]))) + -0.85415345), 0)
    h_3 = max((((-3.819762 * float(x[0]))+ (0.00044885138 * float(x[1]))+ (-3.8202457 * float(x[2]))+ (-0.00018736518 * float(x[3]))+ (3.819572 * float(x[4]))+ (-0.00049131765 * float(x[5]))+ (0.21835068 * float(x[6]))+ (0.00019200495 * float(x[7]))+ (0.21798399 * float(x[8]))+ (-0.00028566373 * float(x[9]))) + -0.21846019), 0)
    h_4 = max((((-3.3158627 * float(x[0]))+ (0.006939412 * float(x[1]))+ (-3.312292 * float(x[2]))+ (0.011187401 * float(x[3]))+ (3.3360114 * float(x[4]))+ (0.006155446 * float(x[5]))+ (0.21085843 * float(x[6]))+ (0.0052557657 * float(x[7]))+ (0.21676855 * float(x[8]))+ (0.008709883 * float(x[9]))) + 3.0680668), 0)
    o[0] = (1.9910675 * h_0)+ (6.7118 * h_1)+ (-0.6416731 * h_2)+ (9.502981 * h_3)+ (-8.262984 * h_4) + -3.4893322

    if num_output_logits == 1:
        if return_probabilities:
            exp_o = 1./(1. + math.exp(-o))
            return [exp_o, 1.-exp_o]
        else:
            return o[0] >= 0
    else:
        if return_probabilities:
            exps = [math.exp(x) for x in o]
            Z = sum(exps)
            return [x/Z for x in exps]
        else:
            return argmax(o)


#for classifying batches
def classify(arr, return_probabilities=False):
    outputs = []
    for row in arr:
        outputs.append(single_classify(row, return_probabilities))
    return outputs

def Validate(cleanvalfile):
    #Binary
    if n_classes == 2:
        with open(cleanvalfile, 'r', encoding='utf-8') as valcsvfile:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
            valcsvreader = csv.reader(valcsvfile)
            preds = []
            y_trues = []
            for valrow in valcsvreader:
                if len(valrow) == 0:
                    continue
                y_true = int(float(valrow[-1]))
                pred = int(single_classify(valrow[:-1]))
                y_trues.append(y_true)
                preds.append(pred)
                if pred == y_true:
                    correct_count += 1
                    if int(float(valrow[-1])) == 1:
                        num_class_1 += 1
                        num_TP += 1
                    else:
                        num_class_0 += 1
                        num_TN += 1
                else:
                    if int(float(valrow[-1])) == 1:
                        num_class_1 += 1
                        num_FN += 1
                    else:
                        num_class_0 += 1
                        num_FP += 1
                count += 1
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds, y_trues

    #Multiclass
    else:
        with open(cleanvalfile, 'r', encoding='utf-8') as valcsvfile:
            count, correct_count = 0, 0
            valcsvreader = csv.reader(valcsvfile)
            numeachclass = {}
            preds = []
            y_trues = []
            for i, valrow in enumerate(valcsvreader):
                pred = int(single_classify(valrow[:-1]))
                preds.append(pred)
                y_true = int(float(valrow[-1]))
                y_trues.append(y_true)
                if len(valrow) == 0:
                    continue
                if pred == y_true:
                    correct_count += 1
                #if class seen, add to its counter
                if y_true in numeachclass.keys():
                    numeachclass[y_true] += 1
                #initialize a new counter
                else:
                    numeachclass[y_true] = 1
                count += 1
        return count, correct_count, numeachclass, preds,  y_trues



def Predict(cleanfile, preprocessedfile, headerless, get_key, classmapping, trim=False):
    with open(cleanfile,'r', encoding='utf-8') as cleancsvfile, open(preprocessedfile,'r', encoding='utf-8') as dirtycsvfile:
        cleancsvreader = csv.reader(cleancsvfile)
        dirtycsvreader = csv.reader(dirtycsvfile)
        if (not headerless):
            print(','.join(next(dirtycsvreader, None) + ["Prediction"]))
        for cleanrow, dirtyrow in zip(cleancsvreader, dirtycsvreader):
            if len(cleanrow) == 0:
                continue
            if not trim and ignorecolumns != []:
                cleanrow = [x for i,x in enumerate(cleanrow) if i in important_idxs]
            print(str(','.join(str(j) for j in (['"' + field + '"' if ',' in field else field for field in dirtyrow]))) + ',' + str(get_key(int(single_classify(cleanrow)), classmapping)))



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


    #Predict
    if not args.validate:
        Predict(cleanfile, preprocessedfile if output!=-1 else args.csvfile, args.headerless, get_key, classmapping, trim=args.trim)


    #Validate
    else: 
        classifier_type = 'NN'
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds, true_labels = Validate(cleanfile)
        else:
            count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)

        #Report Metrics
        model_cap=61
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

    #Clean Up
    if not args.cleanfile:
        os.remove(cleanfile)
        if output!=-1:
            os.remove(preprocessedfile)
