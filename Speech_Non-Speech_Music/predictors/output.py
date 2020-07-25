#!/usr/bin/env python3
#
# This code is was produced by an alpha version of Brainome Daimensions(tm) and is
# licensed under GNU GPL v2.0 or higher. For details, please see:
# https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
#
#
# Output of Brainome Daimensions(tm) 0.96 Table Compiler v0.96.
# Invocation: btc -vv -f NN -o output_v3.py combined_train.csv
# Total compiler execution time: 4:14:05.33. Finished on: Jun-14-2020 02:17:33.
# This source code requires Python 3.
#
"""
Classifier Type: Neural Network
System Type:                        3-way classifier
Best-guess accuracy:                33.69%
Model accuracy:                     95.63% (442535/462719 correct)
Improvement over best guess:        61.94% (of possible 66.31%)
Model capacity (MEC):               795 bits
Generalization ratio:               556.64 bits/bit
Confusion Matrix:
 [33.33% 0.26% 0.10%]
 [0.22% 31.61% 1.37%]
 [0.17% 2.25% 30.69%]

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
TRAINFILE = "combined_train.csv"


#Number of output logits
num_output_logits = 3

#Number of attributes
num_attr = 95
n_classes = 3


# Preprocessor for CSV files
def preprocess(inputcsvfile, outputcsvfile, headerless=False, testfile=False, target='', ignorecolumns=[], ignorelabels=[]):
    il=[]
    
    ignorelabels=[]
    ignorecolumns=[]
    target=""


    if (testfile):
        target = ''
    
    with open(outputcsvfile, "w+") as outputfile:
        with open(inputcsvfile) as csvfile:
            reader = csv.reader(csvfile)
            if (headerless == False):
                header=next(reader, None)
                try:
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
                        if (col == hc):
                            raise ValueError("Attribute '" + ignorecolumns[i] + "' is the target. Header must be same as in file passed to btc.")
                        il=il+[col]
                    except ValueError:
                        raise
                    except:
                        raise NameError("Attribute '" + ignorecolumns[i] + "' not found in header. Header must be same as in file passed to btc.")
                for i in range(0, len(header)):      
                    if (i == hc):
                        continue
                    if (i in il):
                        continue
                    print(header[i] + ",", end='', file=outputfile)
                print(header[hc], file=outputfile)

                for row in csv.DictReader(open(inputcsvfile)):
                    if (row[target] in ignorelabels):
                        continue
                    for name in header:
                        if (name in ignorecolumns):
                            continue
                        if (name==target):
                            continue
                        if (',' in row[name]):
                            print ('"' + row[name] + '"' + ",", end='', file=outputfile)
                        else:
                            print (row[name] + ",", end='', file=outputfile)
                    print (row[target], file=outputfile)

            else:
                try:
                    if (target != ""): 
                        hc = int(target)
                    else:
                        hc =- 1
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
                    if (hc == -1):
                        hc = len(row) - 1
                    if (row[hc] in ignorelabels):
                        continue
                    for i in range(0, len(row)):
                        if (i in il):
                            continue
                        if (i == hc):
                            continue
                        if (',' in row[i]):
                            print ('"' + row[i] + '"'+",", end='', file=outputfile)
                        else:
                            print(row[i]+",", end = '', file=outputfile)
                    print (row[hc], file=outputfile)

def clean(filename, outfile, rounding=-1, headerless=False, testfile=False):
    
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
                result = float(value)
                if (rounding != -1):
                    result = int(result * math.pow(10, rounding)) / math.pow(10, rounding)
                return result
            except:
                result = (binascii.crc32(value.encode('utf8')) % (1 << 32))
                return result

    # function to return key for any value 
    def get_key(val, clean_classmapping):
        if clean_classmapping == {}:
            return val
        for key, value in clean_classmapping.items(): 
            if val == value:
                return key
        if val not in list(clean_classmapping.values):
            raise ValueError("Label key does not exist")

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
            rowlen = num_attr
            if (not testfile):
                rowlen = rowlen + 1    
            if (not len(row) == rowlen):
                raise ValueError("Column count must match trained predictor. Row " + str(rowcount) + " differs.")
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
def single_classify(row):
    x = row
    o = [0] * num_output_logits
    h_0 = max((((-60.95658 * float(x[0]))+ (-72.27302 * float(x[1]))+ (-55.604187 * float(x[2]))+ (-35.946957 * float(x[3]))+ (27.21736 * float(x[4]))+ (25.84039 * float(x[5]))+ (19.176247 * float(x[6]))+ (10.468158 * float(x[7]))+ (45.17017 * float(x[8]))+ (48.561356 * float(x[9]))+ (39.106304 * float(x[10]))+ (60.541557 * float(x[11]))+ (-18.049822 * float(x[12]))+ (-16.805124 * float(x[13]))+ (-21.176455 * float(x[14]))+ (-15.099314 * float(x[15]))+ (11.981693 * float(x[16]))+ (10.85762 * float(x[17]))+ (13.247188 * float(x[18]))+ (12.042729 * float(x[19]))+ (-16.51618 * float(x[20]))+ (-26.58956 * float(x[21]))+ (-14.254891 * float(x[22]))+ (-13.487027 * float(x[23]))+ (5.95347 * float(x[24]))+ (10.1826935 * float(x[25]))+ (-4.0240192 * float(x[26]))+ (-7.8618565 * float(x[27]))+ (-2.99053 * float(x[28]))+ (8.964611 * float(x[29]))+ (-1.7295198 * float(x[30]))+ (2.1419425 * float(x[31]))+ (-3.0624666 * float(x[32]))+ (-7.2396207 * float(x[33]))+ (-3.251264 * float(x[34]))+ (-3.357675 * float(x[35]))+ (5.823252 * float(x[36]))+ (3.788304 * float(x[37]))+ (-3.8259952 * float(x[38]))+ (-3.8160696 * float(x[39]))+ (-7.6789303 * float(x[40]))+ (3.8074176 * float(x[41]))+ (-0.026857527 * float(x[42]))+ (-2.2165706 * float(x[43]))+ (-2.602303 * float(x[44]))+ (-10.074194 * float(x[45]))+ (-2.9072764 * float(x[46]))+ (2.5838866 * float(x[47]))+ (-16.464819 * float(x[48]))+ (-6.0550222 * float(x[49])))+ ((-18.324255 * float(x[50]))+ (-10.844429 * float(x[51]))+ (-0.5762511 * float(x[52]))+ (1.8605539 * float(x[53]))+ (-7.330379 * float(x[54]))+ (-5.81609 * float(x[55]))+ (2.1122575 * float(x[56]))+ (-2.408815 * float(x[57]))+ (-0.5348956 * float(x[58]))+ (-0.2147233 * float(x[59]))+ (-5.9699435 * float(x[60]))+ (-7.8348417 * float(x[61]))+ (-6.3865714 * float(x[62]))+ (-11.240323 * float(x[63]))+ (-13.33869 * float(x[64]))+ (-4.371828 * float(x[65]))+ (3.4682035 * float(x[66]))+ (-9.334897 * float(x[67]))+ (-2.7162921 * float(x[68]))+ (-12.337206 * float(x[69]))+ (-17.412315 * float(x[70]))+ (-3.183779 * float(x[71]))+ (-1.779266 * float(x[72]))+ (-1.1326575 * float(x[73]))+ (15.177793 * float(x[74]))+ (-2.4192102 * float(x[75]))+ (153.11595 * float(x[76]))+ (54.753586 * float(x[77]))+ (61.939884 * float(x[78]))+ (3.6447735 * float(x[79]))+ (22.524141 * float(x[80]))+ (-22.569202 * float(x[81]))+ (10.604531 * float(x[82]))+ (11.44169 * float(x[83]))+ (-8.324102 * float(x[84]))+ (3.720106 * float(x[85]))+ (16.432808 * float(x[86]))+ (11.039695 * float(x[87]))+ (-4.3495474 * float(x[88]))+ (5.011449 * float(x[89]))+ (7.0886927 * float(x[90]))+ (-2.6100895 * float(x[91]))+ (6.445954 * float(x[92]))+ (-2.627587 * float(x[93]))+ (11.700932 * float(x[94]))) + 5.4458904), 0)
    h_1 = max((((-24.443422 * float(x[0]))+ (-34.495968 * float(x[1]))+ (-30.83906 * float(x[2]))+ (-21.99012 * float(x[3]))+ (-3.9777849 * float(x[4]))+ (-2.1000113 * float(x[5]))+ (-1.5139793 * float(x[6]))+ (-3.687087 * float(x[7]))+ (9.040654 * float(x[8]))+ (13.944193 * float(x[9]))+ (12.614662 * float(x[10]))+ (15.284685 * float(x[11]))+ (-2.1863403 * float(x[12]))+ (-6.305596 * float(x[13]))+ (-5.5925455 * float(x[14]))+ (-10.470548 * float(x[15]))+ (-3.80964 * float(x[16]))+ (-0.44685528 * float(x[17]))+ (-0.78435266 * float(x[18]))+ (3.4656906 * float(x[19]))+ (-7.97974 * float(x[20]))+ (-8.445615 * float(x[21]))+ (-9.880174 * float(x[22]))+ (-5.9711075 * float(x[23]))+ (4.898576 * float(x[24]))+ (3.1628485 * float(x[25]))+ (7.782749 * float(x[26]))+ (2.9429579 * float(x[27]))+ (-6.4600077 * float(x[28]))+ (-3.9015357 * float(x[29]))+ (-5.8723845 * float(x[30]))+ (-8.178543 * float(x[31]))+ (-3.8561823 * float(x[32]))+ (-2.5702562 * float(x[33]))+ (0.24021761 * float(x[34]))+ (4.5190434 * float(x[35]))+ (-2.9881415 * float(x[36]))+ (-5.845767 * float(x[37]))+ (-3.4122477 * float(x[38]))+ (-7.252284 * float(x[39]))+ (-1.8663523 * float(x[40]))+ (6.7633405 * float(x[41]))+ (-1.678628 * float(x[42]))+ (1.5504686 * float(x[43]))+ (-9.450191 * float(x[44]))+ (-5.9356356 * float(x[45]))+ (-3.6088498 * float(x[46]))+ (-7.3725667 * float(x[47]))+ (1.1008111 * float(x[48]))+ (-4.022381 * float(x[49])))+ ((-6.5911913 * float(x[50]))+ (-2.7753513 * float(x[51]))+ (-9.070551 * float(x[52]))+ (-5.067471 * float(x[53]))+ (-5.690562 * float(x[54]))+ (-4.0955253 * float(x[55]))+ (7.1242404 * float(x[56]))+ (3.3177931 * float(x[57]))+ (13.494422 * float(x[58]))+ (13.25005 * float(x[59]))+ (-2.1489425 * float(x[60]))+ (-1.5814084 * float(x[61]))+ (-7.1227703 * float(x[62]))+ (-5.9871116 * float(x[63]))+ (-2.4674125 * float(x[64]))+ (-0.87870455 * float(x[65]))+ (-2.2079241 * float(x[66]))+ (-1.3127949 * float(x[67]))+ (-4.476563 * float(x[68]))+ (-9.38049 * float(x[69]))+ (-6.371033 * float(x[70]))+ (-0.3699982 * float(x[71]))+ (3.1916957 * float(x[72]))+ (-3.5227036 * float(x[73]))+ (2.3264008 * float(x[74]))+ (1.5546644 * float(x[75]))+ (51.45848 * float(x[76]))+ (-0.37874705 * float(x[77]))+ (5.6358232 * float(x[78]))+ (0.9763395 * float(x[79]))+ (10.516539 * float(x[80]))+ (-1.6436177 * float(x[81]))+ (23.469519 * float(x[82]))+ (1.62865 * float(x[83]))+ (5.5855374 * float(x[84]))+ (-2.0573926 * float(x[85]))+ (-0.051763818 * float(x[86]))+ (-2.12118 * float(x[87]))+ (5.1322985 * float(x[88]))+ (-1.5344149 * float(x[89]))+ (13.8126955 * float(x[90]))+ (-0.079484835 * float(x[91]))+ (7.254901 * float(x[92]))+ (4.500438 * float(x[93]))+ (17.555017 * float(x[94]))) + 1.0312957), 0)
    h_2 = max((((-7.35376 * float(x[0]))+ (-6.2503395 * float(x[1]))+ (-5.375333 * float(x[2]))+ (-8.52848 * float(x[3]))+ (2.538272 * float(x[4]))+ (4.010568 * float(x[5]))+ (4.953466 * float(x[6]))+ (9.53243 * float(x[7]))+ (31.887833 * float(x[8]))+ (28.09766 * float(x[9]))+ (27.493025 * float(x[10]))+ (36.704483 * float(x[11]))+ (6.53705 * float(x[12]))+ (5.6840262 * float(x[13]))+ (3.3567517 * float(x[14]))+ (3.4519885 * float(x[15]))+ (-1.8899457 * float(x[16]))+ (3.921886 * float(x[17]))+ (7.9563036 * float(x[18]))+ (2.9801323 * float(x[19]))+ (9.53611 * float(x[20]))+ (14.287547 * float(x[21]))+ (10.037969 * float(x[22]))+ (12.701132 * float(x[23]))+ (-6.7558455 * float(x[24]))+ (-7.393866 * float(x[25]))+ (-7.531511 * float(x[26]))+ (-13.481911 * float(x[27]))+ (5.5995307 * float(x[28]))+ (5.5679164 * float(x[29]))+ (7.7329226 * float(x[30]))+ (6.998565 * float(x[31]))+ (14.030332 * float(x[32]))+ (6.051304 * float(x[33]))+ (2.4809477 * float(x[34]))+ (6.0860167 * float(x[35]))+ (-0.7040144 * float(x[36]))+ (-2.210124 * float(x[37]))+ (6.244258 * float(x[38]))+ (-3.3768382 * float(x[39]))+ (6.261568 * float(x[40]))+ (4.586321 * float(x[41]))+ (1.4201243 * float(x[42]))+ (8.390985 * float(x[43]))+ (0.8368033 * float(x[44]))+ (-4.1995134 * float(x[45]))+ (-4.201048 * float(x[46]))+ (1.082481 * float(x[47]))+ (-4.1921005 * float(x[48]))+ (-5.7132244 * float(x[49])))+ ((-3.1320221 * float(x[50]))+ (-6.6709647 * float(x[51]))+ (2.6471484 * float(x[52]))+ (8.122641 * float(x[53]))+ (4.1032763 * float(x[54]))+ (9.641125 * float(x[55]))+ (-4.6759944 * float(x[56]))+ (-5.736771 * float(x[57]))+ (-2.6046782 * float(x[58]))+ (-9.45264 * float(x[59]))+ (-1.0394961 * float(x[60]))+ (0.89264274 * float(x[61]))+ (-4.230859 * float(x[62]))+ (-2.4825182 * float(x[63]))+ (-7.4270453 * float(x[64]))+ (-4.757112 * float(x[65]))+ (-4.080243 * float(x[66]))+ (-5.814342 * float(x[67]))+ (-2.6568434 * float(x[68]))+ (-3.567222 * float(x[69]))+ (-1.4411491 * float(x[70]))+ (1.0789269 * float(x[71]))+ (0.11451503 * float(x[72]))+ (0.5113969 * float(x[73]))+ (-3.033267 * float(x[74]))+ (-0.91098714 * float(x[75]))+ (124.941475 * float(x[76]))+ (18.874828 * float(x[77]))+ (47.053947 * float(x[78]))+ (20.72272 * float(x[79]))+ (14.304702 * float(x[80]))+ (16.741072 * float(x[81]))+ (-4.9309497 * float(x[82]))+ (18.396229 * float(x[83]))+ (13.540529 * float(x[84]))+ (6.356679 * float(x[85]))+ (10.514492 * float(x[86]))+ (4.5886025 * float(x[87]))+ (7.3368444 * float(x[88]))+ (12.357455 * float(x[89]))+ (8.699417 * float(x[90]))+ (9.373304 * float(x[91]))+ (1.4247056 * float(x[92]))+ (9.686273 * float(x[93]))+ (6.6700864 * float(x[94]))) + -1.0023899), 0)
    h_3 = max((((10.581867 * float(x[0]))+ (16.499346 * float(x[1]))+ (27.009212 * float(x[2]))+ (29.299438 * float(x[3]))+ (-4.2860065 * float(x[4]))+ (-1.8850468 * float(x[5]))+ (-10.259572 * float(x[6]))+ (-11.328274 * float(x[7]))+ (5.0056887 * float(x[8]))+ (2.1079726 * float(x[9]))+ (0.692132 * float(x[10]))+ (-2.9385805 * float(x[11]))+ (-2.1263394 * float(x[12]))+ (-0.910463 * float(x[13]))+ (-6.629224 * float(x[14]))+ (-6.664541 * float(x[15]))+ (8.83224 * float(x[16]))+ (5.4671874 * float(x[17]))+ (4.8649898 * float(x[18]))+ (4.014493 * float(x[19]))+ (20.77083 * float(x[20]))+ (20.564896 * float(x[21]))+ (16.980545 * float(x[22]))+ (17.095541 * float(x[23]))+ (2.7699 * float(x[24]))+ (2.9590468 * float(x[25]))+ (-0.34691232 * float(x[26]))+ (-1.3999207 * float(x[27]))+ (3.4982064 * float(x[28]))+ (2.8186421 * float(x[29]))+ (5.5520706 * float(x[30]))+ (8.684059 * float(x[31]))+ (20.992146 * float(x[32]))+ (17.063345 * float(x[33]))+ (12.31909 * float(x[34]))+ (9.978604 * float(x[35]))+ (4.9146185 * float(x[36]))+ (9.180262 * float(x[37]))+ (7.64223 * float(x[38]))+ (8.827608 * float(x[39]))+ (30.483385 * float(x[40]))+ (20.608133 * float(x[41]))+ (21.7419 * float(x[42]))+ (28.762978 * float(x[43]))+ (2.119885 * float(x[44]))+ (3.831411 * float(x[45]))+ (0.1996216 * float(x[46]))+ (-0.8425816 * float(x[47]))+ (18.075233 * float(x[48]))+ (14.273663 * float(x[49])))+ ((16.565151 * float(x[50]))+ (16.50913 * float(x[51]))+ (0.58630973 * float(x[52]))+ (-0.47794184 * float(x[53]))+ (1.6456676 * float(x[54]))+ (2.1178062 * float(x[55]))+ (-1.0891362 * float(x[56]))+ (2.1297972 * float(x[57]))+ (0.59761953 * float(x[58]))+ (-5.149663 * float(x[59]))+ (5.3654037 * float(x[60]))+ (3.4172401 * float(x[61]))+ (3.0565898 * float(x[62]))+ (7.2429223 * float(x[63]))+ (8.103366 * float(x[64]))+ (6.1746173 * float(x[65]))+ (2.7290683 * float(x[66]))+ (4.004366 * float(x[67]))+ (1.9249935 * float(x[68]))+ (1.1677911 * float(x[69]))+ (-0.04486347 * float(x[70]))+ (-1.7167234 * float(x[71]))+ (10.088328 * float(x[72]))+ (10.960578 * float(x[73]))+ (5.229484 * float(x[74]))+ (4.5869904 * float(x[75]))+ (60.986267 * float(x[76]))+ (-18.796211 * float(x[77]))+ (-9.5673485 * float(x[78]))+ (-12.858235 * float(x[79]))+ (-12.253069 * float(x[80]))+ (7.2131295 * float(x[81]))+ (-10.305488 * float(x[82]))+ (-4.3173933 * float(x[83]))+ (10.310348 * float(x[84]))+ (13.485208 * float(x[85]))+ (3.2906287 * float(x[86]))+ (-2.809163 * float(x[87]))+ (6.9546256 * float(x[88]))+ (-5.4704094 * float(x[89]))+ (-9.306476 * float(x[90]))+ (-8.598185 * float(x[91]))+ (-18.893265 * float(x[92]))+ (-16.311855 * float(x[93]))+ (-19.254194 * float(x[94]))) + -1.6030476), 0)
    h_4 = max((((-12.850431 * float(x[0]))+ (-7.0652857 * float(x[1]))+ (4.3175797 * float(x[2]))+ (10.158157 * float(x[3]))+ (10.987566 * float(x[4]))+ (12.3310585 * float(x[5]))+ (4.14894 * float(x[6]))+ (1.8177812 * float(x[7]))+ (7.5925846 * float(x[8]))+ (14.078485 * float(x[9]))+ (9.521161 * float(x[10]))+ (7.009787 * float(x[11]))+ (-2.5303347 * float(x[12]))+ (-6.2207317 * float(x[13]))+ (-1.4313139 * float(x[14]))+ (-5.6805587 * float(x[15]))+ (3.29213 * float(x[16]))+ (-0.83970946 * float(x[17]))+ (-1.4062701 * float(x[18]))+ (2.6403117 * float(x[19]))+ (-4.0849977 * float(x[20]))+ (-1.1477824 * float(x[21]))+ (2.228712 * float(x[22]))+ (0.7680852 * float(x[23]))+ (-4.3778253 * float(x[24]))+ (-4.2961335 * float(x[25]))+ (-2.1506057 * float(x[26]))+ (-3.0761123 * float(x[27]))+ (0.712983 * float(x[28]))+ (2.3896828 * float(x[29]))+ (-2.14287 * float(x[30]))+ (-0.7415576 * float(x[31]))+ (0.37004676 * float(x[32]))+ (-0.48583525 * float(x[33]))+ (-1.9616963 * float(x[34]))+ (4.5933485 * float(x[35]))+ (-7.256851 * float(x[36]))+ (-6.3450546 * float(x[37]))+ (-3.1576188 * float(x[38]))+ (-3.1096978 * float(x[39]))+ (-5.219154 * float(x[40]))+ (1.6113211 * float(x[41]))+ (0.72442406 * float(x[42]))+ (1.5318036 * float(x[43]))+ (-8.81616 * float(x[44]))+ (-5.202873 * float(x[45]))+ (-6.0502424 * float(x[46]))+ (-7.3592305 * float(x[47]))+ (-9.308995 * float(x[48]))+ (-1.6509061 * float(x[49])))+ ((-7.195419 * float(x[50]))+ (-8.170858 * float(x[51]))+ (-3.1986616 * float(x[52]))+ (-4.624453 * float(x[53]))+ (2.0210547 * float(x[54]))+ (-1.1031069 * float(x[55]))+ (8.260479 * float(x[56]))+ (-1.0268947 * float(x[57]))+ (3.7046595 * float(x[58]))+ (7.08483 * float(x[59]))+ (-8.491044 * float(x[60]))+ (-3.4863808 * float(x[61]))+ (-4.860129 * float(x[62]))+ (-3.3768127 * float(x[63]))+ (-0.6741678 * float(x[64]))+ (6.3332567 * float(x[65]))+ (3.6421509 * float(x[66]))+ (4.3021626 * float(x[67]))+ (-7.852176 * float(x[68]))+ (-9.265291 * float(x[69]))+ (-3.824363 * float(x[70]))+ (-5.137028 * float(x[71]))+ (3.9865737 * float(x[72]))+ (-3.5231752 * float(x[73]))+ (-2.9085767 * float(x[74]))+ (-2.7996762 * float(x[75]))+ (104.8374 * float(x[76]))+ (39.543774 * float(x[77]))+ (16.706892 * float(x[78]))+ (7.345362 * float(x[79]))+ (18.87331 * float(x[80]))+ (17.68069 * float(x[81]))+ (10.627692 * float(x[82]))+ (3.9855285 * float(x[83]))+ (1.9712092 * float(x[84]))+ (-2.457443 * float(x[85]))+ (7.0008607 * float(x[86]))+ (-1.4701087 * float(x[87]))+ (3.040785 * float(x[88]))+ (1.1035904 * float(x[89]))+ (8.200738 * float(x[90]))+ (3.8611145 * float(x[91]))+ (12.958962 * float(x[92]))+ (1.8601408 * float(x[93]))+ (10.474859 * float(x[94]))) + -1.0929708), 0)
    h_5 = max((((24.031776 * float(x[0]))+ (-22.783329 * float(x[1]))+ (-26.270271 * float(x[2]))+ (-21.124554 * float(x[3]))+ (13.205943 * float(x[4]))+ (35.325 * float(x[5]))+ (24.237467 * float(x[6]))+ (23.367094 * float(x[7]))+ (6.137942 * float(x[8]))+ (-8.074801 * float(x[9]))+ (-5.0232778 * float(x[10]))+ (-1.1150707 * float(x[11]))+ (-2.2321427 * float(x[12]))+ (8.19128 * float(x[13]))+ (11.48009 * float(x[14]))+ (5.2782025 * float(x[15]))+ (1.0351186 * float(x[16]))+ (-6.9528394 * float(x[17]))+ (-4.4275794 * float(x[18]))+ (-0.78572243 * float(x[19]))+ (3.4373558 * float(x[20]))+ (4.772322 * float(x[21]))+ (-2.6623905 * float(x[22]))+ (3.9440727 * float(x[23]))+ (-3.4783218 * float(x[24]))+ (-2.9008806 * float(x[25]))+ (-2.062196 * float(x[26]))+ (-7.530859 * float(x[27]))+ (3.1305146 * float(x[28]))+ (-4.772989 * float(x[29]))+ (4.288374 * float(x[30]))+ (2.100361 * float(x[31]))+ (0.59857816 * float(x[32]))+ (1.1179781 * float(x[33]))+ (0.5344153 * float(x[34]))+ (5.2484937 * float(x[35]))+ (0.62725157 * float(x[36]))+ (1.1856136 * float(x[37]))+ (2.9763255 * float(x[38]))+ (-4.2440314 * float(x[39]))+ (0.19329594 * float(x[40]))+ (2.1588109 * float(x[41]))+ (1.1717051 * float(x[42]))+ (7.8648367 * float(x[43]))+ (0.21403259 * float(x[44]))+ (-8.503455 * float(x[45]))+ (-2.8369267 * float(x[46]))+ (-7.9526954 * float(x[47]))+ (-3.6370285 * float(x[48]))+ (0.24564931 * float(x[49])))+ ((0.9034166 * float(x[50]))+ (-3.7842822 * float(x[51]))+ (-4.155517 * float(x[52]))+ (-6.264285 * float(x[53]))+ (-4.701075 * float(x[54]))+ (-1.6150304 * float(x[55]))+ (-5.654778 * float(x[56]))+ (-5.053707 * float(x[57]))+ (1.6267427 * float(x[58]))+ (-2.4746838 * float(x[59]))+ (-2.0001752 * float(x[60]))+ (2.5353172 * float(x[61]))+ (-5.082971 * float(x[62]))+ (-1.7945251 * float(x[63]))+ (-3.4637337 * float(x[64]))+ (-5.0505123 * float(x[65]))+ (-2.6135867 * float(x[66]))+ (-5.7875414 * float(x[67]))+ (-2.4987252 * float(x[68]))+ (0.32938415 * float(x[69]))+ (-2.0912921 * float(x[70]))+ (-1.8370765 * float(x[71]))+ (-6.645807 * float(x[72]))+ (-2.2225351 * float(x[73]))+ (-6.2116623 * float(x[74]))+ (-2.2429624 * float(x[75]))+ (26.124277 * float(x[76]))+ (45.371 * float(x[77]))+ (9.814711 * float(x[78]))+ (25.829346 * float(x[79]))+ (7.236918 * float(x[80]))+ (-0.66033787 * float(x[81]))+ (6.458024 * float(x[82]))+ (7.37868 * float(x[83]))+ (7.453331 * float(x[84]))+ (4.2574472 * float(x[85]))+ (6.8862114 * float(x[86]))+ (-3.371191 * float(x[87]))+ (0.94556546 * float(x[88]))+ (-1.4684026 * float(x[89]))+ (4.008521 * float(x[90]))+ (1.3435276 * float(x[91]))+ (5.2855496 * float(x[92]))+ (-1.091568 * float(x[93]))+ (3.1442325 * float(x[94]))) + 4.5407495), 0)
    h_6 = max((((-1.2226591 * float(x[0]))+ (-1.4363482 * float(x[1]))+ (-1.0925888 * float(x[2]))+ (-0.96348405 * float(x[3]))+ (0.6706306 * float(x[4]))+ (0.7563684 * float(x[5]))+ (0.9961064 * float(x[6]))+ (0.67206734 * float(x[7]))+ (0.95883465 * float(x[8]))+ (1.5237161 * float(x[9]))+ (1.5366399 * float(x[10]))+ (1.6561584 * float(x[11]))+ (0.11091303 * float(x[12]))+ (0.011640121 * float(x[13]))+ (0.1572995 * float(x[14]))+ (0.021461183 * float(x[15]))+ (-0.31897864 * float(x[16]))+ (0.8305375 * float(x[17]))+ (0.33964455 * float(x[18]))+ (0.1149533 * float(x[19]))+ (-0.21095243 * float(x[20]))+ (-0.13858639 * float(x[21]))+ (-0.22897272 * float(x[22]))+ (-0.08525039 * float(x[23]))+ (-0.15383911 * float(x[24]))+ (0.47935534 * float(x[25]))+ (0.41899577 * float(x[26]))+ (-0.46592334 * float(x[27]))+ (0.3504033 * float(x[28]))+ (0.29222178 * float(x[29]))+ (0.14777511 * float(x[30]))+ (0.11087994 * float(x[31]))+ (-0.34930533 * float(x[32]))+ (0.0314204 * float(x[33]))+ (-0.021217449 * float(x[34]))+ (0.22288541 * float(x[35]))+ (0.29808035 * float(x[36]))+ (-0.10175928 * float(x[37]))+ (0.13954113 * float(x[38]))+ (0.28362814 * float(x[39]))+ (-0.34750065 * float(x[40]))+ (0.40312254 * float(x[41]))+ (-0.12686148 * float(x[42]))+ (0.29640797 * float(x[43]))+ (0.076088324 * float(x[44]))+ (-0.330517 * float(x[45]))+ (0.032698754 * float(x[46]))+ (0.24398202 * float(x[47]))+ (-0.36097488 * float(x[48]))+ (-0.19714941 * float(x[49])))+ ((-0.7303145 * float(x[50]))+ (-0.3530917 * float(x[51]))+ (0.28507414 * float(x[52]))+ (-0.062047735 * float(x[53]))+ (0.057464786 * float(x[54]))+ (0.54667085 * float(x[55]))+ (-0.26175216 * float(x[56]))+ (0.01134947 * float(x[57]))+ (0.32136843 * float(x[58]))+ (0.41718435 * float(x[59]))+ (0.13342862 * float(x[60]))+ (-0.030716727 * float(x[61]))+ (-0.27168444 * float(x[62]))+ (0.09728727 * float(x[63]))+ (-0.6480618 * float(x[64]))+ (-0.6048668 * float(x[65]))+ (-0.40872565 * float(x[66]))+ (-0.34746093 * float(x[67]))+ (-0.3116712 * float(x[68]))+ (-1.13692 * float(x[69]))+ (-0.62188244 * float(x[70]))+ (0.120651186 * float(x[71]))+ (-0.3480453 * float(x[72]))+ (-0.61541605 * float(x[73]))+ (-0.27242887 * float(x[74]))+ (-0.44616327 * float(x[75]))+ (3.6167762 * float(x[76]))+ (0.9801378 * float(x[77]))+ (0.61061215 * float(x[78]))+ (0.3166044 * float(x[79]))+ (-0.034112792 * float(x[80]))+ (0.17229739 * float(x[81]))+ (-0.19144985 * float(x[82]))+ (0.54289013 * float(x[83]))+ (0.10231316 * float(x[84]))+ (0.2756806 * float(x[85]))+ (0.07394748 * float(x[86]))+ (0.43017083 * float(x[87]))+ (-0.042735774 * float(x[88]))+ (0.32975188 * float(x[89]))+ (-0.07951815 * float(x[90]))+ (0.41797483 * float(x[91]))+ (0.18923916 * float(x[92]))+ (0.7148087 * float(x[93]))+ (0.5878524 * float(x[94]))) + 0.8336576), 0)
    h_7 = max((((1.3835437 * float(x[0]))+ (-1.314029 * float(x[1]))+ (-1.4863309 * float(x[2]))+ (-1.3970207 * float(x[3]))+ (0.91457814 * float(x[4]))+ (2.2831717 * float(x[5]))+ (1.9643431 * float(x[6]))+ (1.729254 * float(x[7]))+ (0.24806702 * float(x[8]))+ (-0.10257597 * float(x[9]))+ (0.24672803 * float(x[10]))+ (0.1640925 * float(x[11]))+ (0.23391949 * float(x[12]))+ (0.72825205 * float(x[13]))+ (1.1467311 * float(x[14]))+ (0.5636795 * float(x[15]))+ (-0.42036796 * float(x[16]))+ (0.14243135 * float(x[17]))+ (-0.23370309 * float(x[18]))+ (-0.17361791 * float(x[19]))+ (0.2163251 * float(x[20]))+ (0.50955606 * float(x[21]))+ (-0.17841692 * float(x[22]))+ (0.28771883 * float(x[23]))+ (-0.42106953 * float(x[24]))+ (0.1300609 * float(x[25]))+ (0.35953346 * float(x[26]))+ (-0.657883 * float(x[27]))+ (0.5167584 * float(x[28]))+ (-0.19054814 * float(x[29]))+ (0.36920723 * float(x[30]))+ (0.14282794 * float(x[31]))+ (-0.3360565 * float(x[32]))+ (0.1628594 * float(x[33]))+ (0.02779391 * float(x[34]))+ (0.5073907 * float(x[35]))+ (0.20154692 * float(x[36]))+ (-0.085851945 * float(x[37]))+ (0.31329072 * float(x[38]))+ (0.09723946 * float(x[39]))+ (-0.22891407 * float(x[40]))+ (0.3732725 * float(x[41]))+ (-0.075580694 * float(x[42]))+ (0.6962412 * float(x[43]))+ (0.117341556 * float(x[44]))+ (-0.6033719 * float(x[45]))+ (-0.06389876 * float(x[46]))+ (-0.29310322 * float(x[47]))+ (-0.24643527 * float(x[48]))+ (-0.034823656 * float(x[49])))+ ((-0.29454464 * float(x[50]))+ (-0.3222263 * float(x[51]))+ (0.010433452 * float(x[52]))+ (-0.51139313 * float(x[53]))+ (-0.118902914 * float(x[54]))+ (0.43465972 * float(x[55]))+ (-0.58710575 * float(x[56]))+ (-0.22286695 * float(x[57]))+ (0.38086873 * float(x[58]))+ (0.28716263 * float(x[59]))+ (0.096428044 * float(x[60]))+ (0.2377862 * float(x[61]))+ (-0.41127524 * float(x[62]))+ (0.18516022 * float(x[63]))+ (-0.529605 * float(x[64]))+ (-0.7562893 * float(x[65]))+ (-0.57340026 * float(x[66]))+ (-0.4661456 * float(x[67]))+ (-0.3813389 * float(x[68]))+ (-0.7913794 * float(x[69]))+ (-0.38828465 * float(x[70]))+ (0.032905754 * float(x[71]))+ (-0.6766631 * float(x[72]))+ (-0.6776222 * float(x[73]))+ (-0.8524635 * float(x[74]))+ (-0.49758095 * float(x[75]))+ (1.3666753 * float(x[76]))+ (2.5027306 * float(x[77]))+ (-0.21376376 * float(x[78]))+ (1.5799824 * float(x[79]))+ (-0.075932086 * float(x[80]))+ (0.38078913 * float(x[81]))+ (0.037953064 * float(x[82]))+ (0.5801303 * float(x[83]))+ (0.59187895 * float(x[84]))+ (0.37692773 * float(x[85]))+ (0.15266517 * float(x[86]))+ (-0.036969762 * float(x[87]))+ (0.060929265 * float(x[88]))+ (0.03285707 * float(x[89]))+ (-0.0052948683 * float(x[90]))+ (0.42220742 * float(x[91]))+ (0.39274648 * float(x[92]))+ (0.54602313 * float(x[93]))+ (0.46107793 * float(x[94]))) + 1.7101953), 0)
    o[0] = (1.0669268 * h_0)+ (2.0285037 * h_1)+ (1.9239185 * h_2)+ (2.8742993 * h_3)+ (1.9516325 * h_4)+ (3.538755 * h_5)+ (3.554911 * h_6)+ (-1.7914352 * h_7) + 1.185954
    o[1] = (1.128735 * h_0)+ (2.0344641 * h_1)+ (1.9484698 * h_2)+ (2.8749301 * h_3)+ (1.9489815 * h_4)+ (3.3273065 * h_5)+ (0.19988742 * h_6)+ (1.8598447 * h_7) + 5.956156
    o[2] = (1.1289487 * h_0)+ (2.0359266 * h_1)+ (1.9486194 * h_2)+ (2.8747385 * h_3)+ (1.9496144 * h_4)+ (3.327738 * h_5)+ (0.15979232 * h_6)+ (1.859456 * h_7) + -1.825723

    if num_output_logits == 1:
        return o[0] >= 0
    else:
        return argmax(o)


#for classifying batches
def classify(arr):
    outputs = []
    for row in arr:
        outputs.append(single_classify(row))
    return outputs

def Validate(cleanvalfile):
    #Binary
    if n_classes == 2:
        with open(cleanvalfile, 'r') as valcsvfile:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
            valcsvreader = csv.reader(valcsvfile)
            for valrow in valcsvreader:
                if len(valrow) == 0:
                    continue
                if int(single_classify(valrow[:-1])) == int(float(valrow[-1])):
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
        return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0

    #Multiclass
    else:
        with open(cleanvalfile, 'r') as valcsvfile:
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
                    numeachclass[y_true] = 0
                count += 1
        return count, correct_count, numeachclass, preds,  y_trues



def Predict(cleanfile, preprocessedfile, headerless, get_key, classmapping):
    with open(cleanfile,'r') as cleancsvfile, open(preprocessedfile,'r') as dirtycsvfile:
        cleancsvreader = csv.reader(cleancsvfile)
        dirtycsvreader = csv.reader(dirtycsvfile)
        if (not headerless):
            print(','.join(next(dirtycsvreader, None) + ["Prediction"]))
        for cleanrow, dirtyrow in zip(cleancsvreader, dirtycsvreader):
            if len(cleanrow) == 0:
                continue
            print(str(','.join(str(j) for j in ([i for i in dirtyrow]))) + ',' + str(get_key(int(single_classify(cleanrow)), classmapping)))



# Main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on '+TRAINFILE)
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-cleanfile', action='store_true', help='Use this flag to save prediction time if the csvfile you are passing has already been preprocessed. Implies headerless.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    args = parser.parse_args()
    faulthandler.enable()
    
    #clean if not already clean
    if not args.cleanfile:
        tempdir = tempfile.gettempdir()
        cleanfile = tempfile.NamedTemporaryFile().name
        preprocessedfile = tempfile.NamedTemporaryFile().name
        preprocess(args.csvfile,preprocessedfile,args.headerless,(not args.validate))
        get_key, classmapping = clean(preprocessedfile, cleanfile, -1, args.headerless, (not args.validate))
    else:
        cleanfile=args.csvfile
        preprocessedfile=args.csvfile
        get_key = lambda x, y: x
        classmapping = {}


    #Predict
    if not args.validate:
        Predict(cleanfile, preprocessedfile, args.headerless, get_key, classmapping)


    #Validate
    else: 
        print("Classifier Type: Neural Network")
        if n_classes == 2:
            count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = Validate(cleanfile)
        else:
            count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)

        #Report Metrics
        model_cap=795
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
            modelacc = int(float(num_correct * 10000) / count) / 100.0
            #Report
            print("System Type:                        Binary classifier")
            print("Best-guess accuracy:                {:.2f}%".format(randguess))
            print("Model accuracy:                     {:.2f}%".format(modelacc) + " (" + str(int(num_correct)) + "/" + str(count) + " correct)")
            print("Improvement over best guess:        {:.2f}%".format(modelacc - randguess) + " (of possible " + str(round(100 - randguess, 2)) + "%)")
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(num_correct * 100) / model_cap) / 100.0) + " bits/bit")
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
            print("System Type:                        " + str(n_classes) + "-way classifier")
            print("Best-guess accuracy:                {:.2f}%".format(randguess))
            print("Model accuracy:                     {:.2f}%".format(modelacc) + " (" + str(int(num_correct)) + "/" + str(count) + " correct)")
            print("Improvement over best guess:        {:.2f}%".format(modelacc - randguess) + " (of possible " + str(round(100 - randguess, 2)) + "%)")
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(num_correct * 100) / model_cap) / 100.0) + " bits/bit")
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


            print("Confusion Matrix:")
            mtrx = confusion_matrix(np.array(true_labels).reshape(-1), np.array(preds).reshape(-1))
            mtrx = mtrx / np.sum(mtrx) * 100.0
            print(' ' + np.array2string(mtrx, formatter={'float': (lambda x: '{:.2f}%'.format(round(float(x), 2)))})[1:-1])


    #Clean Up
    if not args.cleanfile:
        os.remove(cleanfile)
        os.remove(preprocessedfile)

