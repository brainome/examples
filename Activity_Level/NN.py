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
# Invocation: btc train.csv -headerless -f NN -o NN.py -riskoverfit --yes
# Total compiler execution time: 0:13:58.97. Finished on: Mar-17-2021 05:51:54.
# This source code requires Python 3.
#
"""
Classifier Type:                    Neural Network
System Type:                         6-way classifier
Best-guess accuracy:                 18.88%
Overall Model accuracy:              99.96% (5147/5149 correct)
Overall Improvement over best guess: 81.08% (of possible 81.12%)
Model capacity (MEC):                3414 bits
Generalization ratio:                3.86 bits/bit
Model efficiency:                    0.02%/parameter
Confusion Matrix:
 [16.72% 0.00% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 14.99% 0.00% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 13.65% 0.00% 0.00% 0.00%]
 [0.00% 0.00% 0.00% 17.23% 0.02% 0.00%]
 [0.00% 0.00% 0.00% 0.02% 18.49% 0.00%]
 [0.00% 0.00% 0.00% 0.00% 0.00% 18.88%]
Avg. noise resilience per instance:  -0.18dB
Percent of Data Memorized:           80.20%
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


#Number of output logits
num_output_logits = 6

#Number of attributes
num_attr = 561
n_classes = 6
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
    x = row
    o = [0] * num_output_logits
    h_0 = max((((0.0040394743 * float(x[0]))+ (0.121068105 * float(x[1]))+ (0.060362518 * float(x[2]))+ (-0.031669825 * float(x[3]))+ (0.14288175 * float(x[4]))+ (0.02836771 * float(x[5]))+ (-0.045886964 * float(x[6]))+ (-0.031132178 * float(x[7]))+ (-0.04435249 * float(x[8]))+ (0.08157697 * float(x[9]))+ (0.12618625 * float(x[10]))+ (-0.02513086 * float(x[11]))+ (-0.089221604 * float(x[12]))+ (-0.04317546 * float(x[13]))+ (0.04466975 * float(x[14]))+ (0.013078937 * float(x[15]))+ (0.0957078 * float(x[16]))+ (0.11310046 * float(x[17]))+ (0.005445351 * float(x[18]))+ (-0.02237171 * float(x[19]))+ (0.0049456176 * float(x[20]))+ (0.13119246 * float(x[21]))+ (0.012979973 * float(x[22]))+ (0.019847056 * float(x[23]))+ (0.04929632 * float(x[24]))+ (0.096483566 * float(x[25]))+ (-0.0652057 * float(x[26]))+ (0.18837595 * float(x[27]))+ (-0.1347724 * float(x[28]))+ (0.14767796 * float(x[29]))+ (-0.01426425 * float(x[30]))+ (-0.027517073 * float(x[31]))+ (-0.08883849 * float(x[32]))+ (-0.032003317 * float(x[33]))+ (-0.04872453 * float(x[34]))+ (-0.120148785 * float(x[35]))+ (0.057326615 * float(x[36]))+ (0.1236204 * float(x[37]))+ (0.044443265 * float(x[38]))+ (-0.1324841 * float(x[39]))+ (-0.10231749 * float(x[40]))+ (0.101288214 * float(x[41]))+ (0.14231272 * float(x[42]))+ (0.09079933 * float(x[43]))+ (-0.011963304 * float(x[44]))+ (0.011001459 * float(x[45]))+ (-0.01716052 * float(x[46]))+ (0.10625348 * float(x[47]))+ (-0.043377247 * float(x[48]))+ (-0.13258073 * float(x[49])))+ ((0.13479115 * float(x[50]))+ (-0.04050518 * float(x[51]))+ (-0.09428142 * float(x[52]))+ (0.019785414 * float(x[53]))+ (0.10748508 * float(x[54]))+ (0.14564997 * float(x[55]))+ (-0.10749813 * float(x[56]))+ (-0.030866906 * float(x[57]))+ (0.070273794 * float(x[58]))+ (0.07586232 * float(x[59]))+ (0.10682572 * float(x[60]))+ (0.105206735 * float(x[61]))+ (-0.018718462 * float(x[62]))+ (0.06714017 * float(x[63]))+ (-0.017270321 * float(x[64]))+ (0.11806867 * float(x[65]))+ (-0.034631386 * float(x[66]))+ (0.1143877 * float(x[67]))+ (-0.07870278 * float(x[68]))+ (0.010854975 * float(x[69]))+ (-0.12192803 * float(x[70]))+ (-0.049829658 * float(x[71]))+ (0.019287944 * float(x[72]))+ (0.11444196 * float(x[73]))+ (0.029177755 * float(x[74]))+ (-0.04139909 * float(x[75]))+ (0.00458557 * float(x[76]))+ (-0.14878033 * float(x[77]))+ (-0.0036568542 * float(x[78]))+ (0.04258516 * float(x[79]))+ (-0.11743203 * float(x[80]))+ (-0.03344234 * float(x[81]))+ (-0.04449501 * float(x[82]))+ (0.04746515 * float(x[83]))+ (0.05276728 * float(x[84]))+ (0.08882212 * float(x[85]))+ (0.030124182 * float(x[86]))+ (0.09298934 * float(x[87]))+ (-0.0002778665 * float(x[88]))+ (-0.019505389 * float(x[89]))+ (0.040925346 * float(x[90]))+ (0.14070919 * float(x[91]))+ (0.002380478 * float(x[92]))+ (-0.13520946 * float(x[93]))+ (-0.055068597 * float(x[94]))+ (0.047674615 * float(x[95]))+ (0.023688229 * float(x[96]))+ (0.048435308 * float(x[97]))+ (0.054808494 * float(x[98]))+ (0.008667743 * float(x[99])))+ ((0.079214096 * float(x[100]))+ (0.14312974 * float(x[101]))+ (0.0047371574 * float(x[102]))+ (-0.12734143 * float(x[103]))+ (-0.060591586 * float(x[104]))+ (0.010272778 * float(x[105]))+ (-0.054844655 * float(x[106]))+ (0.013070999 * float(x[107]))+ (-0.0074318694 * float(x[108]))+ (0.05676857 * float(x[109]))+ (-0.020138543 * float(x[110]))+ (-0.020433905 * float(x[111]))+ (-0.13458373 * float(x[112]))+ (0.023305908 * float(x[113]))+ (-0.032662153 * float(x[114]))+ (-0.1260623 * float(x[115]))+ (-0.0696263 * float(x[116]))+ (0.104283944 * float(x[117]))+ (-0.021499531 * float(x[118]))+ (0.04255608 * float(x[119]))+ (0.09082162 * float(x[120]))+ (-0.033275 * float(x[121]))+ (0.056051787 * float(x[122]))+ (0.013803964 * float(x[123]))+ (0.11782424 * float(x[124]))+ (0.08931214 * float(x[125]))+ (-0.014075092 * float(x[126]))+ (-0.017786138 * float(x[127]))+ (0.081554964 * float(x[128]))+ (0.13384852 * float(x[129]))+ (0.021644877 * float(x[130]))+ (0.1015171 * float(x[131]))+ (0.048813757 * float(x[132]))+ (-0.075449675 * float(x[133]))+ (0.053948876 * float(x[134]))+ (0.009989302 * float(x[135]))+ (0.047398843 * float(x[136]))+ (0.12165444 * float(x[137]))+ (-0.006551987 * float(x[138]))+ (0.10793802 * float(x[139]))+ (0.042973448 * float(x[140]))+ (0.07319742 * float(x[141]))+ (-0.118093155 * float(x[142]))+ (-0.062293656 * float(x[143]))+ (-0.08780918 * float(x[144]))+ (0.13841721 * float(x[145]))+ (-0.114439934 * float(x[146]))+ (0.030319246 * float(x[147]))+ (0.08239245 * float(x[148]))+ (0.1189983 * float(x[149])))+ ((-0.104204126 * float(x[150]))+ (0.011597805 * float(x[151]))+ (-0.13891552 * float(x[152]))+ (0.1283933 * float(x[153]))+ (-0.06696179 * float(x[154]))+ (0.11184191 * float(x[155]))+ (-0.086883895 * float(x[156]))+ (0.045229606 * float(x[157]))+ (-0.026363559 * float(x[158]))+ (0.071659066 * float(x[159]))+ (0.13170741 * float(x[160]))+ (0.014292805 * float(x[161]))+ (0.0756323 * float(x[162]))+ (-0.009806465 * float(x[163]))+ (0.076887995 * float(x[164]))+ (0.068553954 * float(x[165]))+ (0.06085963 * float(x[166]))+ (0.041830663 * float(x[167]))+ (0.0957982 * float(x[168]))+ (0.010321209 * float(x[169]))+ (-0.011302893 * float(x[170]))+ (0.019537915 * float(x[171]))+ (-0.0012321672 * float(x[172]))+ (-0.13542283 * float(x[173]))+ (-0.091258 * float(x[174]))+ (-0.03093115 * float(x[175]))+ (0.113894805 * float(x[176]))+ (-0.013855899 * float(x[177]))+ (0.13009524 * float(x[178]))+ (-0.026447117 * float(x[179]))+ (0.011182207 * float(x[180]))+ (0.0061269575 * float(x[181]))+ (-0.0067432923 * float(x[182]))+ (0.04908026 * float(x[183]))+ (-0.0089264475 * float(x[184]))+ (0.053076845 * float(x[185]))+ (-0.008206726 * float(x[186]))+ (-0.13041364 * float(x[187]))+ (-0.024274338 * float(x[188]))+ (-0.023435276 * float(x[189]))+ (-0.108607925 * float(x[190]))+ (-0.10670863 * float(x[191]))+ (0.0494019 * float(x[192]))+ (0.07301684 * float(x[193]))+ (-0.110203244 * float(x[194]))+ (0.039956328 * float(x[195]))+ (-0.054398507 * float(x[196]))+ (0.07552595 * float(x[197]))+ (-0.004183823 * float(x[198]))+ (0.1381576 * float(x[199])))+ ((0.085383974 * float(x[200]))+ (-0.042132728 * float(x[201]))+ (0.025283579 * float(x[202]))+ (0.14029925 * float(x[203]))+ (0.0037537606 * float(x[204]))+ (0.1192041 * float(x[205]))+ (0.07705243 * float(x[206]))+ (0.11017648 * float(x[207]))+ (0.017837467 * float(x[208]))+ (-0.010269082 * float(x[209]))+ (-0.0026964424 * float(x[210]))+ (-0.026615636 * float(x[211]))+ (-0.029257718 * float(x[212]))+ (0.148198 * float(x[213]))+ (0.08177377 * float(x[214]))+ (-0.044608254 * float(x[215]))+ (-0.00554137 * float(x[216]))+ (0.059072338 * float(x[217]))+ (-0.0425699 * float(x[218]))+ (-0.02565403 * float(x[219]))+ (0.055288304 * float(x[220]))+ (-0.12553804 * float(x[221]))+ (0.036531594 * float(x[222]))+ (0.054129343 * float(x[223]))+ (-0.10658677 * float(x[224]))+ (0.09163769 * float(x[225]))+ (-0.040057797 * float(x[226]))+ (0.044181794 * float(x[227]))+ (0.13584232 * float(x[228]))+ (-0.030065779 * float(x[229]))+ (-0.046368 * float(x[230]))+ (0.030828053 * float(x[231]))+ (0.04328338 * float(x[232]))+ (0.09563362 * float(x[233]))+ (-0.04438005 * float(x[234]))+ (0.050450966 * float(x[235]))+ (0.019451234 * float(x[236]))+ (0.0030947044 * float(x[237]))+ (-0.11013856 * float(x[238]))+ (0.11608681 * float(x[239]))+ (0.071566425 * float(x[240]))+ (0.049222495 * float(x[241]))+ (0.0325285 * float(x[242]))+ (0.09247514 * float(x[243]))+ (0.0004993537 * float(x[244]))+ (0.11372378 * float(x[245]))+ (0.12286963 * float(x[246]))+ (0.05658072 * float(x[247]))+ (0.09420116 * float(x[248]))+ (0.073161006 * float(x[249])))+ ((-0.10056984 * float(x[250]))+ (-0.06563383 * float(x[251]))+ (-0.011751544 * float(x[252]))+ (0.0076580686 * float(x[253]))+ (0.09951105 * float(x[254]))+ (0.12281571 * float(x[255]))+ (-0.007296312 * float(x[256]))+ (-0.053701658 * float(x[257]))+ (0.12019797 * float(x[258]))+ (-0.0026772975 * float(x[259]))+ (-0.07859088 * float(x[260]))+ (-0.005975874 * float(x[261]))+ (0.07287847 * float(x[262]))+ (0.055457097 * float(x[263]))+ (0.14810435 * float(x[264]))+ (-0.03819897 * float(x[265]))+ (0.006199449 * float(x[266]))+ (0.03910342 * float(x[267]))+ (0.024495848 * float(x[268]))+ (0.14621413 * float(x[269]))+ (0.052960824 * float(x[270]))+ (0.07091204 * float(x[271]))+ (0.02298878 * float(x[272]))+ (0.063960046 * float(x[273]))+ (-0.017081093 * float(x[274]))+ (0.07788186 * float(x[275]))+ (0.01833992 * float(x[276]))+ (-0.0196188 * float(x[277]))+ (0.14291094 * float(x[278]))+ (0.056743454 * float(x[279]))+ (0.081500046 * float(x[280]))+ (0.016037256 * float(x[281]))+ (0.054300275 * float(x[282]))+ (-0.036095873 * float(x[283]))+ (0.059407283 * float(x[284]))+ (0.033046495 * float(x[285]))+ (0.028530596 * float(x[286]))+ (-0.016458021 * float(x[287]))+ (-0.120368406 * float(x[288]))+ (0.009732618 * float(x[289]))+ (0.093922734 * float(x[290]))+ (0.058763955 * float(x[291]))+ (0.102539934 * float(x[292]))+ (-0.11865355 * float(x[293]))+ (0.13659827 * float(x[294]))+ (-0.07350938 * float(x[295]))+ (-0.013155951 * float(x[296]))+ (0.0470998 * float(x[297]))+ (0.04991994 * float(x[298]))+ (0.06925234 * float(x[299])))+ ((0.03320379 * float(x[300]))+ (0.07340944 * float(x[301]))+ (0.06888528 * float(x[302]))+ (0.03127387 * float(x[303]))+ (-0.021483833 * float(x[304]))+ (-0.04973945 * float(x[305]))+ (0.13494062 * float(x[306]))+ (-0.00882635 * float(x[307]))+ (0.022596793 * float(x[308]))+ (0.09840956 * float(x[309]))+ (-0.015206009 * float(x[310]))+ (-0.012003989 * float(x[311]))+ (0.058747135 * float(x[312]))+ (0.049354464 * float(x[313]))+ (0.14373016 * float(x[314]))+ (-0.05559116 * float(x[315]))+ (0.042461805 * float(x[316]))+ (-0.026116291 * float(x[317]))+ (0.13204728 * float(x[318]))+ (-0.0034378825 * float(x[319]))+ (0.11068943 * float(x[320]))+ (-0.036144894 * float(x[321]))+ (0.14574274 * float(x[322]))+ (-0.026740512 * float(x[323]))+ (0.026770461 * float(x[324]))+ (0.020167004 * float(x[325]))+ (0.042834822 * float(x[326]))+ (0.097532704 * float(x[327]))+ (-0.03549379 * float(x[328]))+ (0.07482575 * float(x[329]))+ (0.10256149 * float(x[330]))+ (0.09804109 * float(x[331]))+ (0.0035072386 * float(x[332]))+ (0.05038341 * float(x[333]))+ (-0.051060632 * float(x[334]))+ (-0.04022588 * float(x[335]))+ (-0.027238132 * float(x[336]))+ (0.0912331 * float(x[337]))+ (0.13092287 * float(x[338]))+ (0.0015176414 * float(x[339]))+ (0.11839175 * float(x[340]))+ (0.01937334 * float(x[341]))+ (0.040670626 * float(x[342]))+ (-0.046334263 * float(x[343]))+ (0.0079949 * float(x[344]))+ (0.02291907 * float(x[345]))+ (0.029801693 * float(x[346]))+ (0.14171758 * float(x[347]))+ (0.11525041 * float(x[348]))+ (0.083892256 * float(x[349])))+ ((0.060393665 * float(x[350]))+ (0.046279266 * float(x[351]))+ (0.026022358 * float(x[352]))+ (0.10508171 * float(x[353]))+ (-0.034228727 * float(x[354]))+ (-0.027250238 * float(x[355]))+ (-0.037413996 * float(x[356]))+ (0.008546932 * float(x[357]))+ (-0.0026664084 * float(x[358]))+ (-0.050977863 * float(x[359]))+ (0.12439004 * float(x[360]))+ (0.12398402 * float(x[361]))+ (-0.006364316 * float(x[362]))+ (-0.04207286 * float(x[363]))+ (0.028283568 * float(x[364]))+ (0.13793758 * float(x[365]))+ (-0.028988339 * float(x[366]))+ (-0.11445859 * float(x[367]))+ (0.029350374 * float(x[368]))+ (0.1328478 * float(x[369]))+ (0.14409034 * float(x[370]))+ (0.023333255 * float(x[371]))+ (0.0040564938 * float(x[372]))+ (0.02201142 * float(x[373]))+ (0.037875365 * float(x[374]))+ (0.13524501 * float(x[375]))+ (-0.029662805 * float(x[376]))+ (-0.037025534 * float(x[377]))+ (0.041154277 * float(x[378]))+ (-0.0486105 * float(x[379]))+ (0.1478191 * float(x[380]))+ (0.017607566 * float(x[381]))+ (0.053306755 * float(x[382]))+ (-0.043685105 * float(x[383]))+ (-0.020751135 * float(x[384]))+ (0.11212956 * float(x[385]))+ (0.0058483644 * float(x[386]))+ (-0.03871429 * float(x[387]))+ (-0.025279962 * float(x[388]))+ (-0.019321762 * float(x[389]))+ (0.101703316 * float(x[390]))+ (0.108985774 * float(x[391]))+ (0.05764327 * float(x[392]))+ (0.103858314 * float(x[393]))+ (0.119846486 * float(x[394]))+ (0.078257196 * float(x[395]))+ (0.04876715 * float(x[396]))+ (0.076738246 * float(x[397]))+ (0.141533 * float(x[398]))+ (0.097651474 * float(x[399])))+ ((0.022003012 * float(x[400]))+ (0.1449907 * float(x[401]))+ (0.03441063 * float(x[402]))+ (0.07647989 * float(x[403]))+ (0.0046819113 * float(x[404]))+ (0.12152786 * float(x[405]))+ (0.13413773 * float(x[406]))+ (0.13438082 * float(x[407]))+ (-0.011792724 * float(x[408]))+ (-0.032727808 * float(x[409]))+ (-0.042305853 * float(x[410]))+ (0.07666401 * float(x[411]))+ (0.14409441 * float(x[412]))+ (-0.0038281875 * float(x[413]))+ (0.018094387 * float(x[414]))+ (0.02136232 * float(x[415]))+ (-0.05100903 * float(x[416]))+ (0.11051378 * float(x[417]))+ (0.0473516 * float(x[418]))+ (0.10984107 * float(x[419]))+ (0.0025869966 * float(x[420]))+ (-0.01491851 * float(x[421]))+ (-0.020766236 * float(x[422]))+ (-0.05446081 * float(x[423]))+ (-0.05222603 * float(x[424]))+ (0.12075275 * float(x[425]))+ (0.0064543965 * float(x[426]))+ (0.104538515 * float(x[427]))+ (0.114277475 * float(x[428]))+ (0.012925778 * float(x[429]))+ (-0.007061758 * float(x[430]))+ (-0.018042073 * float(x[431]))+ (0.09956131 * float(x[432]))+ (0.1375457 * float(x[433]))+ (0.0764261 * float(x[434]))+ (0.004080963 * float(x[435]))+ (0.11807534 * float(x[436]))+ (0.14125046 * float(x[437]))+ (0.044859886 * float(x[438]))+ (0.07147754 * float(x[439]))+ (0.04342061 * float(x[440]))+ (0.026862264 * float(x[441]))+ (0.056011118 * float(x[442]))+ (0.13573189 * float(x[443]))+ (0.10640216 * float(x[444]))+ (-0.0364918 * float(x[445]))+ (-0.0767127 * float(x[446]))+ (-0.09759151 * float(x[447]))+ (0.06286169 * float(x[448]))+ (-0.04120615 * float(x[449])))+ ((0.030850863 * float(x[450]))+ (0.14479779 * float(x[451]))+ (-0.027920485 * float(x[452]))+ (0.013691182 * float(x[453]))+ (-0.01113093 * float(x[454]))+ (0.03403818 * float(x[455]))+ (0.037897628 * float(x[456]))+ (0.0029388135 * float(x[457]))+ (-0.023998182 * float(x[458]))+ (0.14091332 * float(x[459]))+ (0.07033911 * float(x[460]))+ (-0.02600297 * float(x[461]))+ (0.02566592 * float(x[462]))+ (0.0672264 * float(x[463]))+ (0.03518883 * float(x[464]))+ (0.06509122 * float(x[465]))+ (0.061504945 * float(x[466]))+ (-0.021388412 * float(x[467]))+ (0.062281344 * float(x[468]))+ (0.12793468 * float(x[469]))+ (0.12045985 * float(x[470]))+ (-0.02485466 * float(x[471]))+ (0.0015700695 * float(x[472]))+ (0.016716296 * float(x[473]))+ (0.0777509 * float(x[474]))+ (0.021459097 * float(x[475]))+ (0.0003975722 * float(x[476]))+ (0.008277298 * float(x[477]))+ (0.063172445 * float(x[478]))+ (0.03198216 * float(x[479]))+ (0.04671176 * float(x[480]))+ (0.0029880365 * float(x[481]))+ (0.10890271 * float(x[482]))+ (0.056519724 * float(x[483]))+ (0.14836389 * float(x[484]))+ (-0.016569877 * float(x[485]))+ (-0.01488919 * float(x[486]))+ (-0.0045034047 * float(x[487]))+ (0.14827853 * float(x[488]))+ (0.019414248 * float(x[489]))+ (0.0054521696 * float(x[490]))+ (0.13771988 * float(x[491]))+ (0.086521454 * float(x[492]))+ (0.082586706 * float(x[493]))+ (0.12110581 * float(x[494]))+ (-0.017517168 * float(x[495]))+ (0.088178724 * float(x[496]))+ (0.05248155 * float(x[497]))+ (0.025165861 * float(x[498]))+ (0.053513683 * float(x[499])))+ ((-0.03044262 * float(x[500]))+ (-0.0066252756 * float(x[501]))+ (0.014711852 * float(x[502]))+ (0.07744884 * float(x[503]))+ (0.0053896126 * float(x[504]))+ (-0.027514093 * float(x[505]))+ (0.090975374 * float(x[506]))+ (-0.012124128 * float(x[507]))+ (0.09760615 * float(x[508]))+ (0.13556749 * float(x[509]))+ (-0.068355806 * float(x[510]))+ (0.064824134 * float(x[511]))+ (-0.0900817 * float(x[512]))+ (0.104973175 * float(x[513]))+ (0.01576422 * float(x[514]))+ (0.13396913 * float(x[515]))+ (0.14088415 * float(x[516]))+ (0.07275931 * float(x[517]))+ (0.07569152 * float(x[518]))+ (0.015379748 * float(x[519]))+ (0.10484661 * float(x[520]))+ (-0.046383277 * float(x[521]))+ (0.063088484 * float(x[522]))+ (0.12392744 * float(x[523]))+ (0.13508846 * float(x[524]))+ (-0.09682266 * float(x[525]))+ (0.0029023844 * float(x[526]))+ (0.000111531146 * float(x[527]))+ (0.14239208 * float(x[528]))+ (-0.025335412 * float(x[529]))+ (0.12636602 * float(x[530]))+ (-0.039840728 * float(x[531]))+ (-0.022261336 * float(x[532]))+ (0.119557686 * float(x[533]))+ (0.0666145 * float(x[534]))+ (-0.012414869 * float(x[535]))+ (-0.07112635 * float(x[536]))+ (-0.054838352 * float(x[537]))+ (0.06228365 * float(x[538]))+ (0.063470945 * float(x[539]))+ (-0.015601288 * float(x[540]))+ (0.09949765 * float(x[541]))+ (0.01403258 * float(x[542]))+ (0.12140835 * float(x[543]))+ (-0.03784489 * float(x[544]))+ (0.0053633987 * float(x[545]))+ (0.08084772 * float(x[546]))+ (-0.055175047 * float(x[547]))+ (-0.034440722 * float(x[548]))+ (-0.14774023 * float(x[549])))+ ((0.103052 * float(x[550]))+ (0.017337754 * float(x[551]))+ (0.07319752 * float(x[552]))+ (0.09746411 * float(x[553]))+ (-0.023821887 * float(x[554]))+ (0.09432854 * float(x[555]))+ (0.078967944 * float(x[556]))+ (-0.12806264 * float(x[557]))+ (-0.010866865 * float(x[558]))+ (0.01833352 * float(x[559]))+ (-0.10549089 * float(x[560]))) + 0.4547036), 0)
    h_1 = max((((2.4746916 * float(x[0]))+ (19.882105 * float(x[1]))+ (-1.5709987 * float(x[2]))+ (0.8253601 * float(x[3]))+ (-0.51756227 * float(x[4]))+ (0.33074427 * float(x[5]))+ (0.5382605 * float(x[6]))+ (-0.25725287 * float(x[7]))+ (0.15103619 * float(x[8]))+ (1.6607491 * float(x[9]))+ (1.0631706 * float(x[10]))+ (-2.324661 * float(x[11]))+ (-0.8446388 * float(x[12]))+ (1.1639241 * float(x[13]))+ (-2.7471206 * float(x[14]))+ (0.5817077 * float(x[15]))+ (-0.2829062 * float(x[16]))+ (0.3969321 * float(x[17]))+ (1.4117639 * float(x[18]))+ (0.6966564 * float(x[19]))+ (-0.35406086 * float(x[20]))+ (-1.332369 * float(x[21]))+ (3.0041604 * float(x[22]))+ (-5.979203 * float(x[23]))+ (2.152513 * float(x[24]))+ (-3.5464349 * float(x[25]))+ (-0.1333349 * float(x[26]))+ (-7.6249256 * float(x[27]))+ (-5.4380364 * float(x[28]))+ (2.6074643 * float(x[29]))+ (-2.4707518 * float(x[30]))+ (-2.16297 * float(x[31]))+ (-7.936565 * float(x[32]))+ (4.255681 * float(x[33]))+ (-1.963829 * float(x[34]))+ (-2.850896 * float(x[35]))+ (-8.620437 * float(x[36]))+ (-5.0085583 * float(x[37]))+ (2.997357 * float(x[38]))+ (-1.1497885 * float(x[39]))+ (4.0114894 * float(x[40]))+ (-8.724478 * float(x[41]))+ (-3.2241535 * float(x[42]))+ (-0.031261694 * float(x[43]))+ (-0.8550093 * float(x[44]))+ (-1.3560538 * float(x[45]))+ (0.0024404044 * float(x[46]))+ (-0.69588006 * float(x[47]))+ (-1.1591498 * float(x[48]))+ (4.396006 * float(x[49])))+ ((-12.985873 * float(x[50]))+ (-1.3397977 * float(x[51]))+ (3.7667153 * float(x[52]))+ (-8.876563 * float(x[53]))+ (5.1167645 * float(x[54]))+ (2.0487776 * float(x[55]))+ (9.688948 * float(x[56]))+ (-9.351303 * float(x[57]))+ (0.78790927 * float(x[58]))+ (-0.218603 * float(x[59]))+ (0.04678123 * float(x[60]))+ (-0.7258973 * float(x[61]))+ (0.77186656 * float(x[62]))+ (-1.9880321 * float(x[63]))+ (3.7697985 * float(x[64]))+ (1.695165 * float(x[65]))+ (-0.7446874 * float(x[66]))+ (0.06574433 * float(x[67]))+ (0.5783753 * float(x[68]))+ (0.37826103 * float(x[69]))+ (0.21943289 * float(x[70]))+ (-1.0546615 * float(x[71]))+ (1.5888423 * float(x[72]))+ (-0.86364615 * float(x[73]))+ (0.9482864 * float(x[74]))+ (-0.9510422 * float(x[75]))+ (1.190582 * float(x[76]))+ (1.3390001 * float(x[77]))+ (-0.19023663 * float(x[78]))+ (0.48817235 * float(x[79]))+ (-0.7355831 * float(x[80]))+ (-9.777635 * float(x[81]))+ (5.517546 * float(x[82]))+ (0.6867866 * float(x[83]))+ (0.55608004 * float(x[84]))+ (-0.16206436 * float(x[85]))+ (0.6502449 * float(x[86]))+ (0.7565738 * float(x[87]))+ (-0.23739891 * float(x[88]))+ (0.16318543 * float(x[89]))+ (-0.61269605 * float(x[90]))+ (-0.46115687 * float(x[91]))+ (-1.5494756 * float(x[92]))+ (-0.34633136 * float(x[93]))+ (-0.33609858 * float(x[94]))+ (0.34011653 * float(x[95]))+ (-0.12237856 * float(x[96]))+ (-0.09350973 * float(x[97]))+ (-0.1262107 * float(x[98]))+ (0.4930088 * float(x[99])))+ ((0.7929077 * float(x[100]))+ (-0.44079703 * float(x[101]))+ (-4.019329 * float(x[102]))+ (-0.5021766 * float(x[103]))+ (-7.6290965 * float(x[104]))+ (6.0568376 * float(x[105]))+ (-3.324161 * float(x[106]))+ (1.9721383 * float(x[107]))+ (3.6117175 * float(x[108]))+ (2.1598325 * float(x[109]))+ (1.0900041 * float(x[110]))+ (-4.921117 * float(x[111]))+ (3.0216155 * float(x[112]))+ (-0.88932455 * float(x[113]))+ (4.172244 * float(x[114]))+ (-3.3065934 * float(x[115]))+ (-3.3036017 * float(x[116]))+ (2.1109784 * float(x[117]))+ (-1.7986258 * float(x[118]))+ (2.013158 * float(x[119]))+ (14.546214 * float(x[120]))+ (11.337043 * float(x[121]))+ (3.4322796 * float(x[122]))+ (-0.1627336 * float(x[123]))+ (0.42961377 * float(x[124]))+ (0.017066743 * float(x[125]))+ (0.1698183 * float(x[126]))+ (0.2601851 * float(x[127]))+ (-0.70759195 * float(x[128]))+ (0.33414367 * float(x[129]))+ (0.1587301 * float(x[130]))+ (-0.01050616 * float(x[131]))+ (1.0357525 * float(x[132]))+ (-0.4828317 * float(x[133]))+ (-1.4669788 * float(x[134]))+ (-2.3479724 * float(x[135]))+ (-0.6410613 * float(x[136]))+ (0.05516003 * float(x[137]))+ (-0.11940213 * float(x[138]))+ (-0.113032065 * float(x[139]))+ (-0.4791023 * float(x[140]))+ (-1.1668234 * float(x[141]))+ (3.4246056 * float(x[142]))+ (-0.122282036 * float(x[143]))+ (0.45028764 * float(x[144]))+ (-7.855575 * float(x[145]))+ (-5.261985 * float(x[146]))+ (1.6301119 * float(x[147]))+ (-1.9181943 * float(x[148]))+ (-0.037723474 * float(x[149])))+ ((-2.6160994 * float(x[150]))+ (-3.0665796 * float(x[151]))+ (-1.2671951 * float(x[152]))+ (3.8247716 * float(x[153]))+ (3.7543066 * float(x[154]))+ (1.7229025 * float(x[155]))+ (7.7399592 * float(x[156]))+ (1.7153906 * float(x[157]))+ (-2.3551016 * float(x[158]))+ (-0.35351777 * float(x[159]))+ (5.5485973 * float(x[160]))+ (1.4503934 * float(x[161]))+ (-4.681667 * float(x[162]))+ (1.0425751 * float(x[163]))+ (0.37367484 * float(x[164]))+ (0.67341274 * float(x[165]))+ (1.4085196 * float(x[166]))+ (0.25307694 * float(x[167]))+ (0.3669007 * float(x[168]))+ (0.47952724 * float(x[169]))+ (0.38142705 * float(x[170]))+ (-0.1257004 * float(x[171]))+ (0.19437799 * float(x[172]))+ (-0.083172746 * float(x[173]))+ (-0.7128547 * float(x[174]))+ (0.6806496 * float(x[175]))+ (0.10525386 * float(x[176]))+ (-0.23614193 * float(x[177]))+ (-0.11066121 * float(x[178]))+ (1.7349355 * float(x[179]))+ (0.1454637 * float(x[180]))+ (0.07674697 * float(x[181]))+ (21.642767 * float(x[182]))+ (-8.640179 * float(x[183]))+ (-1.370746 * float(x[184]))+ (-6.0896397 * float(x[185]))+ (-12.423636 * float(x[186]))+ (-8.730058 * float(x[187]))+ (-0.962954 * float(x[188]))+ (2.2244642 * float(x[189]))+ (0.9207274 * float(x[190]))+ (0.19519308 * float(x[191]))+ (4.2355795 * float(x[192]))+ (-7.3081107 * float(x[193]))+ (-6.448293 * float(x[194]))+ (3.7524974 * float(x[195]))+ (4.716172 * float(x[196]))+ (-3.0172932 * float(x[197]))+ (-0.15344703 * float(x[198]))+ (1.3032777 * float(x[199])))+ ((0.65770286 * float(x[200]))+ (0.28029776 * float(x[201]))+ (0.46161926 * float(x[202]))+ (0.28780594 * float(x[203]))+ (0.619947 * float(x[204]))+ (0.62556404 * float(x[205]))+ (0.3928708 * float(x[206]))+ (1.2045091 * float(x[207]))+ (-0.2975492 * float(x[208]))+ (2.2697217 * float(x[209]))+ (0.9437084 * float(x[210]))+ (-2.3205602 * float(x[211]))+ (-1.0690105 * float(x[212]))+ (0.6246881 * float(x[213]))+ (0.38536286 * float(x[214]))+ (0.4669459 * float(x[215]))+ (0.23643546 * float(x[216]))+ (0.5976859 * float(x[217]))+ (0.6620721 * float(x[218]))+ (0.33326927 * float(x[219]))+ (1.1989859 * float(x[220]))+ (-0.26913595 * float(x[221]))+ (2.2416074 * float(x[222]))+ (0.9022494 * float(x[223]))+ (-2.3601468 * float(x[224]))+ (-1.1390941 * float(x[225]))+ (0.48330835 * float(x[226]))+ (0.38795882 * float(x[227]))+ (0.39320764 * float(x[228]))+ (0.48386922 * float(x[229]))+ (0.20395826 * float(x[230]))+ (0.5196819 * float(x[231]))+ (-0.007259608 * float(x[232]))+ (0.18925786 * float(x[233]))+ (-0.7257444 * float(x[234]))+ (2.1749032 * float(x[235]))+ (-1.278243 * float(x[236]))+ (2.0732167 * float(x[237]))+ (0.67033774 * float(x[238]))+ (-2.0656657 * float(x[239]))+ (-0.011196959 * float(x[240]))+ (-0.7469927 * float(x[241]))+ (-0.1063 * float(x[242]))+ (-2.1357665 * float(x[243]))+ (-2.1318395 * float(x[244]))+ (-0.40342745 * float(x[245]))+ (-3.5285041 * float(x[246]))+ (-6.6115756 * float(x[247]))+ (-0.04972607 * float(x[248]))+ (0.5617605 * float(x[249])))+ ((-1.9004128 * float(x[250]))+ (-0.1565918 * float(x[251]))+ (0.53717095 * float(x[252]))+ (0.39486402 * float(x[253]))+ (0.36555055 * float(x[254]))+ (0.3692977 * float(x[255]))+ (0.5686766 * float(x[256]))+ (0.6385308 * float(x[257]))+ (-0.07779267 * float(x[258]))+ (0.3570956 * float(x[259]))+ (0.45125782 * float(x[260]))+ (-0.84854156 * float(x[261]))+ (-2.0519814 * float(x[262]))+ (-0.13426252 * float(x[263]))+ (-0.56379807 * float(x[264]))+ (0.9166078 * float(x[265]))+ (-0.11317734 * float(x[266]))+ (-0.26610687 * float(x[267]))+ (0.7529669 * float(x[268]))+ (-0.73378587 * float(x[269]))+ (0.45714247 * float(x[270]))+ (0.7022618 * float(x[271]))+ (-0.015477824 * float(x[272]))+ (-0.32608113 * float(x[273]))+ (0.47786522 * float(x[274]))+ (-1.4403172 * float(x[275]))+ (0.13642995 * float(x[276]))+ (1.2513283 * float(x[277]))+ (-2.1739228 * float(x[278]))+ (-0.43581522 * float(x[279]))+ (0.3576953 * float(x[280]))+ (-0.32560375 * float(x[281]))+ (0.2994388 * float(x[282]))+ (0.7162806 * float(x[283]))+ (0.15887651 * float(x[284]))+ (-0.15397887 * float(x[285]))+ (-0.7195949 * float(x[286]))+ (3.292467 * float(x[287]))+ (-3.178615 * float(x[288]))+ (-1.1388403 * float(x[289]))+ (-0.3123826 * float(x[290]))+ (-2.9041412 * float(x[291]))+ (-0.08168578 * float(x[292]))+ (-0.8728494 * float(x[293]))+ (-2.7273777 * float(x[294]))+ (0.9272967 * float(x[295]))+ (0.4144385 * float(x[296]))+ (-2.25871 * float(x[297]))+ (2.8998756 * float(x[298]))+ (-1.3285953 * float(x[299])))+ ((0.7348664 * float(x[300]))+ (2.0636234 * float(x[301]))+ (-0.36953095 * float(x[302]))+ (0.13598661 * float(x[303]))+ (-0.13655835 * float(x[304]))+ (-0.1848366 * float(x[305]))+ (-0.1366402 * float(x[306]))+ (-0.10772529 * float(x[307]))+ (-0.054115463 * float(x[308]))+ (0.17138793 * float(x[309]))+ (-0.25810018 * float(x[310]))+ (-0.076923795 * float(x[311]))+ (-0.03541071 * float(x[312]))+ (-0.09236509 * float(x[313]))+ (-0.2837471 * float(x[314]))+ (0.0004988581 * float(x[315]))+ (0.24423334 * float(x[316]))+ (0.058753412 * float(x[317]))+ (0.053213876 * float(x[318]))+ (0.14445166 * float(x[319]))+ (-0.12787066 * float(x[320]))+ (0.049569003 * float(x[321]))+ (-0.2237337 * float(x[322]))+ (-0.047980554 * float(x[323]))+ (0.3228527 * float(x[324]))+ (0.12118355 * float(x[325]))+ (-0.117987946 * float(x[326]))+ (-0.21997374 * float(x[327]))+ (0.2335454 * float(x[328]))+ (0.060140174 * float(x[329]))+ (0.8831411 * float(x[330]))+ (-0.063547276 * float(x[331]))+ (-0.21229954 * float(x[332]))+ (-0.08872408 * float(x[333]))+ (-0.06318563 * float(x[334]))+ (-0.009939501 * float(x[335]))+ (-0.047397595 * float(x[336]))+ (0.03456329 * float(x[337]))+ (0.73843974 * float(x[338]))+ (-0.19168968 * float(x[339]))+ (-0.025349515 * float(x[340]))+ (-0.018935025 * float(x[341]))+ (0.7713812 * float(x[342]))+ (-0.024228754 * float(x[343]))+ (0.7003292 * float(x[344]))+ (0.47463235 * float(x[345]))+ (-0.3911419 * float(x[346]))+ (0.748712 * float(x[347]))+ (0.58703226 * float(x[348]))+ (-0.036668923 * float(x[349])))+ ((0.81792825 * float(x[350]))+ (0.47700304 * float(x[351]))+ (-0.20942536 * float(x[352]))+ (0.6144663 * float(x[353]))+ (0.47874853 * float(x[354]))+ (-0.0959596 * float(x[355]))+ (0.856302 * float(x[356]))+ (0.7355932 * float(x[357]))+ (-0.23524387 * float(x[358]))+ (0.37137493 * float(x[359]))+ (-0.13985662 * float(x[360]))+ (-0.08378062 * float(x[361]))+ (-0.27335495 * float(x[362]))+ (0.8488899 * float(x[363]))+ (0.4098557 * float(x[364]))+ (-0.6868462 * float(x[365]))+ (3.939871 * float(x[366]))+ (-0.12300503 * float(x[367]))+ (-2.9022708 * float(x[368]))+ (0.79140466 * float(x[369]))+ (0.78435534 * float(x[370]))+ (0.17766339 * float(x[371]))+ (-1.6004742 * float(x[372]))+ (2.1498697 * float(x[373]))+ (3.3877473 * float(x[374]))+ (-3.1232657 * float(x[375]))+ (3.1254473 * float(x[376]))+ (1.2909204 * float(x[377]))+ (1.2531056 * float(x[378]))+ (1.3024573 * float(x[379]))+ (1.8604202 * float(x[380]))+ (-0.42231026 * float(x[381]))+ (0.4257871 * float(x[382]))+ (-0.14263096 * float(x[383]))+ (-0.18491928 * float(x[384]))+ (-0.16871972 * float(x[385]))+ (-0.30956766 * float(x[386]))+ (-0.3328417 * float(x[387]))+ (-0.020398285 * float(x[388]))+ (0.04116423 * float(x[389]))+ (-0.16109395 * float(x[390]))+ (-0.29127994 * float(x[391]))+ (-0.16255303 * float(x[392]))+ (0.057431336 * float(x[393]))+ (-0.1478557 * float(x[394]))+ (-0.4435312 * float(x[395]))+ (-0.051333282 * float(x[396]))+ (0.1371167 * float(x[397]))+ (0.006168378 * float(x[398]))+ (-0.18652222 * float(x[399])))+ ((-0.15477544 * float(x[400]))+ (-0.06861482 * float(x[401]))+ (-0.042973123 * float(x[402]))+ (-0.13632582 * float(x[403]))+ (0.13178676 * float(x[404]))+ (-0.14260286 * float(x[405]))+ (-0.029205518 * float(x[406]))+ (0.061823107 * float(x[407]))+ (0.035169102 * float(x[408]))+ (0.24600805 * float(x[409]))+ (-0.050619952 * float(x[410]))+ (-0.28722546 * float(x[411]))+ (-0.066227645 * float(x[412]))+ (-0.17267339 * float(x[413]))+ (-0.25707656 * float(x[414]))+ (-0.09712186 * float(x[415]))+ (0.011156791 * float(x[416]))+ (-0.089027435 * float(x[417]))+ (-0.18588902 * float(x[418]))+ (-0.066931374 * float(x[419]))+ (-0.19378385 * float(x[420]))+ (-0.2356024 * float(x[421]))+ (-0.1601179 * float(x[422]))+ (0.48376626 * float(x[423]))+ (0.080044255 * float(x[424]))+ (0.24377577 * float(x[425]))+ (-0.26855403 * float(x[426]))+ (0.7181472 * float(x[427]))+ (-0.0038564252 * float(x[428]))+ (0.11373177 * float(x[429]))+ (0.48923838 * float(x[430]))+ (0.3910422 * float(x[431]))+ (-0.39356267 * float(x[432]))+ (0.35076186 * float(x[433]))+ (-0.2873177 * float(x[434]))+ (0.30095252 * float(x[435]))+ (-1.79832 * float(x[436]))+ (-0.5056441 * float(x[437]))+ (0.3419181 * float(x[438]))+ (-0.45905274 * float(x[439]))+ (0.1914849 * float(x[440]))+ (0.14019112 * float(x[441]))+ (0.6825775 * float(x[442]))+ (0.2150455 * float(x[443]))+ (-0.35201916 * float(x[444]))+ (8.239945 * float(x[445]))+ (-3.239563 * float(x[446]))+ (-2.407959 * float(x[447]))+ (-1.7894467 * float(x[448]))+ (-0.21576495 * float(x[449])))+ ((1.6539581 * float(x[450]))+ (-4.7488947 * float(x[451]))+ (1.0625539 * float(x[452]))+ (0.28768012 * float(x[453]))+ (5.4680986 * float(x[454]))+ (-1.9042152 * float(x[455]))+ (-4.1898866 * float(x[456]))+ (1.6578755 * float(x[457]))+ (-3.981207 * float(x[458]))+ (0.8319901 * float(x[459]))+ (-0.4476568 * float(x[460]))+ (0.22778627 * float(x[461]))+ (0.18839146 * float(x[462]))+ (0.2386486 * float(x[463]))+ (0.15877232 * float(x[464]))+ (-0.0023920785 * float(x[465]))+ (0.1965973 * float(x[466]))+ (-0.010353988 * float(x[467]))+ (-0.47350967 * float(x[468]))+ (0.20478232 * float(x[469]))+ (0.1680001 * float(x[470]))+ (0.0077748923 * float(x[471]))+ (-0.40208352 * float(x[472]))+ (0.20899098 * float(x[473]))+ (0.5036718 * float(x[474]))+ (-0.27345788 * float(x[475]))+ (-0.13454688 * float(x[476]))+ (-0.30216002 * float(x[477]))+ (-0.23424643 * float(x[478]))+ (-0.11888548 * float(x[479]))+ (-0.19736363 * float(x[480]))+ (-0.20563579 * float(x[481]))+ (0.22122365 * float(x[482]))+ (-0.28731808 * float(x[483]))+ (-0.13233854 * float(x[484]))+ (-0.29128474 * float(x[485]))+ (0.056429517 * float(x[486]))+ (-0.28320917 * float(x[487]))+ (0.31259173 * float(x[488]))+ (-0.2737986 * float(x[489]))+ (-0.19087048 * float(x[490]))+ (-0.12603915 * float(x[491]))+ (-0.13267656 * float(x[492]))+ (-0.26751047 * float(x[493]))+ (-0.14955272 * float(x[494]))+ (-0.38697767 * float(x[495]))+ (0.16623458 * float(x[496]))+ (-0.14895248 * float(x[497]))+ (-0.23431185 * float(x[498]))+ (-0.20482202 * float(x[499])))+ ((0.11187916 * float(x[500]))+ (-0.008324148 * float(x[501]))+ (-0.63751394 * float(x[502]))+ (0.53490794 * float(x[503]))+ (-0.84257317 * float(x[504]))+ (1.0121663 * float(x[505]))+ (-1.0403502 * float(x[506]))+ (-0.7249252 * float(x[507]))+ (-0.047041867 * float(x[508]))+ (-0.39905176 * float(x[509]))+ (-0.37234318 * float(x[510]))+ (1.6601378 * float(x[511]))+ (-2.4516816 * float(x[512]))+ (-3.491763 * float(x[513]))+ (2.3794003 * float(x[514]))+ (0.16566595 * float(x[515]))+ (0.588478 * float(x[516]))+ (0.60320675 * float(x[517]))+ (0.17191629 * float(x[518]))+ (0.5059137 * float(x[519]))+ (0.16311672 * float(x[520]))+ (-0.09730854 * float(x[521]))+ (0.174441 * float(x[522]))+ (-0.26286873 * float(x[523]))+ (0.70397365 * float(x[524]))+ (1.5587871 * float(x[525]))+ (2.1630983 * float(x[526]))+ (-0.10404846 * float(x[527]))+ (0.116135776 * float(x[528]))+ (0.13206668 * float(x[529]))+ (0.32674497 * float(x[530]))+ (-0.64280117 * float(x[531]))+ (-1.4538851 * float(x[532]))+ (0.136583 * float(x[533]))+ (0.026153227 * float(x[534]))+ (-0.20861521 * float(x[535]))+ (-0.7800639 * float(x[536]))+ (0.46995082 * float(x[537]))+ (2.695161 * float(x[538]))+ (2.040641 * float(x[539]))+ (-0.46649632 * float(x[540]))+ (0.15877724 * float(x[541]))+ (0.71059704 * float(x[542]))+ (0.5024703 * float(x[543]))+ (0.647043 * float(x[544]))+ (-0.025279554 * float(x[545]))+ (0.1871005 * float(x[546]))+ (-0.21158564 * float(x[547]))+ (0.41033608 * float(x[548]))+ (3.8296392 * float(x[549])))+ ((0.7373563 * float(x[550]))+ (-0.7583665 * float(x[551]))+ (-0.35253754 * float(x[552]))+ (0.6239549 * float(x[553]))+ (-2.9690118 * float(x[554]))+ (0.06133596 * float(x[555]))+ (0.35206598 * float(x[556]))+ (1.3067528 * float(x[557]))+ (-2.4958858 * float(x[558]))+ (3.6939409 * float(x[559]))+ (3.7037323 * float(x[560]))) + 0.37756142), 0)
    h_2 = max((((2.5169086 * float(x[0]))+ (1.9220712 * float(x[1]))+ (-0.5453239 * float(x[2]))+ (3.4925427 * float(x[3]))+ (-1.197721 * float(x[4]))+ (-0.40468958 * float(x[5]))+ (2.2005055 * float(x[6]))+ (-0.7664702 * float(x[7]))+ (-0.37409338 * float(x[8]))+ (6.6746354 * float(x[9]))+ (1.9481341 * float(x[10]))+ (0.29145572 * float(x[11]))+ (-0.040966652 * float(x[12]))+ (2.0240939 * float(x[13]))+ (-1.423456 * float(x[14]))+ (1.4473614 * float(x[15]))+ (1.7392815 * float(x[16]))+ (-0.09489122 * float(x[17]))+ (-0.15713964 * float(x[18]))+ (0.35152245 * float(x[19]))+ (-0.38721466 * float(x[20]))+ (1.2563932 * float(x[21]))+ (-2.6206834 * float(x[22]))+ (-2.5478482 * float(x[23]))+ (-2.8821127 * float(x[24]))+ (2.3993766 * float(x[25]))+ (-0.912525 * float(x[26]))+ (-0.70924175 * float(x[27]))+ (5.2449236 * float(x[28]))+ (1.6174347 * float(x[29]))+ (-0.9803501 * float(x[30]))+ (2.9483936 * float(x[31]))+ (0.63295627 * float(x[32]))+ (0.75416714 * float(x[33]))+ (-0.19496347 * float(x[34]))+ (2.1482215 * float(x[35]))+ (1.4903902 * float(x[36]))+ (-6.004136 * float(x[37]))+ (1.085691 * float(x[38]))+ (4.337464 * float(x[39]))+ (3.8435388 * float(x[40]))+ (-1.329409 * float(x[41]))+ (0.75317764 * float(x[42]))+ (-1.3965185 * float(x[43]))+ (-0.55839676 * float(x[44]))+ (-0.67885596 * float(x[45]))+ (-1.4639118 * float(x[46]))+ (-0.45426375 * float(x[47]))+ (-0.5632273 * float(x[48]))+ (3.632454 * float(x[49])))+ ((-1.5091221 * float(x[50]))+ (1.5330596 * float(x[51]))+ (4.1009293 * float(x[52]))+ (-0.6475434 * float(x[53]))+ (2.1491687 * float(x[54]))+ (14.87992 * float(x[55]))+ (2.4406376 * float(x[56]))+ (-1.1783136 * float(x[57]))+ (-2.682549 * float(x[58]))+ (-1.3960894 * float(x[59]))+ (-0.082751684 * float(x[60]))+ (-0.6036933 * float(x[61]))+ (3.1499941 * float(x[62]))+ (-1.7637684 * float(x[63]))+ (-1.5190116 * float(x[64]))+ (-1.4927918 * float(x[65]))+ (1.5871531 * float(x[66]))+ (-1.6043512 * float(x[67]))+ (1.5706259 * float(x[68]))+ (1.1131418 * float(x[69]))+ (-0.91110414 * float(x[70]))+ (0.6213317 * float(x[71]))+ (-0.021703828 * float(x[72]))+ (-0.11229482 * float(x[73]))+ (-0.023151705 * float(x[74]))+ (0.3421301 * float(x[75]))+ (-0.44834328 * float(x[76]))+ (3.1935453 * float(x[77]))+ (1.4395983 * float(x[78]))+ (-0.4852607 * float(x[79]))+ (-1.6912602 * float(x[80]))+ (0.25879374 * float(x[81]))+ (4.41046 * float(x[82]))+ (0.070362434 * float(x[83]))+ (-2.134259 * float(x[84]))+ (-0.46257067 * float(x[85]))+ (0.25200522 * float(x[86]))+ (-2.1559083 * float(x[87]))+ (-0.48422372 * float(x[88]))+ (0.8659826 * float(x[89]))+ (0.16212848 * float(x[90]))+ (0.34325516 * float(x[91]))+ (-0.36607352 * float(x[92]))+ (1.1024805 * float(x[93]))+ (0.7034615 * float(x[94]))+ (-0.6042938 * float(x[95]))+ (0.12787104 * float(x[96]))+ (0.05113724 * float(x[97]))+ (0.16984625 * float(x[98]))+ (1.0949538 * float(x[99])))+ ((-1.8063388 * float(x[100]))+ (-0.6383156 * float(x[101]))+ (-2.8866088 * float(x[102]))+ (-0.6074061 * float(x[103]))+ (-4.67502 * float(x[104]))+ (-0.039775446 * float(x[105]))+ (-0.36567816 * float(x[106]))+ (-2.2822468 * float(x[107]))+ (-2.8482897 * float(x[108]))+ (-1.0250669 * float(x[109]))+ (-0.72252667 * float(x[110]))+ (0.38386175 * float(x[111]))+ (0.42859367 * float(x[112]))+ (-1.1625655 * float(x[113]))+ (-1.7839509 * float(x[114]))+ (-0.7137758 * float(x[115]))+ (-3.992251 * float(x[116]))+ (-11.446022 * float(x[117]))+ (-3.8492694 * float(x[118]))+ (-5.0734854 * float(x[119]))+ (3.7854497 * float(x[120]))+ (-1.6065881 * float(x[121]))+ (2.5899577 * float(x[122]))+ (0.97466934 * float(x[123]))+ (-0.6621134 * float(x[124]))+ (0.9684987 * float(x[125]))+ (0.7671907 * float(x[126]))+ (-0.45451143 * float(x[127]))+ (3.2358317 * float(x[128]))+ (0.914312 * float(x[129]))+ (-1.6600372 * float(x[130]))+ (-0.15112172 * float(x[131]))+ (-1.8644423 * float(x[132]))+ (-1.1548349 * float(x[133]))+ (1.5953267 * float(x[134]))+ (-0.17514205 * float(x[135]))+ (0.13110583 * float(x[136]))+ (0.31596136 * float(x[137]))+ (0.20330566 * float(x[138]))+ (0.63470066 * float(x[139]))+ (0.6414294 * float(x[140]))+ (5.8225446 * float(x[141]))+ (1.5962101 * float(x[142]))+ (5.565167 * float(x[143]))+ (-3.6532903 * float(x[144]))+ (-3.7496498 * float(x[145]))+ (3.1088815 * float(x[146]))+ (1.8313686 * float(x[147]))+ (-3.2451475 * float(x[148]))+ (2.9754932 * float(x[149])))+ ((-3.3034902 * float(x[150]))+ (-0.922947 * float(x[151]))+ (3.7640493 * float(x[152]))+ (2.4690866 * float(x[153]))+ (0.57660544 * float(x[154]))+ (-3.8094025 * float(x[155]))+ (-3.8899415 * float(x[156]))+ (6.114972 * float(x[157]))+ (-1.0221717 * float(x[158]))+ (-14.393662 * float(x[159]))+ (1.3810295 * float(x[160]))+ (-2.52053 * float(x[161]))+ (-3.4321563 * float(x[162]))+ (-1.9016321 * float(x[163]))+ (-0.7255054 * float(x[164]))+ (-1.5634271 * float(x[165]))+ (-1.7405468 * float(x[166]))+ (-0.61872566 * float(x[167]))+ (-2.3511815 * float(x[168]))+ (-1.60359 * float(x[169]))+ (-0.20116596 * float(x[170]))+ (-0.72010684 * float(x[171]))+ (-0.030459983 * float(x[172]))+ (0.3966871 * float(x[173]))+ (-0.5630212 * float(x[174]))+ (-1.225451 * float(x[175]))+ (-0.5406821 * float(x[176]))+ (0.52013546 * float(x[177]))+ (-0.1937427 * float(x[178]))+ (-1.2798384 * float(x[179]))+ (-1.0416768 * float(x[180]))+ (-2.9254029 * float(x[181]))+ (-0.44252914 * float(x[182]))+ (1.6519164 * float(x[183]))+ (-0.3104578 * float(x[184]))+ (-4.399358 * float(x[185]))+ (0.6756597 * float(x[186]))+ (0.035246916 * float(x[187]))+ (1.2103716 * float(x[188]))+ (3.4164996 * float(x[189]))+ (-0.65031636 * float(x[190]))+ (-1.6908224 * float(x[191]))+ (-2.7847617 * float(x[192]))+ (5.3108044 * float(x[193]))+ (4.7403383 * float(x[194]))+ (4.506955 * float(x[195]))+ (-1.260301 * float(x[196]))+ (-9.5683155 * float(x[197]))+ (-6.456294 * float(x[198]))+ (-4.387883 * float(x[199])))+ ((0.5690828 * float(x[200]))+ (3.3346522 * float(x[201]))+ (2.1048648 * float(x[202]))+ (1.9113739 * float(x[203]))+ (0.4868866 * float(x[204]))+ (0.5823652 * float(x[205]))+ (1.622161 * float(x[206]))+ (0.14043844 * float(x[207]))+ (-0.45809686 * float(x[208]))+ (-3.839 * float(x[209]))+ (2.273437 * float(x[210]))+ (3.1933146 * float(x[211]))+ (2.6176896 * float(x[212]))+ (0.66531867 * float(x[213]))+ (3.276642 * float(x[214]))+ (2.0024025 * float(x[215]))+ (1.9724405 * float(x[216]))+ (0.4594649 * float(x[217]))+ (0.596114 * float(x[218]))+ (1.4552723 * float(x[219]))+ (0.032557268 * float(x[220]))+ (-0.32775792 * float(x[221]))+ (-3.882879 * float(x[222]))+ (2.1380258 * float(x[223]))+ (3.276548 * float(x[224]))+ (2.7109895 * float(x[225]))+ (-0.8322175 * float(x[226]))+ (-0.84387505 * float(x[227]))+ (-1.1852398 * float(x[228]))+ (0.31118512 * float(x[229]))+ (0.68943095 * float(x[230]))+ (-0.9194738 * float(x[231]))+ (0.16255654 * float(x[232]))+ (-2.4247804 * float(x[233]))+ (1.0950651 * float(x[234]))+ (2.4552395 * float(x[235]))+ (0.937254 * float(x[236]))+ (-1.4360944 * float(x[237]))+ (0.5909325 * float(x[238]))+ (0.13188106 * float(x[239]))+ (0.41722232 * float(x[240]))+ (-0.290305 * float(x[241]))+ (2.2485285 * float(x[242]))+ (2.7413533 * float(x[243]))+ (0.07856184 * float(x[244]))+ (0.40283144 * float(x[245]))+ (-1.3279418 * float(x[246]))+ (-3.8584979 * float(x[247]))+ (0.7777453 * float(x[248]))+ (0.4780224 * float(x[249])))+ ((-0.6998736 * float(x[250]))+ (-2.29683 * float(x[251]))+ (-1.029523 * float(x[252]))+ (-0.22016154 * float(x[253]))+ (-0.33299264 * float(x[254]))+ (0.34789863 * float(x[255]))+ (-0.63635147 * float(x[256]))+ (-0.91500396 * float(x[257]))+ (0.11471456 * float(x[258]))+ (-0.95051664 * float(x[259]))+ (-0.28716943 * float(x[260]))+ (2.3463519 * float(x[261]))+ (-0.4035811 * float(x[262]))+ (-2.8645868 * float(x[263]))+ (-0.7896807 * float(x[264]))+ (2.1525488 * float(x[265]))+ (-2.07739 * float(x[266]))+ (-0.6550465 * float(x[267]))+ (3.7917094 * float(x[268]))+ (-0.7619871 * float(x[269]))+ (-0.44362497 * float(x[270]))+ (2.437587 * float(x[271]))+ (-1.4487599 * float(x[272]))+ (-0.76879466 * float(x[273]))+ (4.773447 * float(x[274]))+ (-0.02799149 * float(x[275]))+ (-0.0096986815 * float(x[276]))+ (-2.1547534 * float(x[277]))+ (-1.4990245 * float(x[278]))+ (-1.100961 * float(x[279]))+ (0.5217292 * float(x[280]))+ (2.030084 * float(x[281]))+ (0.39789727 * float(x[282]))+ (0.22572798 * float(x[283]))+ (1.9097849 * float(x[284]))+ (-0.4505211 * float(x[285]))+ (-1.4902586 * float(x[286]))+ (-0.07505884 * float(x[287]))+ (-2.1658351 * float(x[288]))+ (-1.036588 * float(x[289]))+ (2.6894488 * float(x[290]))+ (2.4405825 * float(x[291]))+ (-1.241385 * float(x[292]))+ (1.3491063 * float(x[293]))+ (-1.0402621 * float(x[294]))+ (-3.0603087 * float(x[295]))+ (-0.094524495 * float(x[296]))+ (0.72144943 * float(x[297]))+ (0.9298273 * float(x[298]))+ (1.1201494 * float(x[299])))+ ((0.37410405 * float(x[300]))+ (1.4533308 * float(x[301]))+ (2.771815 * float(x[302]))+ (-0.53657633 * float(x[303]))+ (0.47962904 * float(x[304]))+ (0.32631868 * float(x[305]))+ (0.39740473 * float(x[306]))+ (1.4494236 * float(x[307]))+ (0.7762429 * float(x[308]))+ (0.20602974 * float(x[309]))+ (2.1395278 * float(x[310]))+ (0.5453296 * float(x[311]))+ (0.7971391 * float(x[312]))+ (0.63036466 * float(x[313]))+ (1.9799793 * float(x[314]))+ (0.5456441 * float(x[315]))+ (1.3253646 * float(x[316]))+ (-1.1062105 * float(x[317]))+ (-0.08460381 * float(x[318]))+ (-0.10185434 * float(x[319]))+ (0.49658436 * float(x[320]))+ (-0.43859273 * float(x[321]))+ (-0.8203597 * float(x[322]))+ (-0.2763097 * float(x[323]))+ (0.46926376 * float(x[324]))+ (-0.008732596 * float(x[325]))+ (0.23879677 * float(x[326]))+ (-0.67788476 * float(x[327]))+ (0.3408866 * float(x[328]))+ (0.0977737 * float(x[329]))+ (0.13332494 * float(x[330]))+ (0.1729748 * float(x[331]))+ (0.08670608 * float(x[332]))+ (-0.04757377 * float(x[333]))+ (-0.5315516 * float(x[334]))+ (0.5867383 * float(x[335]))+ (0.44642034 * float(x[336]))+ (-0.6788038 * float(x[337]))+ (0.2701515 * float(x[338]))+ (-5.3983327e-05 * float(x[339]))+ (-0.23021036 * float(x[340]))+ (0.14806037 * float(x[341]))+ (0.1877828 * float(x[342]))+ (0.024293499 * float(x[343]))+ (0.57997465 * float(x[344]))+ (-1.456688 * float(x[345]))+ (-0.16071975 * float(x[346]))+ (-0.6124753 * float(x[347]))+ (-2.8064768 * float(x[348]))+ (-0.9337572 * float(x[349])))+ ((-0.67874086 * float(x[350]))+ (-2.2003624 * float(x[351]))+ (-0.92200357 * float(x[352]))+ (-1.3473423 * float(x[353]))+ (-2.6285508 * float(x[354]))+ (-0.92893934 * float(x[355]))+ (-1.6663421 * float(x[356]))+ (0.5966368 * float(x[357]))+ (0.23779987 * float(x[358]))+ (-0.16246831 * float(x[359]))+ (0.19114143 * float(x[360]))+ (-0.038646318 * float(x[361]))+ (0.31725433 * float(x[362]))+ (-0.48555744 * float(x[363]))+ (-0.5846654 * float(x[364]))+ (-0.21941122 * float(x[365]))+ (1.20666 * float(x[366]))+ (-1.0949702 * float(x[367]))+ (-2.4236138 * float(x[368]))+ (-2.4152012 * float(x[369]))+ (-0.8090258 * float(x[370]))+ (-3.8230624 * float(x[371]))+ (-2.7808669 * float(x[372]))+ (-3.1537716 * float(x[373]))+ (-6.1524596 * float(x[374]))+ (-1.1304785 * float(x[375]))+ (0.16174473 * float(x[376]))+ (-3.8761911 * float(x[377]))+ (-0.6829725 * float(x[378]))+ (-3.51227 * float(x[379]))+ (-1.3974905 * float(x[380]))+ (0.81531495 * float(x[381]))+ (-1.6875076 * float(x[382]))+ (0.63643223 * float(x[383]))+ (0.27910224 * float(x[384]))+ (0.53077245 * float(x[385]))+ (1.3960667 * float(x[386]))+ (0.898459 * float(x[387]))+ (-0.57304984 * float(x[388]))+ (-0.6818902 * float(x[389]))+ (0.7909003 * float(x[390]))+ (0.89393115 * float(x[391]))+ (0.9439648 * float(x[392]))+ (-0.2646985 * float(x[393]))+ (1.067202 * float(x[394]))+ (1.7805939 * float(x[395]))+ (-0.68130016 * float(x[396]))+ (-0.3674544 * float(x[397]))+ (-0.09232525 * float(x[398]))+ (0.36879107 * float(x[399])))+ ((0.47011736 * float(x[400]))+ (-0.5171522 * float(x[401]))+ (-0.13831386 * float(x[402]))+ (0.010348766 * float(x[403]))+ (-0.16252509 * float(x[404]))+ (0.7264863 * float(x[405]))+ (-0.6323153 * float(x[406]))+ (-0.12119114 * float(x[407]))+ (0.27229482 * float(x[408]))+ (-0.55117035 * float(x[409]))+ (0.75672877 * float(x[410]))+ (0.35420898 * float(x[411]))+ (0.035108496 * float(x[412]))+ (-0.27424777 * float(x[413]))+ (0.85962373 * float(x[414]))+ (0.8311 * float(x[415]))+ (-0.19656436 * float(x[416]))+ (0.56137323 * float(x[417]))+ (0.17419873 * float(x[418]))+ (0.14416562 * float(x[419]))+ (0.6792191 * float(x[420]))+ (0.6127654 * float(x[421]))+ (0.07053963 * float(x[422]))+ (0.595784 * float(x[423]))+ (-0.3605009 * float(x[424]))+ (0.8428732 * float(x[425]))+ (0.9190694 * float(x[426]))+ (-1.1386813 * float(x[427]))+ (0.56210935 * float(x[428]))+ (0.5070129 * float(x[429]))+ (-1.5849131 * float(x[430]))+ (-0.3485562 * float(x[431]))+ (1.6078821 * float(x[432]))+ (-0.101741545 * float(x[433]))+ (0.8694586 * float(x[434]))+ (-0.15886386 * float(x[435]))+ (-0.67474437 * float(x[436]))+ (-0.22608323 * float(x[437]))+ (0.44393998 * float(x[438]))+ (-0.26463485 * float(x[439]))+ (0.3756993 * float(x[440]))+ (0.6503639 * float(x[441]))+ (0.9535921 * float(x[442]))+ (0.96805596 * float(x[443]))+ (0.21327175 * float(x[444]))+ (0.13848767 * float(x[445]))+ (0.060850024 * float(x[446]))+ (2.8986197 * float(x[447]))+ (1.016656 * float(x[448]))+ (-1.435229 * float(x[449])))+ ((-2.7248523 * float(x[450]))+ (-3.769333 * float(x[451]))+ (-3.0149007 * float(x[452]))+ (-2.0068486 * float(x[453]))+ (-2.2797744 * float(x[454]))+ (-1.531677 * float(x[455]))+ (-2.8660133 * float(x[456]))+ (0.13536419 * float(x[457]))+ (0.22168428 * float(x[458]))+ (2.4956076 * float(x[459]))+ (-0.039724104 * float(x[460]))+ (-1.1243972 * float(x[461]))+ (-0.4889251 * float(x[462]))+ (-0.39695403 * float(x[463]))+ (-0.31374392 * float(x[464]))+ (-0.5120926 * float(x[465]))+ (0.1814295 * float(x[466]))+ (0.21446669 * float(x[467]))+ (-0.32408753 * float(x[468]))+ (-0.49489057 * float(x[469]))+ (-0.53269 * float(x[470]))+ (0.28131807 * float(x[471]))+ (-0.23815396 * float(x[472]))+ (-0.40817142 * float(x[473]))+ (-1.0389762 * float(x[474]))+ (0.68911356 * float(x[475]))+ (0.40845314 * float(x[476]))+ (0.62872237 * float(x[477]))+ (0.0329331 * float(x[478]))+ (0.40163577 * float(x[479]))+ (0.46348077 * float(x[480]))+ (0.63941896 * float(x[481]))+ (-0.060885146 * float(x[482]))+ (0.46869946 * float(x[483]))+ (0.08440812 * float(x[484]))+ (0.6170785 * float(x[485]))+ (0.22623803 * float(x[486]))+ (0.47102705 * float(x[487]))+ (0.30593646 * float(x[488]))+ (0.49870288 * float(x[489]))+ (0.9349453 * float(x[490]))+ (0.078082345 * float(x[491]))+ (0.35188383 * float(x[492]))+ (0.20641913 * float(x[493]))+ (0.15181224 * float(x[494]))+ (0.2550159 * float(x[495]))+ (0.36729595 * float(x[496]))+ (1.0367123 * float(x[497]))+ (0.20680627 * float(x[498]))+ (0.104865514 * float(x[499])))+ ((0.54617155 * float(x[500]))+ (0.21380922 * float(x[501]))+ (2.2757008 * float(x[502]))+ (2.9913766 * float(x[503]))+ (3.0088298 * float(x[504]))+ (2.4557557 * float(x[505]))+ (-1.7239522 * float(x[506]))+ (2.2591736 * float(x[507]))+ (1.9861585 * float(x[508]))+ (0.66929 * float(x[509]))+ (-0.19037643 * float(x[510]))+ (0.79321194 * float(x[511]))+ (3.4752722 * float(x[512]))+ (-0.64666134 * float(x[513]))+ (-1.3885396 * float(x[514]))+ (0.66846466 * float(x[515]))+ (-2.671583 * float(x[516]))+ (-1.871289 * float(x[517]))+ (-1.6024865 * float(x[518]))+ (-0.20387341 * float(x[519]))+ (0.61545575 * float(x[520]))+ (0.07315783 * float(x[521]))+ (0.014306145 * float(x[522]))+ (-0.74771607 * float(x[523]))+ (-0.2773809 * float(x[524]))+ (0.87280536 * float(x[525]))+ (-1.9083503 * float(x[526]))+ (-1.548365 * float(x[527]))+ (0.6044231 * float(x[528]))+ (0.0035584387 * float(x[529]))+ (-0.08504426 * float(x[530]))+ (-1.1049082 * float(x[531]))+ (1.274733 * float(x[532]))+ (0.6365002 * float(x[533]))+ (-0.29324895 * float(x[534]))+ (1.7788525 * float(x[535]))+ (-0.3439071 * float(x[536]))+ (-0.9203837 * float(x[537]))+ (-5.6187334 * float(x[538]))+ (-0.4617309 * float(x[539]))+ (-0.038939096 * float(x[540]))+ (0.43606374 * float(x[541]))+ (-0.8336727 * float(x[542]))+ (-0.48206273 * float(x[543]))+ (-1.2817292 * float(x[544]))+ (-0.08278947 * float(x[545]))+ (0.29568216 * float(x[546]))+ (0.35094324 * float(x[547]))+ (0.8344222 * float(x[548]))+ (-0.27366376 * float(x[549])))+ ((0.24424334 * float(x[550]))+ (-2.475054 * float(x[551]))+ (-1.0728565 * float(x[552]))+ (-0.6990733 * float(x[553]))+ (-3.282969 * float(x[554]))+ (-1.6929555 * float(x[555]))+ (0.9519662 * float(x[556]))+ (-1.5498558 * float(x[557]))+ (-2.0844574 * float(x[558]))+ (2.184074 * float(x[559]))+ (0.6102486 * float(x[560]))) + 0.47798786), 0)
    h_3 = max((((-0.039564822 * float(x[0]))+ (0.035910394 * float(x[1]))+ (0.103049725 * float(x[2]))+ (0.110139884 * float(x[3]))+ (-0.091190346 * float(x[4]))+ (-0.12913582 * float(x[5]))+ (0.059745535 * float(x[6]))+ (-0.034293 * float(x[7]))+ (-0.10768322 * float(x[8]))+ (0.027751759 * float(x[9]))+ (-0.058379002 * float(x[10]))+ (0.018925913 * float(x[11]))+ (-0.058662746 * float(x[12]))+ (-0.04909105 * float(x[13]))+ (0.020918824 * float(x[14]))+ (0.06191358 * float(x[15]))+ (0.049895655 * float(x[16]))+ (-0.01620942 * float(x[17]))+ (0.03560045 * float(x[18]))+ (0.043240108 * float(x[19]))+ (0.015790118 * float(x[20]))+ (-0.067343615 * float(x[21]))+ (-0.10984901 * float(x[22]))+ (0.030370817 * float(x[23]))+ (-0.060641028 * float(x[24]))+ (0.1124114 * float(x[25]))+ (-0.033733785 * float(x[26]))+ (0.09406915 * float(x[27]))+ (-0.026006827 * float(x[28]))+ (0.010702117 * float(x[29]))+ (-0.032169636 * float(x[30]))+ (-0.04506495 * float(x[31]))+ (-0.0016592541 * float(x[32]))+ (0.069137976 * float(x[33]))+ (0.008031667 * float(x[34]))+ (0.038363826 * float(x[35]))+ (-0.03599712 * float(x[36]))+ (0.033680096 * float(x[37]))+ (0.04563879 * float(x[38]))+ (0.054317776 * float(x[39]))+ (-0.0078962445 * float(x[40]))+ (0.12947643 * float(x[41]))+ (0.07482715 * float(x[42]))+ (0.106837995 * float(x[43]))+ (-0.035145257 * float(x[44]))+ (0.10845666 * float(x[45]))+ (-0.01981806 * float(x[46]))+ (0.04378917 * float(x[47]))+ (0.12622978 * float(x[48]))+ (-0.03792958 * float(x[49])))+ ((0.0939808 * float(x[50]))+ (0.023661958 * float(x[51]))+ (-0.10695702 * float(x[52]))+ (0.11859647 * float(x[53]))+ (-0.02462208 * float(x[54]))+ (0.035084117 * float(x[55]))+ (-0.052007955 * float(x[56]))+ (-0.008477787 * float(x[57]))+ (-0.04999943 * float(x[58]))+ (0.09687176 * float(x[59]))+ (-0.04783834 * float(x[60]))+ (0.021448476 * float(x[61]))+ (0.041561257 * float(x[62]))+ (0.07643469 * float(x[63]))+ (-0.005211677 * float(x[64]))+ (0.05650699 * float(x[65]))+ (-0.106506586 * float(x[66]))+ (0.07384866 * float(x[67]))+ (-0.061608966 * float(x[68]))+ (0.08602134 * float(x[69]))+ (0.06294706 * float(x[70]))+ (0.122007914 * float(x[71]))+ (-0.104606315 * float(x[72]))+ (-0.054064304 * float(x[73]))+ (-0.0001346003 * float(x[74]))+ (-0.029983131 * float(x[75]))+ (-0.050884105 * float(x[76]))+ (0.020966616 * float(x[77]))+ (0.03353437 * float(x[78]))+ (-0.04297024 * float(x[79]))+ (0.039519474 * float(x[80]))+ (-0.11263443 * float(x[81]))+ (-0.022972662 * float(x[82]))+ (0.13706987 * float(x[83]))+ (0.10264994 * float(x[84]))+ (0.13362506 * float(x[85]))+ (0.13615568 * float(x[86]))+ (0.0728897 * float(x[87]))+ (0.07029557 * float(x[88]))+ (0.059891827 * float(x[89]))+ (0.017079106 * float(x[90]))+ (0.12082428 * float(x[91]))+ (0.05381288 * float(x[92]))+ (-0.02823464 * float(x[93]))+ (-0.11015468 * float(x[94]))+ (0.06894135 * float(x[95]))+ (0.048148755 * float(x[96]))+ (0.063870795 * float(x[97]))+ (0.09093301 * float(x[98]))+ (0.13586053 * float(x[99])))+ ((-0.054968897 * float(x[100]))+ (-0.043377794 * float(x[101]))+ (0.033436075 * float(x[102]))+ (-0.13494685 * float(x[103]))+ (0.018216515 * float(x[104]))+ (0.038092863 * float(x[105]))+ (-0.010319798 * float(x[106]))+ (-0.063233554 * float(x[107]))+ (0.08823791 * float(x[108]))+ (-0.0183769 * float(x[109]))+ (0.047687985 * float(x[110]))+ (-0.061982818 * float(x[111]))+ (-0.05244859 * float(x[112]))+ (0.041353505 * float(x[113]))+ (-0.008056777 * float(x[114]))+ (-0.0060558394 * float(x[115]))+ (0.07671015 * float(x[116]))+ (-0.024170216 * float(x[117]))+ (-0.04871874 * float(x[118]))+ (-0.010235698 * float(x[119]))+ (-0.06259549 * float(x[120]))+ (-0.04598992 * float(x[121]))+ (0.012390623 * float(x[122]))+ (0.16098134 * float(x[123]))+ (0.1001402 * float(x[124]))+ (-0.0637604 * float(x[125]))+ (0.08338049 * float(x[126]))+ (-0.05259287 * float(x[127]))+ (0.011347307 * float(x[128]))+ (-0.03369603 * float(x[129]))+ (0.007076137 * float(x[130]))+ (0.13010786 * float(x[131]))+ (0.050441056 * float(x[132]))+ (-0.07544011 * float(x[133]))+ (0.024773994 * float(x[134]))+ (-0.1090229 * float(x[135]))+ (0.049837254 * float(x[136]))+ (0.079896405 * float(x[137]))+ (-0.023152342 * float(x[138]))+ (0.119203635 * float(x[139]))+ (0.12287213 * float(x[140]))+ (0.058823124 * float(x[141]))+ (0.050509367 * float(x[142]))+ (0.051853225 * float(x[143]))+ (-0.011047545 * float(x[144]))+ (0.0870619 * float(x[145]))+ (-0.06534386 * float(x[146]))+ (-0.036409326 * float(x[147]))+ (0.0905977 * float(x[148]))+ (0.15421961 * float(x[149])))+ ((-0.1401868 * float(x[150]))+ (0.14175338 * float(x[151]))+ (-0.010710338 * float(x[152]))+ (-0.021646218 * float(x[153]))+ (0.006719479 * float(x[154]))+ (0.044666644 * float(x[155]))+ (-0.06335673 * float(x[156]))+ (0.12941886 * float(x[157]))+ (0.011254752 * float(x[158]))+ (-0.13869992 * float(x[159]))+ (0.12609413 * float(x[160]))+ (-0.1319554 * float(x[161]))+ (0.11255228 * float(x[162]))+ (0.09792063 * float(x[163]))+ (0.021842811 * float(x[164]))+ (0.13597983 * float(x[165]))+ (0.03358084 * float(x[166]))+ (0.111975804 * float(x[167]))+ (0.10004119 * float(x[168]))+ (-0.024293814 * float(x[169]))+ (0.11491138 * float(x[170]))+ (0.018267373 * float(x[171]))+ (-0.052544232 * float(x[172]))+ (0.020566758 * float(x[173]))+ (-0.009680177 * float(x[174]))+ (0.14035401 * float(x[175]))+ (0.10567527 * float(x[176]))+ (0.04723139 * float(x[177]))+ (0.13784471 * float(x[178]))+ (0.0023904976 * float(x[179]))+ (-0.038029317 * float(x[180]))+ (-0.00047428714 * float(x[181]))+ (-0.09569624 * float(x[182]))+ (-0.04449532 * float(x[183]))+ (-0.07456369 * float(x[184]))+ (-0.027360689 * float(x[185]))+ (-0.04994544 * float(x[186]))+ (-0.10508917 * float(x[187]))+ (-0.03656004 * float(x[188]))+ (0.021313244 * float(x[189]))+ (-0.08967791 * float(x[190]))+ (0.18123692 * float(x[191]))+ (-0.020465048 * float(x[192]))+ (0.025319811 * float(x[193]))+ (-0.05604446 * float(x[194]))+ (0.05986024 * float(x[195]))+ (-0.022976153 * float(x[196]))+ (0.011083869 * float(x[197]))+ (0.051141445 * float(x[198]))+ (-0.099446975 * float(x[199])))+ ((-0.070677824 * float(x[200]))+ (0.13504 * float(x[201]))+ (0.16818456 * float(x[202]))+ (-0.009950152 * float(x[203]))+ (0.12156111 * float(x[204]))+ (-0.041559763 * float(x[205]))+ (-0.03873944 * float(x[206]))+ (0.11374135 * float(x[207]))+ (0.06347402 * float(x[208]))+ (-0.0062074545 * float(x[209]))+ (-0.043392565 * float(x[210]))+ (0.05353435 * float(x[211]))+ (0.064847946 * float(x[212]))+ (0.07625816 * float(x[213]))+ (0.029907173 * float(x[214]))+ (0.15216488 * float(x[215]))+ (-0.050392665 * float(x[216]))+ (0.005962441 * float(x[217]))+ (0.0057323044 * float(x[218]))+ (0.13782707 * float(x[219]))+ (0.0041316585 * float(x[220]))+ (-0.020251546 * float(x[221]))+ (-0.04475935 * float(x[222]))+ (-0.11234409 * float(x[223]))+ (-0.020780567 * float(x[224]))+ (-0.0568861 * float(x[225]))+ (0.002267793 * float(x[226]))+ (0.0063029374 * float(x[227]))+ (0.04985633 * float(x[228]))+ (0.10659834 * float(x[229]))+ (0.008721783 * float(x[230]))+ (0.0897245 * float(x[231]))+ (-0.0048521585 * float(x[232]))+ (-0.034448616 * float(x[233]))+ (-0.055937633 * float(x[234]))+ (-0.062730424 * float(x[235]))+ (0.037162248 * float(x[236]))+ (-0.09280843 * float(x[237]))+ (0.12365363 * float(x[238]))+ (-0.08205248 * float(x[239]))+ (0.0988693 * float(x[240]))+ (0.03873596 * float(x[241]))+ (-0.024145115 * float(x[242]))+ (0.048968993 * float(x[243]))+ (0.08837747 * float(x[244]))+ (0.14384782 * float(x[245]))+ (-0.032937385 * float(x[246]))+ (-0.02530865 * float(x[247]))+ (0.1350963 * float(x[248]))+ (-0.038084574 * float(x[249])))+ ((0.021919452 * float(x[250]))+ (-0.031231575 * float(x[251]))+ (0.118968986 * float(x[252]))+ (0.103845626 * float(x[253]))+ (-0.00368434 * float(x[254]))+ (0.13489021 * float(x[255]))+ (0.05992826 * float(x[256]))+ (0.027351042 * float(x[257]))+ (0.09733216 * float(x[258]))+ (0.06536158 * float(x[259]))+ (-0.059148524 * float(x[260]))+ (-0.06779424 * float(x[261]))+ (0.10765497 * float(x[262]))+ (-0.011625239 * float(x[263]))+ (-0.03187994 * float(x[264]))+ (0.06877178 * float(x[265]))+ (-0.081288464 * float(x[266]))+ (0.107006244 * float(x[267]))+ (0.104598686 * float(x[268]))+ (-0.0049323626 * float(x[269]))+ (-0.11810814 * float(x[270]))+ (0.116554976 * float(x[271]))+ (0.057667557 * float(x[272]))+ (-0.031338383 * float(x[273]))+ (0.16834542 * float(x[274]))+ (0.030945813 * float(x[275]))+ (-0.12029729 * float(x[276]))+ (0.021526571 * float(x[277]))+ (-0.0017082789 * float(x[278]))+ (0.053500973 * float(x[279]))+ (0.047267433 * float(x[280]))+ (-0.024691647 * float(x[281]))+ (0.1472937 * float(x[282]))+ (-0.020661755 * float(x[283]))+ (-0.012371753 * float(x[284]))+ (0.062864505 * float(x[285]))+ (0.08318797 * float(x[286]))+ (0.06316097 * float(x[287]))+ (-0.008415488 * float(x[288]))+ (-0.012640643 * float(x[289]))+ (-0.051665552 * float(x[290]))+ (-0.046988524 * float(x[291]))+ (0.092748925 * float(x[292]))+ (-0.039552674 * float(x[293]))+ (-0.022959668 * float(x[294]))+ (0.08911324 * float(x[295]))+ (-0.11803218 * float(x[296]))+ (0.07744878 * float(x[297]))+ (0.058830973 * float(x[298]))+ (0.10564511 * float(x[299])))+ ((0.0774814 * float(x[300]))+ (-0.04489123 * float(x[301]))+ (0.09881502 * float(x[302]))+ (0.12837188 * float(x[303]))+ (0.06742031 * float(x[304]))+ (0.096738 * float(x[305]))+ (-0.04845228 * float(x[306]))+ (0.06460359 * float(x[307]))+ (0.13455541 * float(x[308]))+ (0.0075969333 * float(x[309]))+ (0.106562346 * float(x[310]))+ (0.074875385 * float(x[311]))+ (0.05571382 * float(x[312]))+ (0.09464538 * float(x[313]))+ (0.0905713 * float(x[314]))+ (-0.058556523 * float(x[315]))+ (0.033058424 * float(x[316]))+ (-0.007547539 * float(x[317]))+ (0.09477912 * float(x[318]))+ (-0.017463995 * float(x[319]))+ (-0.0116122505 * float(x[320]))+ (0.07029411 * float(x[321]))+ (0.04156372 * float(x[322]))+ (0.008953539 * float(x[323]))+ (-0.01274109 * float(x[324]))+ (0.038526107 * float(x[325]))+ (0.091628894 * float(x[326]))+ (0.038008448 * float(x[327]))+ (0.00464904 * float(x[328]))+ (-0.05368371 * float(x[329]))+ (-0.01691753 * float(x[330]))+ (-0.009841253 * float(x[331]))+ (0.132029 * float(x[332]))+ (-0.017583143 * float(x[333]))+ (0.02582621 * float(x[334]))+ (0.013117838 * float(x[335]))+ (-0.011546596 * float(x[336]))+ (0.12764794 * float(x[337]))+ (0.030637989 * float(x[338]))+ (-0.032244198 * float(x[339]))+ (0.020565527 * float(x[340]))+ (0.11994464 * float(x[341]))+ (0.016944801 * float(x[342]))+ (-0.056364413 * float(x[343]))+ (0.010490576 * float(x[344]))+ (0.057111766 * float(x[345]))+ (0.08451015 * float(x[346]))+ (0.077927336 * float(x[347]))+ (0.08704377 * float(x[348]))+ (-0.034063954 * float(x[349])))+ ((0.044100165 * float(x[350]))+ (0.07818479 * float(x[351]))+ (-0.026195282 * float(x[352]))+ (0.050381046 * float(x[353]))+ (0.103189796 * float(x[354]))+ (0.058974676 * float(x[355]))+ (0.043591946 * float(x[356]))+ (0.09489953 * float(x[357]))+ (0.123139784 * float(x[358]))+ (0.10190963 * float(x[359]))+ (0.123241454 * float(x[360]))+ (0.0010581134 * float(x[361]))+ (0.1198967 * float(x[362]))+ (0.0075267064 * float(x[363]))+ (-0.051397886 * float(x[364]))+ (-0.019578168 * float(x[365]))+ (0.026319092 * float(x[366]))+ (-0.016849345 * float(x[367]))+ (0.062484622 * float(x[368]))+ (0.11408012 * float(x[369]))+ (0.11812518 * float(x[370]))+ (-0.033746157 * float(x[371]))+ (0.003970959 * float(x[372]))+ (0.015063806 * float(x[373]))+ (0.11097325 * float(x[374]))+ (0.007077108 * float(x[375]))+ (-0.05951327 * float(x[376]))+ (0.08142802 * float(x[377]))+ (0.07791596 * float(x[378]))+ (0.11699792 * float(x[379]))+ (-0.006429791 * float(x[380]))+ (0.08895265 * float(x[381]))+ (-0.062942356 * float(x[382]))+ (0.059292465 * float(x[383]))+ (-0.008484068 * float(x[384]))+ (0.06270421 * float(x[385]))+ (-0.02788436 * float(x[386]))+ (-0.016548261 * float(x[387]))+ (0.0076457122 * float(x[388]))+ (0.043061327 * float(x[389]))+ (0.0144283455 * float(x[390]))+ (0.1043796 * float(x[391]))+ (0.0804834 * float(x[392]))+ (0.013267879 * float(x[393]))+ (0.06632279 * float(x[394]))+ (0.13960041 * float(x[395]))+ (-0.014280795 * float(x[396]))+ (0.012985968 * float(x[397]))+ (0.08042517 * float(x[398]))+ (0.041543357 * float(x[399])))+ ((-0.055085856 * float(x[400]))+ (-0.018959606 * float(x[401]))+ (0.12790504 * float(x[402]))+ (0.0322271 * float(x[403]))+ (0.13437106 * float(x[404]))+ (0.093743145 * float(x[405]))+ (-0.003867983 * float(x[406]))+ (0.05849844 * float(x[407]))+ (0.10508243 * float(x[408]))+ (-0.039518874 * float(x[409]))+ (0.06823632 * float(x[410]))+ (0.044202123 * float(x[411]))+ (0.031711064 * float(x[412]))+ (-0.046875406 * float(x[413]))+ (-0.025947085 * float(x[414]))+ (0.08328991 * float(x[415]))+ (-0.037050683 * float(x[416]))+ (-0.021461323 * float(x[417]))+ (-0.0097187525 * float(x[418]))+ (0.047679804 * float(x[419]))+ (0.046763245 * float(x[420]))+ (0.0028708745 * float(x[421]))+ (0.042592287 * float(x[422]))+ (0.16060382 * float(x[423]))+ (0.020027595 * float(x[424]))+ (0.083185196 * float(x[425]))+ (-0.011330353 * float(x[426]))+ (0.03481057 * float(x[427]))+ (0.11316356 * float(x[428]))+ (0.08225107 * float(x[429]))+ (-0.054918673 * float(x[430]))+ (0.0110457 * float(x[431]))+ (0.039764635 * float(x[432]))+ (0.0021545677 * float(x[433]))+ (0.13107903 * float(x[434]))+ (0.11295445 * float(x[435]))+ (-0.039748896 * float(x[436]))+ (0.105974235 * float(x[437]))+ (0.08608991 * float(x[438]))+ (-0.05311136 * float(x[439]))+ (0.09161555 * float(x[440]))+ (-0.03210544 * float(x[441]))+ (0.0060522608 * float(x[442]))+ (0.06082196 * float(x[443]))+ (0.0450146 * float(x[444]))+ (-0.099088766 * float(x[445]))+ (-0.038413204 * float(x[446]))+ (0.06682264 * float(x[447]))+ (0.06366927 * float(x[448]))+ (0.096995756 * float(x[449])))+ ((-0.04350473 * float(x[450]))+ (0.11849163 * float(x[451]))+ (0.048352946 * float(x[452]))+ (0.083911896 * float(x[453]))+ (-0.04067693 * float(x[454]))+ (0.11487099 * float(x[455]))+ (-0.0118766455 * float(x[456]))+ (0.007479182 * float(x[457]))+ (0.113093704 * float(x[458]))+ (0.10648272 * float(x[459]))+ (0.048874367 * float(x[460]))+ (0.043716636 * float(x[461]))+ (0.07480652 * float(x[462]))+ (-0.044106804 * float(x[463]))+ (-0.0017109816 * float(x[464]))+ (-0.054138027 * float(x[465]))+ (0.13551351 * float(x[466]))+ (0.07279956 * float(x[467]))+ (-0.0037537017 * float(x[468]))+ (0.123659536 * float(x[469]))+ (0.08679453 * float(x[470]))+ (-0.05408126 * float(x[471]))+ (-0.007970925 * float(x[472]))+ (0.1253968 * float(x[473]))+ (-0.016046457 * float(x[474]))+ (0.06487359 * float(x[475]))+ (0.041978747 * float(x[476]))+ (0.02612142 * float(x[477]))+ (0.06859853 * float(x[478]))+ (0.06519737 * float(x[479]))+ (0.05956295 * float(x[480]))+ (0.033436924 * float(x[481]))+ (-0.028735759 * float(x[482]))+ (0.022564026 * float(x[483]))+ (-0.016395602 * float(x[484]))+ (0.10773992 * float(x[485]))+ (0.02397245 * float(x[486]))+ (-0.044650655 * float(x[487]))+ (0.102469735 * float(x[488]))+ (-0.012866502 * float(x[489]))+ (0.09127178 * float(x[490]))+ (-0.063043445 * float(x[491]))+ (0.08453528 * float(x[492]))+ (0.059908014 * float(x[493]))+ (-0.029119786 * float(x[494]))+ (0.06620571 * float(x[495]))+ (0.046439502 * float(x[496]))+ (0.032503564 * float(x[497]))+ (0.063964605 * float(x[498]))+ (0.0044535752 * float(x[499])))+ ((-0.0056914235 * float(x[500]))+ (-0.012208774 * float(x[501]))+ (0.17598024 * float(x[502]))+ (0.030726418 * float(x[503]))+ (0.037933957 * float(x[504]))+ (0.069037855 * float(x[505]))+ (0.102761604 * float(x[506]))+ (0.16395463 * float(x[507]))+ (0.11404247 * float(x[508]))+ (0.0026372503 * float(x[509]))+ (-0.03360736 * float(x[510]))+ (0.11676186 * float(x[511]))+ (-0.019002492 * float(x[512]))+ (-0.00037952064 * float(x[513]))+ (0.013256789 * float(x[514]))+ (0.18025734 * float(x[515]))+ (0.15923396 * float(x[516]))+ (0.1768137 * float(x[517]))+ (0.07855648 * float(x[518]))+ (0.09113793 * float(x[519]))+ (0.16306536 * float(x[520]))+ (0.08774251 * float(x[521]))+ (0.027392464 * float(x[522]))+ (0.008339861 * float(x[523]))+ (0.0225934 * float(x[524]))+ (0.006267223 * float(x[525]))+ (0.11245896 * float(x[526]))+ (0.093243584 * float(x[527]))+ (-0.04007441 * float(x[528]))+ (0.1455925 * float(x[529]))+ (-0.05206466 * float(x[530]))+ (0.04956461 * float(x[531]))+ (-0.012886635 * float(x[532]))+ (0.031132868 * float(x[533]))+ (0.13491899 * float(x[534]))+ (-0.023714056 * float(x[535]))+ (-0.033482067 * float(x[536]))+ (0.10195976 * float(x[537]))+ (0.12506479 * float(x[538]))+ (0.057273198 * float(x[539]))+ (0.10133475 * float(x[540]))+ (0.09318062 * float(x[541]))+ (0.08188639 * float(x[542]))+ (-0.03882509 * float(x[543]))+ (0.064250045 * float(x[544]))+ (-0.04297219 * float(x[545]))+ (0.123049356 * float(x[546]))+ (0.056613747 * float(x[547]))+ (-0.02278588 * float(x[548]))+ (0.011098187 * float(x[549])))+ ((0.069635205 * float(x[550]))+ (-0.09349563 * float(x[551]))+ (0.016595436 * float(x[552]))+ (0.124999605 * float(x[553]))+ (-0.12067989 * float(x[554]))+ (-0.09272416 * float(x[555]))+ (0.021301212 * float(x[556]))+ (-0.039684504 * float(x[557]))+ (0.115927294 * float(x[558]))+ (-0.1102912 * float(x[559]))+ (0.008385818 * float(x[560]))) + 0.1007153), 0)
    h_4 = max((((-0.35687912 * float(x[0]))+ (7.930541 * float(x[1]))+ (1.739576 * float(x[2]))+ (-0.5489326 * float(x[3]))+ (-3.7552004 * float(x[4]))+ (-1.8126628 * float(x[5]))+ (0.45825234 * float(x[6]))+ (-2.0256395 * float(x[7]))+ (-1.3012418 * float(x[8]))+ (3.998897 * float(x[9]))+ (-3.5447912 * float(x[10]))+ (-1.8578826 * float(x[11]))+ (-0.16023268 * float(x[12]))+ (1.0595349 * float(x[13]))+ (0.7833242 * float(x[14]))+ (-0.5420049 * float(x[15]))+ (-0.28486758 * float(x[16]))+ (-0.1081216 * float(x[17]))+ (0.007786396 * float(x[18]))+ (-0.37687814 * float(x[19]))+ (0.024711119 * float(x[20]))+ (-1.7342154 * float(x[21]))+ (-3.801735 * float(x[22]))+ (4.9909286 * float(x[23]))+ (10.290202 * float(x[24]))+ (0.4278492 * float(x[25]))+ (-2.2629738 * float(x[26]))+ (-1.4109036 * float(x[27]))+ (6.7481885 * float(x[28]))+ (1.1381307 * float(x[29]))+ (-1.0768209 * float(x[30]))+ (0.6565072 * float(x[31]))+ (3.0306096 * float(x[32]))+ (0.7801168 * float(x[33]))+ (3.4276211 * float(x[34]))+ (-3.7470462 * float(x[35]))+ (-4.982194 * float(x[36]))+ (-9.208649 * float(x[37]))+ (-6.7119493 * float(x[38]))+ (-1.3497216 * float(x[39]))+ (-0.86100703 * float(x[40]))+ (5.3867316 * float(x[41]))+ (6.492836 * float(x[42]))+ (0.5713711 * float(x[43]))+ (-0.731568 * float(x[44]))+ (0.8521819 * float(x[45]))+ (0.4148505 * float(x[46]))+ (-0.76737124 * float(x[47]))+ (0.8327347 * float(x[48]))+ (-0.6923175 * float(x[49])))+ ((4.267837 * float(x[50]))+ (6.3657475 * float(x[51]))+ (-1.1748841 * float(x[52]))+ (5.529557 * float(x[53]))+ (4.2097898 * float(x[54]))+ (-2.9585536 * float(x[55]))+ (-1.3780377 * float(x[56]))+ (-1.0918741 * float(x[57]))+ (1.309516 * float(x[58]))+ (0.593509 * float(x[59]))+ (-0.8804019 * float(x[60]))+ (0.9324713 * float(x[61]))+ (-0.34294024 * float(x[62]))+ (-2.257398 * float(x[63]))+ (0.36860645 * float(x[64]))+ (-1.0847596 * float(x[65]))+ (1.2649124 * float(x[66]))+ (-1.0920955 * float(x[67]))+ (1.1698005 * float(x[68]))+ (1.1453053 * float(x[69]))+ (-1.2489309 * float(x[70]))+ (1.0000051 * float(x[71]))+ (-0.7374604 * float(x[72]))+ (-0.07282566 * float(x[73]))+ (-0.15136899 * float(x[74]))+ (0.26115754 * float(x[75]))+ (-0.37410185 * float(x[76]))+ (1.0444939 * float(x[77]))+ (-1.1766537 * float(x[78]))+ (1.5832266 * float(x[79]))+ (-1.5943244 * float(x[80]))+ (-1.0555074 * float(x[81]))+ (2.2950218 * float(x[82]))+ (-0.5159719 * float(x[83]))+ (0.28579453 * float(x[84]))+ (1.155358 * float(x[85]))+ (-0.79510534 * float(x[86]))+ (-0.48398057 * float(x[87]))+ (1.5303365 * float(x[88]))+ (9.275581 * float(x[89]))+ (2.2144144 * float(x[90]))+ (1.1519079 * float(x[91]))+ (9.175053 * float(x[92]))+ (3.71184 * float(x[93]))+ (0.7770735 * float(x[94]))+ (0.29297027 * float(x[95]))+ (-0.086056694 * float(x[96]))+ (0.60280794 * float(x[97]))+ (0.46938926 * float(x[98]))+ (-1.8871388 * float(x[99])))+ ((-3.4127977 * float(x[100]))+ (1.3480005 * float(x[101]))+ (-3.2309947 * float(x[102]))+ (-0.8663641 * float(x[103]))+ (-2.348334 * float(x[104]))+ (0.5708294 * float(x[105]))+ (-6.33427 * float(x[106]))+ (-2.8706613 * float(x[107]))+ (-6.771844 * float(x[108]))+ (9.9882905e-05 * float(x[109]))+ (-0.8032093 * float(x[110]))+ (-3.3957944 * float(x[111]))+ (4.196988 * float(x[112]))+ (1.6654927 * float(x[113]))+ (7.4287457 * float(x[114]))+ (6.3012486 * float(x[115]))+ (-1.8897367 * float(x[116]))+ (4.430434 * float(x[117]))+ (2.6314397 * float(x[118]))+ (-1.966174 * float(x[119]))+ (0.039661665 * float(x[120]))+ (-0.46981123 * float(x[121]))+ (-1.7827705 * float(x[122]))+ (0.65196425 * float(x[123]))+ (-1.9942281 * float(x[124]))+ (-1.32718 * float(x[125]))+ (0.84024686 * float(x[126]))+ (-2.2807314 * float(x[127]))+ (-1.0688734 * float(x[128]))+ (-0.73791647 * float(x[129]))+ (-1.554867 * float(x[130]))+ (-7.5699415 * float(x[131]))+ (-0.8038935 * float(x[132]))+ (-0.020346534 * float(x[133]))+ (-3.3279412 * float(x[134]))+ (0.74252933 * float(x[135]))+ (0.9430154 * float(x[136]))+ (-0.95871305 * float(x[137]))+ (-0.030698322 * float(x[138]))+ (1.426839 * float(x[139]))+ (-2.4162922 * float(x[140]))+ (-0.5032366 * float(x[141]))+ (5.5796757 * float(x[142]))+ (3.09092 * float(x[143]))+ (4.3562326 * float(x[144]))+ (-4.3841324 * float(x[145]))+ (2.383091 * float(x[146]))+ (2.2577946 * float(x[147]))+ (-1.760729 * float(x[148]))+ (-0.46298504 * float(x[149])))+ ((0.24444349 * float(x[150]))+ (-0.9497238 * float(x[151]))+ (2.8041797 * float(x[152]))+ (-2.2789557 * float(x[153]))+ (1.8241441 * float(x[154]))+ (0.42703444 * float(x[155]))+ (2.9869573 * float(x[156]))+ (-4.2282233 * float(x[157]))+ (1.7927562 * float(x[158]))+ (-8.202681 * float(x[159]))+ (1.0977082 * float(x[160]))+ (1.5424452 * float(x[161]))+ (5.165913 * float(x[162]))+ (-1.5956941 * float(x[163]))+ (0.8334963 * float(x[164]))+ (0.65168273 * float(x[165]))+ (-2.381503 * float(x[166]))+ (0.882183 * float(x[167]))+ (0.8704899 * float(x[168]))+ (-1.7743202 * float(x[169]))+ (0.4211045 * float(x[170]))+ (4.2118216 * float(x[171]))+ (0.07189938 * float(x[172]))+ (-0.60984087 * float(x[173]))+ (1.7940859 * float(x[174]))+ (0.113237366 * float(x[175]))+ (-0.91421 * float(x[176]))+ (0.42303059 * float(x[177]))+ (0.06718469 * float(x[178]))+ (-3.892926 * float(x[179]))+ (0.9876698 * float(x[180]))+ (1.2968485 * float(x[181]))+ (2.533805 * float(x[182]))+ (2.1602755 * float(x[183]))+ (-3.0340374 * float(x[184]))+ (-5.934743 * float(x[185]))+ (-1.8313134 * float(x[186]))+ (-3.14462 * float(x[187]))+ (1.7603459 * float(x[188]))+ (-0.9324986 * float(x[189]))+ (-1.1938223 * float(x[190]))+ (-1.3011794 * float(x[191]))+ (-0.44167006 * float(x[192]))+ (-4.9247637 * float(x[193]))+ (-1.2035916 * float(x[194]))+ (3.3532288 * float(x[195]))+ (-0.63248616 * float(x[196]))+ (7.66525 * float(x[197]))+ (3.8050835 * float(x[198]))+ (2.7494915 * float(x[199])))+ ((-2.6887827 * float(x[200]))+ (4.4591465 * float(x[201]))+ (3.0389023 * float(x[202]))+ (2.543882 * float(x[203]))+ (0.11929664 * float(x[204]))+ (-2.7176757 * float(x[205]))+ (-0.076929055 * float(x[206]))+ (1.1795822 * float(x[207]))+ (1.4617476 * float(x[208]))+ (-0.12460618 * float(x[209]))+ (0.5874874 * float(x[210]))+ (0.48770157 * float(x[211]))+ (-1.3099312 * float(x[212]))+ (-2.6901765 * float(x[213]))+ (4.5219035 * float(x[214]))+ (3.1298757 * float(x[215]))+ (2.5383406 * float(x[216]))+ (0.12046155 * float(x[217]))+ (-2.6511006 * float(x[218]))+ (-0.1994919 * float(x[219]))+ (1.1861161 * float(x[220]))+ (1.3661506 * float(x[221]))+ (-0.12403646 * float(x[222]))+ (0.6506409 * float(x[223]))+ (0.46425286 * float(x[224]))+ (-1.1948074 * float(x[225]))+ (-0.21082845 * float(x[226]))+ (1.5637038 * float(x[227]))+ (2.5271633 * float(x[228]))+ (-0.4845047 * float(x[229]))+ (-1.2017603 * float(x[230]))+ (-0.22921777 * float(x[231]))+ (0.39614576 * float(x[232]))+ (2.8204286 * float(x[233]))+ (1.4825761 * float(x[234]))+ (1.4246424 * float(x[235]))+ (2.267315 * float(x[236]))+ (-2.106679 * float(x[237]))+ (-3.5088053 * float(x[238]))+ (2.013104 * float(x[239]))+ (0.3326782 * float(x[240]))+ (-0.06797369 * float(x[241]))+ (0.6847973 * float(x[242]))+ (-0.32410622 * float(x[243]))+ (2.1107519 * float(x[244]))+ (0.48438585 * float(x[245]))+ (-0.93021375 * float(x[246]))+ (0.69314474 * float(x[247]))+ (3.6602957 * float(x[248]))+ (-1.6210803 * float(x[249])))+ ((-1.9336433 * float(x[250]))+ (1.2846756 * float(x[251]))+ (0.38501346 * float(x[252]))+ (1.1615342 * float(x[253]))+ (0.8087869 * float(x[254]))+ (1.1720436 * float(x[255]))+ (-0.71129394 * float(x[256]))+ (0.35686034 * float(x[257]))+ (0.19747789 * float(x[258]))+ (0.07219898 * float(x[259]))+ (0.53110445 * float(x[260]))+ (-1.3890983 * float(x[261]))+ (3.3070679 * float(x[262]))+ (1.624038 * float(x[263]))+ (-3.5397706 * float(x[264]))+ (-0.20169792 * float(x[265]))+ (-0.9069123 * float(x[266]))+ (-0.049365792 * float(x[267]))+ (-0.84642214 * float(x[268]))+ (-4.4776573 * float(x[269]))+ (-2.7127423 * float(x[270]))+ (0.14323373 * float(x[271]))+ (-1.1967994 * float(x[272]))+ (-0.7884108 * float(x[273]))+ (-1.7041051 * float(x[274]))+ (-2.6240199 * float(x[275]))+ (-2.0484273 * float(x[276]))+ (0.46781006 * float(x[277]))+ (0.35642502 * float(x[278]))+ (-0.4577202 * float(x[279]))+ (-0.31970316 * float(x[280]))+ (-0.58161277 * float(x[281]))+ (0.16441178 * float(x[282]))+ (-0.4592414 * float(x[283]))+ (0.32737723 * float(x[284]))+ (-0.3322285 * float(x[285]))+ (2.4078012 * float(x[286]))+ (2.0441873 * float(x[287]))+ (1.0005605 * float(x[288]))+ (3.7204301 * float(x[289]))+ (-0.92719084 * float(x[290]))+ (0.4617408 * float(x[291]))+ (1.7348659 * float(x[292]))+ (-3.7605038 * float(x[293]))+ (0.68344826 * float(x[294]))+ (5.8172817 * float(x[295]))+ (1.292213 * float(x[296]))+ (1.1289668 * float(x[297]))+ (-3.6960957 * float(x[298]))+ (-1.6114336 * float(x[299])))+ ((0.27891958 * float(x[300]))+ (2.188763 * float(x[301]))+ (-0.044476114 * float(x[302]))+ (-1.5317636 * float(x[303]))+ (0.6243568 * float(x[304]))+ (0.031839535 * float(x[305]))+ (0.3652354 * float(x[306]))+ (0.2600691 * float(x[307]))+ (-0.25924352 * float(x[308]))+ (-0.52743495 * float(x[309]))+ (-0.6190584 * float(x[310]))+ (0.544527 * float(x[311]))+ (0.46059534 * float(x[312]))+ (-0.33786002 * float(x[313]))+ (-0.51549274 * float(x[314]))+ (0.227045 * float(x[315]))+ (-0.3754619 * float(x[316]))+ (0.6800035 * float(x[317]))+ (0.13654567 * float(x[318]))+ (0.26130906 * float(x[319]))+ (0.27315465 * float(x[320]))+ (-0.10367551 * float(x[321]))+ (0.5475967 * float(x[322]))+ (-0.11519454 * float(x[323]))+ (0.17005692 * float(x[324]))+ (0.21097675 * float(x[325]))+ (0.23276621 * float(x[326]))+ (0.26243052 * float(x[327]))+ (0.12131235 * float(x[328]))+ (0.14183433 * float(x[329]))+ (-0.7824875 * float(x[330]))+ (-0.73046637 * float(x[331]))+ (0.78408605 * float(x[332]))+ (0.1774289 * float(x[333]))+ (0.08085116 * float(x[334]))+ (0.11716216 * float(x[335]))+ (0.07198157 * float(x[336]))+ (-0.7325653 * float(x[337]))+ (-0.83174515 * float(x[338]))+ (0.564447 * float(x[339]))+ (0.20436037 * float(x[340]))+ (-0.19692744 * float(x[341]))+ (-0.55849195 * float(x[342]))+ (0.13331777 * float(x[343]))+ (0.07931675 * float(x[344]))+ (0.33459842 * float(x[345]))+ (1.4239646 * float(x[346]))+ (-1.1928767 * float(x[347]))+ (0.06276945 * float(x[348]))+ (0.80837107 * float(x[349])))+ ((-0.9552817 * float(x[350]))+ (-0.07870383 * float(x[351]))+ (1.1626465 * float(x[352]))+ (-0.51423305 * float(x[353]))+ (0.8223574 * float(x[354]))+ (0.7286163 * float(x[355]))+ (-0.9513158 * float(x[356]))+ (-1.4861999 * float(x[357]))+ (-0.40404087 * float(x[358]))+ (1.2842088 * float(x[359]))+ (-0.008183497 * float(x[360]))+ (0.6212094 * float(x[361]))+ (0.39472935 * float(x[362]))+ (1.423632 * float(x[363]))+ (-0.37549442 * float(x[364]))+ (1.1791937 * float(x[365]))+ (1.2774879 * float(x[366]))+ (1.5470222 * float(x[367]))+ (4.938796 * float(x[368]))+ (-0.68585366 * float(x[369]))+ (0.051088553 * float(x[370]))+ (2.4783847 * float(x[371]))+ (-1.8249902 * float(x[372]))+ (-0.08860494 * float(x[373]))+ (3.3849754 * float(x[374]))+ (2.072965 * float(x[375]))+ (1.9440585 * float(x[376]))+ (-0.330804 * float(x[377]))+ (-0.21048811 * float(x[378]))+ (0.12962608 * float(x[379]))+ (-0.2651724 * float(x[380]))+ (0.7129168 * float(x[381]))+ (-2.1621368 * float(x[382]))+ (1.0745966 * float(x[383]))+ (0.13918532 * float(x[384]))+ (0.7715263 * float(x[385]))+ (1.139542 * float(x[386]))+ (0.4769385 * float(x[387]))+ (0.04228398 * float(x[388]))+ (-0.95062584 * float(x[389]))+ (1.1639854 * float(x[390]))+ (0.8968086 * float(x[391]))+ (0.5754312 * float(x[392]))+ (-0.31736916 * float(x[393]))+ (1.0110857 * float(x[394]))+ (1.5465362 * float(x[395]))+ (0.5675261 * float(x[396]))+ (-0.30859578 * float(x[397]))+ (0.32430932 * float(x[398]))+ (-0.09543355 * float(x[399])))+ ((0.6339482 * float(x[400]))+ (0.30086038 * float(x[401]))+ (0.16759759 * float(x[402]))+ (1.0142783 * float(x[403]))+ (0.085841596 * float(x[404]))+ (0.19916159 * float(x[405]))+ (0.31216174 * float(x[406]))+ (0.61569506 * float(x[407]))+ (0.34149846 * float(x[408]))+ (0.23016095 * float(x[409]))+ (-0.4585222 * float(x[410]))+ (1.0696673 * float(x[411]))+ (0.15699555 * float(x[412]))+ (0.1956841 * float(x[413]))+ (0.3154159 * float(x[414]))+ (0.64546686 * float(x[415]))+ (0.19974095 * float(x[416]))+ (-0.07968656 * float(x[417]))+ (0.5639324 * float(x[418]))+ (0.17875995 * float(x[419]))+ (0.67984617 * float(x[420]))+ (0.63825643 * float(x[421]))+ (0.14563233 * float(x[422]))+ (-0.44761014 * float(x[423]))+ (-0.07117739 * float(x[424]))+ (-0.56892574 * float(x[425]))+ (0.8692812 * float(x[426]))+ (-3.3005111 * float(x[427]))+ (-1.3124982 * float(x[428]))+ (1.1370071 * float(x[429]))+ (-0.5672557 * float(x[430]))+ (0.09440039 * float(x[431]))+ (1.7538314 * float(x[432]))+ (-3.5168858 * float(x[433]))+ (-1.8994201 * float(x[434]))+ (-0.20140174 * float(x[435]))+ (-0.80259264 * float(x[436]))+ (-0.68480414 * float(x[437]))+ (-0.27409032 * float(x[438]))+ (0.5134074 * float(x[439]))+ (-0.94960934 * float(x[440]))+ (-0.08575621 * float(x[441]))+ (-0.5045262 * float(x[442]))+ (1.9582213 * float(x[443]))+ (1.8823628 * float(x[444]))+ (-1.233066 * float(x[445]))+ (0.7686286 * float(x[446]))+ (-0.32133988 * float(x[447]))+ (0.42215562 * float(x[448]))+ (-0.5373529 * float(x[449])))+ ((2.5829344 * float(x[450]))+ (-5.6514287 * float(x[451]))+ (-0.62180024 * float(x[452]))+ (-1.4173908 * float(x[453]))+ (0.21152778 * float(x[454]))+ (1.3022596 * float(x[455]))+ (-2.9545786 * float(x[456]))+ (-1.3872074 * float(x[457]))+ (-5.7535906 * float(x[458]))+ (-3.346094 * float(x[459]))+ (0.7139304 * float(x[460]))+ (-0.7613718 * float(x[461]))+ (-0.98129445 * float(x[462]))+ (-0.8574937 * float(x[463]))+ (-0.5293144 * float(x[464]))+ (-0.411732 * float(x[465]))+ (-0.89064145 * float(x[466]))+ (-0.5391111 * float(x[467]))+ (0.42690066 * float(x[468]))+ (-1.1104906 * float(x[469]))+ (-0.5228089 * float(x[470]))+ (-0.7920264 * float(x[471]))+ (0.42557552 * float(x[472]))+ (-0.7128023 * float(x[473]))+ (-1.9274013 * float(x[474]))+ (-0.27612755 * float(x[475]))+ (0.031273555 * float(x[476]))+ (0.43302202 * float(x[477]))+ (0.042630743 * float(x[478]))+ (0.23360422 * float(x[479]))+ (-0.30498526 * float(x[480]))+ (-1.1431202 * float(x[481]))+ (-1.3708153 * float(x[482]))+ (0.23083806 * float(x[483]))+ (0.12469247 * float(x[484]))+ (-0.7587874 * float(x[485]))+ (-1.1328965 * float(x[486]))+ (0.3720441 * float(x[487]))+ (-0.53884673 * float(x[488]))+ (1.4218379 * float(x[489]))+ (-1.2600039 * float(x[490]))+ (-0.13171983 * float(x[491]))+ (-0.061385546 * float(x[492]))+ (-0.16934332 * float(x[493]))+ (-0.17795303 * float(x[494]))+ (-0.06419147 * float(x[495]))+ (0.14654203 * float(x[496]))+ (-1.2706133 * float(x[497]))+ (-0.106822126 * float(x[498]))+ (-0.1577874 * float(x[499])))+ ((0.04954512 * float(x[500]))+ (-0.24450378 * float(x[501]))+ (5.3224835 * float(x[502]))+ (2.3722847 * float(x[503]))+ (4.230133 * float(x[504]))+ (1.056055 * float(x[505]))+ (0.6838951 * float(x[506]))+ (5.145082 * float(x[507]))+ (1.9707953 * float(x[508]))+ (1.3773509 * float(x[509]))+ (3.4574504 * float(x[510]))+ (-0.43131888 * float(x[511]))+ (0.604858 * float(x[512]))+ (-2.008938 * float(x[513]))+ (-0.89620733 * float(x[514]))+ (1.5400598 * float(x[515]))+ (1.7177494 * float(x[516]))+ (1.1557306 * float(x[517]))+ (2.8058054 * float(x[518]))+ (-0.27326226 * float(x[519]))+ (1.7118628 * float(x[520]))+ (0.98522514 * float(x[521]))+ (0.57051575 * float(x[522]))+ (1.5005842 * float(x[523]))+ (-0.29181388 * float(x[524]))+ (-1.0773653 * float(x[525]))+ (4.1615047 * float(x[526]))+ (2.9481764 * float(x[527]))+ (0.69590914 * float(x[528]))+ (-0.30774012 * float(x[529]))+ (-0.18830137 * float(x[530]))+ (-0.19143261 * float(x[531]))+ (0.12179226 * float(x[532]))+ (0.65353227 * float(x[533]))+ (-0.1276465 * float(x[534]))+ (-1.0086741 * float(x[535]))+ (0.8223905 * float(x[536]))+ (-0.07823203 * float(x[537]))+ (-0.5907831 * float(x[538]))+ (0.32403955 * float(x[539]))+ (0.57153964 * float(x[540]))+ (1.491686 * float(x[541]))+ (0.71546924 * float(x[542]))+ (1.3939183 * float(x[543]))+ (0.36575425 * float(x[544]))+ (-0.4166767 * float(x[545]))+ (1.4657835 * float(x[546]))+ (0.5887214 * float(x[547]))+ (1.2736151 * float(x[548]))+ (3.1810443 * float(x[549])))+ ((-1.1373996 * float(x[550]))+ (4.2405457 * float(x[551]))+ (-4.3140044 * float(x[552]))+ (-2.0475957 * float(x[553]))+ (-2.022011 * float(x[554]))+ (-0.6536501 * float(x[555]))+ (0.67980504 * float(x[556]))+ (-0.36678252 * float(x[557]))+ (-0.21333113 * float(x[558]))+ (-3.8278446 * float(x[559]))+ (-6.7317333 * float(x[560]))) + 0.30990952), 0)
    h_5 = max((((-3.5560198 * float(x[0]))+ (-10.744395 * float(x[1]))+ (2.7814512 * float(x[2]))+ (-1.7485739 * float(x[3]))+ (-0.45233727 * float(x[4]))+ (-0.6987925 * float(x[5]))+ (-1.4409732 * float(x[6]))+ (-0.41698766 * float(x[7]))+ (-0.7805029 * float(x[8]))+ (-4.1837883 * float(x[9]))+ (-1.1307648 * float(x[10]))+ (1.6762532 * float(x[11]))+ (0.39659926 * float(x[12]))+ (0.30634192 * float(x[13]))+ (2.5717804 * float(x[14]))+ (-1.2525187 * float(x[15]))+ (-0.2399588 * float(x[16]))+ (-0.34976456 * float(x[17]))+ (-0.46383625 * float(x[18]))+ (-0.9844789 * float(x[19]))+ (0.041715678 * float(x[20]))+ (0.027021712 * float(x[21]))+ (0.5973072 * float(x[22]))+ (0.24017695 * float(x[23]))+ (-0.3168919 * float(x[24]))+ (1.7682682 * float(x[25]))+ (0.5820341 * float(x[26]))+ (4.7756777 * float(x[27]))+ (3.4322836 * float(x[28]))+ (-0.9082496 * float(x[29]))+ (0.68692446 * float(x[30]))+ (1.5741539 * float(x[31]))+ (3.2385576 * float(x[32]))+ (-3.1548667 * float(x[33]))+ (1.7477995 * float(x[34]))+ (2.217942 * float(x[35]))+ (5.166565 * float(x[36]))+ (2.45846 * float(x[37]))+ (-1.2180647 * float(x[38]))+ (4.6054196 * float(x[39]))+ (-7.7530293 * float(x[40]))+ (1.0950447 * float(x[41]))+ (2.6704946 * float(x[42]))+ (1.5208093 * float(x[43]))+ (1.618107 * float(x[44]))+ (2.7099142 * float(x[45]))+ (1.5447502 * float(x[46]))+ (1.5015357 * float(x[47]))+ (2.643622 * float(x[48]))+ (-7.7723494 * float(x[49])))+ ((4.619487 * float(x[50]))+ (3.6767235 * float(x[51]))+ (-7.849733 * float(x[52]))+ (-0.57669973 * float(x[53]))+ (-3.9916377 * float(x[54]))+ (-6.304363 * float(x[55]))+ (-11.654998 * float(x[56]))+ (9.468389 * float(x[57]))+ (2.76899 * float(x[58]))+ (1.5338557 * float(x[59]))+ (0.8420743 * float(x[60]))+ (2.4099624 * float(x[61]))+ (0.6731729 * float(x[62]))+ (2.717892 * float(x[63]))+ (-1.1659999 * float(x[64]))+ (0.15254954 * float(x[65]))+ (-0.24516727 * float(x[66]))+ (0.1642398 * float(x[67]))+ (-0.19914486 * float(x[68]))+ (-0.89960146 * float(x[69]))+ (0.43956852 * float(x[70]))+ (0.17036225 * float(x[71]))+ (-0.870486 * float(x[72]))+ (0.09066347 * float(x[73]))+ (-0.1661331 * float(x[74]))+ (0.26376674 * float(x[75]))+ (-0.4240488 * float(x[76]))+ (1.6998646 * float(x[77]))+ (0.5268063 * float(x[78]))+ (0.9979044 * float(x[79]))+ (1.1620076 * float(x[80]))+ (0.4692753 * float(x[81]))+ (-5.1956263 * float(x[82]))+ (-1.2976054 * float(x[83]))+ (-0.75937575 * float(x[84]))+ (-0.16343878 * float(x[85]))+ (-0.9864327 * float(x[86]))+ (-1.1803478 * float(x[87]))+ (-0.27177137 * float(x[88]))+ (-1.2117169 * float(x[89]))+ (0.15166688 * float(x[90]))+ (-0.012243716 * float(x[91]))+ (1.6066399 * float(x[92]))+ (0.24822202 * float(x[93]))+ (0.47075763 * float(x[94]))+ (-0.9021132 * float(x[95]))+ (-0.22972916 * float(x[96]))+ (-0.26646754 * float(x[97]))+ (0.014121969 * float(x[98]))+ (-0.93607384 * float(x[99])))+ ((-1.1995519 * float(x[100]))+ (-0.21847212 * float(x[101]))+ (2.1260972 * float(x[102]))+ (-1.4116843 * float(x[103]))+ (3.4888928 * float(x[104]))+ (-3.1768875 * float(x[105]))+ (3.65982 * float(x[106]))+ (-2.01855 * float(x[107]))+ (-1.3375624 * float(x[108]))+ (-0.011347522 * float(x[109]))+ (0.39892292 * float(x[110]))+ (4.7562423 * float(x[111]))+ (-0.68654513 * float(x[112]))+ (0.049690917 * float(x[113]))+ (-1.8961937 * float(x[114]))+ (3.804334 * float(x[115]))+ (-1.2097334 * float(x[116]))+ (-4.9004283 * float(x[117]))+ (1.1093194 * float(x[118]))+ (-2.5378964 * float(x[119]))+ (-10.453895 * float(x[120]))+ (-3.8513186 * float(x[121]))+ (-1.8140256 * float(x[122]))+ (-0.4788013 * float(x[123]))+ (-0.79433054 * float(x[124]))+ (-0.0043488955 * float(x[125]))+ (-1.035583 * float(x[126]))+ (-0.77151686 * float(x[127]))+ (0.20139796 * float(x[128]))+ (-0.054446608 * float(x[129]))+ (0.116541386 * float(x[130]))+ (0.86730397 * float(x[131]))+ (-0.2846504 * float(x[132]))+ (0.3051374 * float(x[133]))+ (1.3541781 * float(x[134]))+ (1.1216928 * float(x[135]))+ (0.021743497 * float(x[136]))+ (-0.33269587 * float(x[137]))+ (0.09111575 * float(x[138]))+ (-1.0025364 * float(x[139]))+ (-0.7204862 * float(x[140]))+ (0.099993035 * float(x[141]))+ (3.200233 * float(x[142]))+ (3.7447777 * float(x[143]))+ (-0.8879916 * float(x[144]))+ (2.4773 * float(x[145]))+ (1.2624199 * float(x[146]))+ (0.47671106 * float(x[147]))+ (0.46198222 * float(x[148]))+ (2.538053 * float(x[149])))+ ((-0.72613865 * float(x[150]))+ (2.5185452 * float(x[151]))+ (-0.94950855 * float(x[152]))+ (-2.0691328 * float(x[153]))+ (-2.7046578 * float(x[154]))+ (-1.2377825 * float(x[155]))+ (-0.9429411 * float(x[156]))+ (4.012753 * float(x[157]))+ (-1.5325533 * float(x[158]))+ (-4.1264896 * float(x[159]))+ (-3.7211137 * float(x[160]))+ (-0.74935484 * float(x[161]))+ (4.5511317 * float(x[162]))+ (-0.40452427 * float(x[163]))+ (-0.30094868 * float(x[164]))+ (-0.6950455 * float(x[165]))+ (-0.8591992 * float(x[166]))+ (-0.14936848 * float(x[167]))+ (-0.71585804 * float(x[168]))+ (-0.019335382 * float(x[169]))+ (-0.041416943 * float(x[170]))+ (-0.11503253 * float(x[171]))+ (-0.3787151 * float(x[172]))+ (-0.033924855 * float(x[173]))+ (0.5223004 * float(x[174]))+ (-0.52586627 * float(x[175]))+ (0.023590822 * float(x[176]))+ (0.07527367 * float(x[177]))+ (0.025419097 * float(x[178]))+ (-1.2885324 * float(x[179]))+ (-0.2588418 * float(x[180]))+ (-0.4935478 * float(x[181]))+ (-11.1216 * float(x[182]))+ (4.5296516 * float(x[183]))+ (0.16780992 * float(x[184]))+ (0.822227 * float(x[185]))+ (4.2053094 * float(x[186]))+ (3.7989917 * float(x[187]))+ (3.7078776 * float(x[188]))+ (1.9522123 * float(x[189]))+ (-0.23132661 * float(x[190]))+ (-0.20307459 * float(x[191]))+ (1.1986538 * float(x[192]))+ (3.768421 * float(x[193]))+ (1.1201392 * float(x[194]))+ (-3.9859276 * float(x[195]))+ (-2.1962886 * float(x[196]))+ (-1.8781705 * float(x[197]))+ (-1.9233042 * float(x[198]))+ (-3.135301 * float(x[199])))+ ((-1.2553029 * float(x[200]))+ (0.18242782 * float(x[201]))+ (0.3924113 * float(x[202]))+ (-0.83699524 * float(x[203]))+ (-1.0600846 * float(x[204]))+ (-1.0754867 * float(x[205]))+ (-0.56677973 * float(x[206]))+ (0.34315056 * float(x[207]))+ (0.4036904 * float(x[208]))+ (0.51838636 * float(x[209]))+ (0.2660258 * float(x[210]))+ (-1.5407815 * float(x[211]))+ (0.19917822 * float(x[212]))+ (-1.2126603 * float(x[213]))+ (0.30286795 * float(x[214]))+ (0.386981 * float(x[215]))+ (-0.7842146 * float(x[216]))+ (-1.2080553 * float(x[217]))+ (-1.2368436 * float(x[218]))+ (-0.4676787 * float(x[219]))+ (0.35188824 * float(x[220]))+ (0.25673157 * float(x[221]))+ (0.3427174 * float(x[222]))+ (0.264743 * float(x[223]))+ (-1.5192573 * float(x[224]))+ (0.33650458 * float(x[225]))+ (-0.8544358 * float(x[226]))+ (-0.8502254 * float(x[227]))+ (-0.7866244 * float(x[228]))+ (-0.801281 * float(x[229]))+ (-0.56589836 * float(x[230]))+ (-0.90131974 * float(x[231]))+ (-0.17650682 * float(x[232]))+ (-0.6560712 * float(x[233]))+ (-0.9811011 * float(x[234]))+ (-1.6918787 * float(x[235]))+ (0.37139156 * float(x[236]))+ (0.05473525 * float(x[237]))+ (-0.89406335 * float(x[238]))+ (0.72497696 * float(x[239]))+ (-0.110701025 * float(x[240]))+ (0.42597917 * float(x[241]))+ (-0.062820114 * float(x[242]))+ (1.6480838 * float(x[243]))+ (0.81332093 * float(x[244]))+ (-0.12985073 * float(x[245]))+ (2.2686427 * float(x[246]))+ (4.6025276 * float(x[247]))+ (2.3702471 * float(x[248]))+ (-1.7790594 * float(x[249])))+ ((0.03326334 * float(x[250]))+ (-0.18629687 * float(x[251]))+ (-0.59405327 * float(x[252]))+ (-0.08426604 * float(x[253]))+ (-0.16721535 * float(x[254]))+ (-0.1729861 * float(x[255]))+ (-0.5369273 * float(x[256]))+ (-0.55210876 * float(x[257]))+ (-0.036864143 * float(x[258]))+ (-0.27684084 * float(x[259]))+ (0.6599835 * float(x[260]))+ (-0.47261393 * float(x[261]))+ (4.48216 * float(x[262]))+ (0.23577346 * float(x[263]))+ (-1.3939248 * float(x[264]))+ (-1.5920991 * float(x[265]))+ (-0.48382103 * float(x[266]))+ (-0.072976835 * float(x[267]))+ (-1.8251548 * float(x[268]))+ (-0.38856888 * float(x[269]))+ (-0.8206866 * float(x[270]))+ (-1.84016 * float(x[271]))+ (-1.022921 * float(x[272]))+ (-0.3073978 * float(x[273]))+ (-1.5651762 * float(x[274]))+ (0.78603137 * float(x[275]))+ (-0.29131332 * float(x[276]))+ (0.067223586 * float(x[277]))+ (1.8944101 * float(x[278]))+ (1.2668633 * float(x[279]))+ (-1.0817727 * float(x[280]))+ (-0.46279845 * float(x[281]))+ (-0.73778987 * float(x[282]))+ (-0.5151992 * float(x[283]))+ (-0.86182165 * float(x[284]))+ (-0.45940667 * float(x[285]))+ (0.07223201 * float(x[286]))+ (-2.7519445 * float(x[287]))+ (1.4846429 * float(x[288]))+ (0.10033493 * float(x[289]))+ (0.35848963 * float(x[290]))+ (2.0942311 * float(x[291]))+ (-1.9920691 * float(x[292]))+ (1.8246865 * float(x[293]))+ (-2.1742046 * float(x[294]))+ (-2.3772948 * float(x[295]))+ (-0.50945055 * float(x[296]))+ (2.1602318 * float(x[297]))+ (-0.3960603 * float(x[298]))+ (2.1570303 * float(x[299])))+ ((2.5029528 * float(x[300]))+ (0.984944 * float(x[301]))+ (-0.5139596 * float(x[302]))+ (-0.076729745 * float(x[303]))+ (-0.12396598 * float(x[304]))+ (-0.04379417 * float(x[305]))+ (-0.0951034 * float(x[306]))+ (-0.07632887 * float(x[307]))+ (0.05621983 * float(x[308]))+ (0.06322784 * float(x[309]))+ (-0.4383395 * float(x[310]))+ (-0.20632434 * float(x[311]))+ (-0.046417866 * float(x[312]))+ (0.018910028 * float(x[313]))+ (-0.30466285 * float(x[314]))+ (-0.0667339 * float(x[315]))+ (-0.81968015 * float(x[316]))+ (-0.2383016 * float(x[317]))+ (-0.038314868 * float(x[318]))+ (-0.06601315 * float(x[319]))+ (0.14141297 * float(x[320]))+ (-0.09005217 * float(x[321]))+ (0.22612901 * float(x[322]))+ (0.2746464 * float(x[323]))+ (-0.90759325 * float(x[324]))+ (-0.14335926 * float(x[325]))+ (0.019297563 * float(x[326]))+ (0.18521653 * float(x[327]))+ (-0.72601664 * float(x[328]))+ (-0.06190398 * float(x[329]))+ (-0.7647212 * float(x[330]))+ (-0.016523525 * float(x[331]))+ (0.10849595 * float(x[332]))+ (0.015277189 * float(x[333]))+ (0.027724857 * float(x[334]))+ (0.02684417 * float(x[335]))+ (0.22679588 * float(x[336]))+ (0.48020706 * float(x[337]))+ (-0.6085672 * float(x[338]))+ (0.06361694 * float(x[339]))+ (0.14974765 * float(x[340]))+ (0.18429653 * float(x[341]))+ (-0.5111602 * float(x[342]))+ (0.18880703 * float(x[343]))+ (-1.3199924 * float(x[344]))+ (-0.7886988 * float(x[345]))+ (-0.15071158 * float(x[346]))+ (-1.1292317 * float(x[347]))+ (-0.88090324 * float(x[348]))+ (-0.13281645 * float(x[349])))+ ((-1.2060596 * float(x[350]))+ (-0.8597465 * float(x[351]))+ (-0.15189485 * float(x[352]))+ (-0.9207653 * float(x[353]))+ (-0.70633864 * float(x[354]))+ (-0.2590602 * float(x[355]))+ (-0.54247874 * float(x[356]))+ (-0.29013476 * float(x[357]))+ (0.015072255 * float(x[358]))+ (-1.0179238 * float(x[359]))+ (-0.17576392 * float(x[360]))+ (-0.18809079 * float(x[361]))+ (-0.080126 * float(x[362]))+ (-1.2640572 * float(x[363]))+ (-0.55343914 * float(x[364]))+ (0.039201643 * float(x[365]))+ (-4.1889267 * float(x[366]))+ (-0.9946549 * float(x[367]))+ (0.9542069 * float(x[368]))+ (-1.6482048 * float(x[369]))+ (-0.19608226 * float(x[370]))+ (-1.0831001 * float(x[371]))+ (0.9741513 * float(x[372]))+ (0.3791932 * float(x[373]))+ (-4.782561 * float(x[374]))+ (1.8352336 * float(x[375]))+ (-2.0410714 * float(x[376]))+ (-1.2606611 * float(x[377]))+ (-1.0539876 * float(x[378]))+ (-0.5525325 * float(x[379]))+ (-0.59304756 * float(x[380]))+ (-0.27613717 * float(x[381]))+ (-0.21583544 * float(x[382]))+ (-0.10119502 * float(x[383]))+ (-0.15356618 * float(x[384]))+ (-0.11869566 * float(x[385]))+ (-0.11932187 * float(x[386]))+ (0.039660648 * float(x[387]))+ (-0.0016584541 * float(x[388]))+ (-0.18058501 * float(x[389]))+ (-0.15731548 * float(x[390]))+ (-0.06576642 * float(x[391]))+ (0.028562123 * float(x[392]))+ (-0.26967067 * float(x[393]))+ (-0.13236894 * float(x[394]))+ (-0.520256 * float(x[395]))+ (-0.16609345 * float(x[396]))+ (-0.15791117 * float(x[397]))+ (-0.017305698 * float(x[398]))+ (-0.09770236 * float(x[399])))+ ((-0.12067696 * float(x[400]))+ (-0.056434397 * float(x[401]))+ (0.08845188 * float(x[402]))+ (-0.26182216 * float(x[403]))+ (-0.08612484 * float(x[404]))+ (-0.20560306 * float(x[405]))+ (-0.0880008 * float(x[406]))+ (-0.2963529 * float(x[407]))+ (-0.16780043 * float(x[408]))+ (-0.3448581 * float(x[409]))+ (-0.11885583 * float(x[410]))+ (-0.04359196 * float(x[411]))+ (0.14757065 * float(x[412]))+ (0.07710517 * float(x[413]))+ (0.10537605 * float(x[414]))+ (-0.02060196 * float(x[415]))+ (0.12540203 * float(x[416]))+ (-0.18001506 * float(x[417]))+ (0.11620532 * float(x[418]))+ (0.008468602 * float(x[419]))+ (-0.103005685 * float(x[420]))+ (-0.09750738 * float(x[421]))+ (0.16402994 * float(x[422]))+ (-0.69480413 * float(x[423]))+ (-0.41304937 * float(x[424]))+ (-0.52647996 * float(x[425]))+ (-0.43751416 * float(x[426]))+ (-0.9064094 * float(x[427]))+ (0.0972347 * float(x[428]))+ (-0.6136604 * float(x[429]))+ (-0.6231243 * float(x[430]))+ (-0.4693925 * float(x[431]))+ (-0.6743017 * float(x[432]))+ (-0.804729 * float(x[433]))+ (0.44378707 * float(x[434]))+ (-0.21600959 * float(x[435]))+ (0.9773415 * float(x[436]))+ (0.6575429 * float(x[437]))+ (-0.56286746 * float(x[438]))+ (-0.1133019 * float(x[439]))+ (-0.27555272 * float(x[440]))+ (-0.3342979 * float(x[441]))+ (-0.5200494 * float(x[442]))+ (-0.36848703 * float(x[443]))+ (-0.4604418 * float(x[444]))+ (-3.1056943 * float(x[445]))+ (1.768737 * float(x[446]))+ (1.0593712 * float(x[447]))+ (0.7416403 * float(x[448]))+ (0.19149314 * float(x[449])))+ ((-1.3733246 * float(x[450]))+ (-3.9389226 * float(x[451]))+ (0.80558634 * float(x[452]))+ (0.59411484 * float(x[453]))+ (-2.7261627 * float(x[454]))+ (1.2786059 * float(x[455]))+ (0.5516639 * float(x[456]))+ (-1.5474691 * float(x[457]))+ (1.759372 * float(x[458]))+ (-0.9126898 * float(x[459]))+ (0.076691896 * float(x[460]))+ (-0.26394203 * float(x[461]))+ (-0.046253357 * float(x[462]))+ (0.0104402285 * float(x[463]))+ (-0.12503532 * float(x[464]))+ (0.015728211 * float(x[465]))+ (-0.029307341 * float(x[466]))+ (-0.028663538 * float(x[467]))+ (-0.051916055 * float(x[468]))+ (-0.04750503 * float(x[469]))+ (0.071678154 * float(x[470]))+ (-0.082415104 * float(x[471]))+ (-0.13966611 * float(x[472]))+ (0.1069732 * float(x[473]))+ (-0.6962336 * float(x[474]))+ (0.01498377 * float(x[475]))+ (-0.027731871 * float(x[476]))+ (0.14065306 * float(x[477]))+ (0.124793954 * float(x[478]))+ (0.08535917 * float(x[479]))+ (0.12287696 * float(x[480]))+ (0.024457894 * float(x[481]))+ (-0.34896764 * float(x[482]))+ (0.07038973 * float(x[483]))+ (0.14220864 * float(x[484]))+ (0.019373592 * float(x[485]))+ (-0.2654861 * float(x[486]))+ (0.059922922 * float(x[487]))+ (-0.2520289 * float(x[488]))+ (-0.08731825 * float(x[489]))+ (0.04424664 * float(x[490]))+ (0.08271973 * float(x[491]))+ (0.16467321 * float(x[492]))+ (0.18482281 * float(x[493]))+ (0.23918718 * float(x[494]))+ (0.248033 * float(x[495]))+ (-0.3479525 * float(x[496]))+ (-0.022232803 * float(x[497]))+ (0.07716924 * float(x[498]))+ (0.2581504 * float(x[499])))+ ((-0.17538211 * float(x[500]))+ (0.10586666 * float(x[501]))+ (0.33500636 * float(x[502]))+ (0.17628632 * float(x[503]))+ (0.4203029 * float(x[504]))+ (0.18885662 * float(x[505]))+ (2.1662698 * float(x[506]))+ (0.46413028 * float(x[507]))+ (0.5579974 * float(x[508]))+ (-0.32693183 * float(x[509]))+ (-0.27366963 * float(x[510]))+ (0.25920284 * float(x[511]))+ (1.3948959 * float(x[512]))+ (2.5233512 * float(x[513]))+ (-0.6495888 * float(x[514]))+ (-0.7975331 * float(x[515]))+ (-0.766135 * float(x[516]))+ (-0.74821204 * float(x[517]))+ (-0.575175 * float(x[518]))+ (-0.31920543 * float(x[519]))+ (-0.8519482 * float(x[520]))+ (-0.0998581 * float(x[521]))+ (-0.68476725 * float(x[522]))+ (0.15535599 * float(x[523]))+ (-0.77290374 * float(x[524]))+ (-2.1199534 * float(x[525]))+ (-1.9352753 * float(x[526]))+ (0.18826479 * float(x[527]))+ (-0.29066408 * float(x[528]))+ (-0.0006318196 * float(x[529]))+ (-0.28267542 * float(x[530]))+ (0.7458709 * float(x[531]))+ (0.18089397 * float(x[532]))+ (-0.32535392 * float(x[533]))+ (-0.009080999 * float(x[534]))+ (-0.038646214 * float(x[535]))+ (1.5429144 * float(x[536]))+ (-0.2517293 * float(x[537]))+ (-4.3547764 * float(x[538]))+ (-1.7731549 * float(x[539]))+ (0.26847482 * float(x[540]))+ (-0.101677984 * float(x[541]))+ (-0.09658377 * float(x[542]))+ (-0.19805732 * float(x[543]))+ (0.03677721 * float(x[544]))+ (0.18735494 * float(x[545]))+ (-0.014634477 * float(x[546]))+ (0.09131531 * float(x[547]))+ (-0.26548186 * float(x[548]))+ (-1.8444287 * float(x[549])))+ ((-0.6815657 * float(x[550]))+ (-1.738461 * float(x[551]))+ (0.41069698 * float(x[552]))+ (-0.26916608 * float(x[553]))+ (0.33966634 * float(x[554]))+ (1.8679851 * float(x[555]))+ (0.86844504 * float(x[556]))+ (-1.990147 * float(x[557]))+ (7.1053147 * float(x[558]))+ (-1.1123686 * float(x[559]))+ (-4.2276077 * float(x[560]))) + -0.08524584), 0)
    o[0] = (-0.45950228 * h_0)+ (0.94549096 * h_1)+ (-0.90291476 * h_2)+ (-0.070191115 * h_3)+ (0.5522076 * h_4)+ (-9.695454 * h_5) + 2.222727
    o[1] = (-0.15282127 * h_0)+ (0.66573083 * h_1)+ (1.4994626 * h_2)+ (-0.37859255 * h_3)+ (-1.4732488 * h_4)+ (-3.8098516 * h_5) + -0.2868913
    o[2] = (-0.22193535 * h_0)+ (0.12451646 * h_1)+ (0.81085914 * h_2)+ (0.21541181 * h_3)+ (1.4373764 * h_4)+ (-5.1051593 * h_5) + -8.163938
    o[3] = (0.40404114 * h_0)+ (-1.1293354 * h_1)+ (-0.37222478 * h_2)+ (-0.14316033 * h_3)+ (-0.68383366 * h_4)+ (0.29451442 * h_5) + 3.2720578
    o[4] = (-0.5720233 * h_0)+ (0.4019652 * h_1)+ (0.4272129 * h_2)+ (0.5397098 * h_3)+ (-2.201081 * h_4)+ (-0.32525015 * h_5) + 1.4790982
    o[5] = (0.4836738 * h_0)+ (-7.810798 * h_1)+ (-1.446368 * h_2)+ (0.09239885 * h_3)+ (0.052494586 * h_4)+ (0.70898104 * h_5) + -23.20472

    if num_output_logits == 1:
        if return_probabilities:
            if o[0] < 0:
                exp_o = 1. - 1./(1. + math.exp(o[0]))
            else:
                exp_o = 1./(1. + math.exp(-o[0]))
            return [1.-exp_o, exp_o]
        else:
            return o[0] >= 0
    else:
        if return_probabilities:
            max_val = max(o)
            exps = [math.exp(x - max_val) for x in o]
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
        count, correct_count, numeachclass, preds, true_labels = Validate(cleanfile)

        #Report Metrics
        model_cap = 3414
        cap_utilized = 3414
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
