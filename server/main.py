# =========================================================================================================#
# ====================================== EXPERIMENT PARAMETERS ============================================#
# =========================================================================================================#

# ==================================== #
# HOW DO YOU WANT TO IMPORT THE DATA?
binaryClassification = False
multiClassification = True
# ==================================== #
# DO YOU WANT TO INTERPOLATE?
interpolate = False
FACTOR = 224
# ==================================== #
# WHICH BINARIZATION DO YOU WANT TO USE?
simple = False
THRESHOLD = 127

dynamic = False

thermometer = True
N = 4

circularThermometer = False
nBits = 4
# ==================================== #
# TRAINING AND TESTING INFO
SPLIT_SIZE = 0.3
numberOfRuns = 1
addressSize = 20

confusionPlotFilename = 'confusion_circulartherm.png'

wisard = True
if wisard:
    bleachingActivated=True
    ignoreZero=False

cluswisard = False
if cluswisard:
    minScore = 0.1
    threshold = 10
    discriminatorLimit = 100

# ==================================== #
# FILE TO SAVE RESULTS
filename = 'circulartherm.txt'

# ==================================== #
# DEBUG OPTIONS
inicio = 0
fim = 9100
tamanho = fim-inicio

# =========================================================================================================#
# ======================================== IMPORT MODULES =================================================#
# =========================================================================================================#
print("Importing modules...")

import matplotlib
import matplotlib.pyplot as plt
import wisardpkg as wp
import numpy as np
import pandas as pd
import time
import cv2
import os
import pickle
import itertools
from sklearn import model_selection
from sklearn.preprocessing import binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import multiprocessing

def plot_confusion_matrix(name, cm, classes,
                          normalize=True,
                          title=' ',
                          cmap=plt.cm.Blues):
    #plt.figure(figsize=(40,40))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(name, pad_inches=0.2, bbox_inches='tight', dpi=300)
    plt.close()
    
def clf_eval(name, y_true, y_pred, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
    clf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(name, clf_matrix, classes=classes)

print("Done!")

# =========================================================================================================#
# ====================================== IMPORT DATA ======================================================#
# =========================================================================================================#
print("Importing data...")

data = pickle.load( open( "data.p", "rb" ) )
print(data.shape)
if binaryClassification:
    y = pickle.load( open( "y2.p", "rb" ) )
if multiClassification:
    y = pickle.load( open( "y26.p", "rb" ) )

#y_6 = pickle.load( open( "y6.p", "rb" ) )
print("Done!")

data = data[inicio:fim]
y = y[inicio:fim]

print(y)

# =========================================================================================================#
# ========================================== INTERPOLATION ================================================#
# =========================================================================================================#
if interpolate:
    print("Starting interpolation...")
    new_data = []

    for i in range(len(data)):
        new_data.append(cv2.resize(data[i], dsize=(FACTOR, FACTOR), interpolation=cv2.INTER_CUBIC))
    data = new_data
    print("Done!")

# =========================================================================================================#
# ========================================== RESHAPING ====================================================#
# =========================================================================================================#
print("Reshaping...")   
data = np.array(data).reshape(tamanho, 224*224*3)
print("Done!")

# =========================================================================================================#
# ========================================= BINARIZATION ==================================================#
# =========================================================================================================#
print("Binarization...")

if simple:
    X = binarize(data, threshold = THRESHOLD)

if dynamic:
    for i in range(len(data)):
        med = np.median(data[i])
        data[i] = np.where(data[i] > med, 1, 0)
    X = data

if thermometer:
    X = [[0 for i in range(224*224*N)] for j in range(tamanho)]

    def thermometer2(j):
        print(j)
        X.append(list(np.zeros(224*224*N, dtype=int)))
        for i in range(224*224):
            for k in range(N):
                if data[j][i] >= k*255/N and data[j][i] < (k+1)*255/N:
                    X[j][N*i+k] = 1

    pool = multiprocessing.Pool(processes = 8)
    pool.map(thermometer2, range(tamanho))
    pool.close()
    pool.join()

#    for j in range(tamanho):
#        print(j)
#        for i in range(224*224):
#            for k in range(N):
#                if data[j][i] >= k*255/N and data[j][i] < (k+1)*255/N:
#                    X[j][N*i+k] = 1

if circularThermometer:
    X = []
    for i in range(tamanho):
        print(i)
        X.append(list(np.zeros(224*224*nBits, dtype=int)))
    for j in range(tamanho):
        print(j)
        for i in range(224*224):
            for k in range(nBits):
                if data[j][i] >= k*255/nBits and data[j][i] < (k+1)*255/nBits:
                    # print("data: ", data[j][i])
                    for l in range(int(nBits/2)):
                        X[j][nBits*i+ (k+l) % nBits] = 1
                    # print(X[j][nBits*i])
                    # print(X[j][nBits*i+1])
                    # print(X[j][nBits*i+2])
                    # print(X[j][nBits*i+3])
 
print("Done")

# =========================================================================================================#
# ================================= TRAINING AND TESTING ==================================================#
# =========================================================================================================#
print("Training and testing...")

f1 = []
precision = []
recall = []
accuracy = []
train = []
test = []

for i in range(numberOfRuns):
    X_traincv, X_testcv, y_traincv, y_testcv = model_selection.train_test_split(X,
                                                   	                            y,
                                                                            	  test_size=SPLIT_SIZE,
                                                                            	  random_state=0)
    
    if wisard:
        print("WiSARD")
        wsd = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = True)

        start_train = time.time()
        wsd.train(X_traincv,y_traincv)
        finish_train = time.time()

        start_classify = time.time()
        out = wsd.classify(X_testcv)
        finish_classify = time.time()

        total = 0
        corrects = 0
        for count in range(len(y_testcv)):
            if y_testcv[count] == out[count]:
                corrects = corrects + 1
            total = total + 1

        clf_eval(confusionPlotFilename, y_testcv, out, classes = list(dict.fromkeys(y)))

        f1.append(f1_score(y_testcv, out,average='weighted'))
        precision.append(precision_score(y_testcv, out, average='weighted'))
        recall.append(recall_score(y_testcv, out, average='weighted'))
        accuracy.append(float(corrects)/total)
        train.append(finish_train-start_train)
        test.append(finish_classify-start_classify)

    if cluswisard:
        print("ClusWiSARD")
        clus = wp.ClusWisard(addressSize, minScore, threshold, discriminatorLimit, verbose = True)

        start_train = time.time()
        clus.train(X_traincv,y_traincv)
        finish_train = time.time()

        start_classify = time.time()
        out = clus.classify(X_testcv)
        finish_classify = time.time()

        total = 0
        corrects = 0
        for count in range(len(y_testcv)):
            if y_testcv[count] == out[count]:
                corrects = corrects + 1
            total = total + 1

        clf_eval(confusionPlotFilename, y_testcv, out, classes = list(dict.fromkeys(y)))
        
        f1.append(f1_score(y_testcv, out, average='weighted'))
        precision.append(precision_score(y_testcv, out, average='weighted'))
        recall.append(recall_score(y_testcv, out, average='weighted'))
        accuracy.append(float(corrects)/total)
        train.append(finish_train-start_train)
        test.append(finish_classify-start_classify)

f = open(filename, "w")
f.write("==============================")
f.write('\n')
f.write("===== EXPERIMENT RESULTS =====")
f.write('\n')
f.write("==============================")
f.write('\n')

f.write("F1-score mean: " + str(np.mean(f1)))
f.write('\n')
f.write("F1-score std: " + str(np.std(f1)))
f.write('\n')
f.write("Precision mean: "+str(np.mean(precision)))
f.write('\n')
f.write("Precision std: "+str(np.std(precision)))
f.write('\n')
f.write("Recall mean: "+str(np.mean(recall)))
f.write('\n')
f.write("Recall std: "+str(np.std(recall)))
f.write('\n')
f.write("Accuracy mean: "+str(np.mean(accuracy)))
f.write('\n')
f.write("Accuracy std: "+str(np.std(accuracy)))
f.write('\n')
f.write("Training size: "+str((fim-inicio)*(1-SPLIT_SIZE)))
f.write('\n')
f.write("Time to train mean: "+str(np.mean(train)))
f.write('\n')
f.write("Time to train std: "+str(np.std(train)))
f.write('\n')
f.write("Testing size: "+str((fim-inicio)*SPLIT_SIZE))
f.write('\n')
f.write("Time to test mean: "+str(np.mean(test)))
f.write('\n')
f.write("Time to test std: "+str(np.std(test)))
f.write('\n')

f.write("==============================")
f.close()
