# =========================================================================================================#
# ====================================== EXPERIMENT PARAMETERS ============================================#
# =========================================================================================================#

# ==================================== #
# TRAINING AND TESTING INFO
SPLIT_SIZE = 0.3
numberOfRuns = 1
addressSize = 20

allClasses = {'adware': ['Adposhel', 'Amonetize','BrowseFox','InstallCore','MultiPlug','Neoreklami'],
              'Trojan': ['Agent', 'Dinwod', 'Elex', 'HackKMS', 'Injector','Regrun','Snarasite','VBKrypt','Vilsel'],
              'Worm': ['Allaple', 'Autorun', 'Fasong', 'Hlux'],
              'Backdoor': ['Androm','Stantinko'],
              'Virus': ['Expiro','Neshta','Sality','VBA']
             }
              

classes = 'Virus'

confusionPlotFilename = 'thermometer8_' + classes + 'forget.png'

wisard = True
if wisard:
    bleachingActivated=True
    ignoreZero=False

cluswisard = False
if cluswisard:
    minScore = 0.1
    threshold = 20
    discriminatorLimit = 5

# ==================================== #
# FILE TO SAVE RESULTS
filename = 'thermometer8_' + classes + 'forget.txt'

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
    plt.figure(figsize=(20,20))
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

X = pickle.load( open( "thermometer8_X.p", "rb" ) )
y = pickle.load( open( "thermometer8_y.p", "rb" ) )
X_new = []
y_new = []

print("Class: " + classes)

for i in range(len(y)):
  if y[i] in allClasses[classes]:
    X_new.append(X[i])
    y_new.append(y[i])

X = X_new
y = y_new

print("Done!")

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
        
        for i in range(len(X)):
            wsd.train(X_traincv[i],y_traincv[i])
            out = wsd.classify(X_traincv[i])
            if out[-1] != y_traincv[i]:
                wsd.leaveOneOut(X_traincv[i], y_traincv[i])
            

        out = wsd.classify(X_testcv)

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

f.write("==============================")
f.close()
