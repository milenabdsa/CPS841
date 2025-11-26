import matplotlib
import matplotlib.pyplot as plt
import wisardpkg as wp
import numpy as np
import pandas as pd
import time
import cv2
import gc
import os
import pickle

import itertools
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

def plot_confusion_matrix(name, cm, classes,
                          normalize=True,
                          title=' ',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # "K-S D-value = {}\nK-S p-value = {:.3e}".format(round(ks_D, 3), ks_p)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(name, pad_inches=0.2, bbox_inches='tight')
    plt.close()
    
def clf_eval(name, y_true, y_pred, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
    clf_matrix = confusion_matrix(y_true, y_pred)
    #print('Classification Report')
    # print(classification_report(y_true, y_pred, target_names=classes))

# Import data

print("Importing X...")

f = open("X.p", "rb")
X = pickle.load(f)
while 1:
    try:
        X = np.concatenate((X,pickle.load(f)),axis=0)
    except EOFError:
        break
f.close()

print("Import X done")

print("Importing y...")

f = open("y.p", "rb")
y = pickle.load(f)
f.close()

print("Import y done")

SPLIT_SIZE = 0.3

ADDRESS_SIZE = [14,28,42,56,70,84,98]
ADDRESS_SIZE = [50]

NUM_EXPERIMENTS = 1

prec = []
sd_prec = []
t_train = []
t_test = []

for addressSize in ADDRESS_SIZE:
    
    print(ADDRESS_SIZE)

    precision = []
    time_train = 0
    time_test = 0
    for j in range(NUM_EXPERIMENTS):
        print("Experiment "+str(j))
        ignoreZero  = False # optional; causes the rams to ignore the address 0

    # False by default for performance reasons,
    # when True, WiSARD prints the progress of train() and classify()
        verbose = False

        wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)
        
        print("Spliting...")

        X_traincv, X_testcv, y_traincv, y_testcv = model_selection.train_test_split(X,
                                                                                    y,
                                                                                    test_size=SPLIT_SIZE,
                                                                                    random_state=0)
        
        print("Split done")

        print("Training")
        start_train = time.time()
        wsd.train(X_traincv,y_traincv)
        finish_train = time.time()
        
        print("Netword trained")
        
        # classify some data
        print("Classifying")
        start_classify = time.time()
        out = wsd.classify(X_testcv)
        finish_classify = time.time()
        
        print("Tests done")

        # the output of classify is a string list in the same sequence as the input
        total = 0
        corrects = 0
        for count in range(len(y_testcv)):
            if y_testcv[count] == out[count]:
                corrects = corrects + 1
            total = total + 1

        clf_eval('confusion ' + str(int(SPLIT_SIZE*100)) + ' ' + str(addressSize) + '.png', y_testcv, out, classes = list(dict.fromkeys(y)))
        time_train = time_train + finish_train-start_train
        time_test = finish_classify-start_classify
        precision.append(float(corrects)/total)

    prec.append(np.mean(precision))
    sd_prec.append(np.std(precision))
    t_train.append(time_train/NUM_EXPERIMENTS)
    t_test.append(time_test/NUM_EXPERIMENTS)

print(prec)
