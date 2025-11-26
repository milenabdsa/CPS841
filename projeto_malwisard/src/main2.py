import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wisardpkg as wp
import time
from sklearn import model_selection

import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import imblearn

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
    plt.savefig(name, pad_inches=0.2, bbox_inches='tight', dpi=300)
    plt.close()

def clf_eval(name, y_true, y_pred, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
    clf_matrix = confusion_matrix(y_true, y_pred)
    #print('Classification Report')
    # print(classification_report(y_true, y_pred, target_names=classes))

    plot_confusion_matrix(name, clf_matrix, classes=classes)

data = pd.read_csv("dynamic/top_1000_pe_imports.csv")
data.drop('hash', axis = 1, inplace = True)
output = data['malware']
data.drop('malware', axis = 1, inplace = True)

X = data.values
output = output.values
y = []
mal = 0
good = 0
for i in range(len(output)):
    if output[i] == 1:
        y.append("malware")
    else:
        y.append("goodware")

undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority')
X_over, y_over = undersample.fit_resample(X,y)
X = X_over
y = y_over

from collections import Counter
print(Counter(y))
print(Counter(y_over))

SPLIT_SIZE = 0.30
ADDRESS_SIZE = [10,20,30,40,50,60]
ADDRESS_SIZE = [20]
NUM_EXP = 10
#NUM_EXP = 2

precision = []
train = []
test = []
f1 = []
acc = []
recall = []

for addressSize in ADDRESS_SIZE:

    prec = []
    time_train = []
    time_test = []
    f1_value = []
    acc_value = []
    recall_value = []
    
    for i in range(NUM_EXP):

        X_traincv, X_testcv, y_traincv, y_testcv = model_selection.train_test_split(X,
                                                                            y,
                                                                            test_size=SPLIT_SIZE,
                                                                            random_state=0)

        bleachingActivated=True
        ignoreZero=False

        wsd = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = True)
        minScore = 0.2
        threshold = 10
        discriminatorLimit = 100

        clus = wp.ClusWisard(addressSize, minScore, threshold, discriminatorLimit, verbose = True)



        start_train = time.time()
        clus.train(X_traincv,y_traincv)
        finish_train = time.time()


        # classify some data
        start_classify = time.time()
        out = clus.classify(X_testcv)
        finish_classify = time.time()

        # the output of classify is a string list in the same sequence as the input
        total = 0
        corrects = 0
        for count in range(len(y_testcv)):
            if y_testcv[count] == out[count]:
                corrects = corrects + 1
            total = total + 1

        clf_eval('confusion_'+str(addressSize)+'_'+str(SPLIT_SIZE)+'.png', y_testcv, out, classes = list(dict.fromkeys(y)))
        prec.append(float(corrects)/total)
        time_train.append(finish_train-start_train)
        time_test.append(finish_classify-start_classify)

        new_y_testcv = []
        new_out = []

        for i in range(len(out)):
            if out[i] == 'malware':
                new_out.append(1)
            else:
                new_out.append(0)

        for i in range(len(y_testcv)):
            if y_testcv[i] == 'malware':
                new_y_testcv.append(1)
            else:
                new_y_testcv.append(0)
       
        f1_value.append(f1_score(new_y_testcv, new_out))
        acc_value.append(precision_score(new_y_testcv, new_out))
        recall_value.append(recall_score(new_y_testcv, new_out))

    precision.append(prec)
    train.append(time_train)
    test.append(time_test)
    acc.append(acc_value)
    recall.append(recall_value)
    f1.append(f1_value)

for i in range(len(ADDRESS_SIZE)):
    print("Mean accuracy for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.mean(precision[i])))
    print("Std dev accuracy for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.std(precision[i])))
    print("------------------------------------------------------------------------------")
    print("Mean precision for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.mean(acc[i])))
    print("Std dev precision for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.std(acc[i])))
    print("------------------------------------------------------------------------------")
    print("Mean recall for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.mean(recall[i])))
    print("Std dev recall for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.std(recall[i])))
    print("------------------------------------------------------------------------------")
    print("Mean f1 score for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.mean(f1[i])))
    print("Std dev f1 score for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.std(f1[i])))
    print("------------------------------------------------------------------------------")
    print("Mean train time for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.mean(train[i])))
    print("Std train time for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.std(train[i])))
    print("------------------------------------------------------------------------------")
    print("Mean test time for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.mean(test[i])))
    print("Std test time for "+str(ADDRESS_SIZE[i])+" address size: "+str(np.std(test[i])))
    print("================================================================================")


if False:
    from sklearn.model_selection import KFold
    print("5-fold CV")
    kf = KFold(n_splits=5)

    prec = []
    t_train = []
    t_test = []
    acc = []
    f1 = []
    recall = []

    addressSize = 20

    X = np.array(X)
    y = np.array(y)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ignoreZero  = False
        verbose = False

        wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)

        start_train = time.time()
        wsd.train(X_train,y_train)
        finish_train = time.time()

        start_classify = time.time()
        out = wsd.classify(X_test)
        finish_classify = time.time()

        # the output of classify is a string list in the same sequence as the input
        total = 0
        corrects = 0
        for count in range(len(y_test)):
            if y_test[count] == out[count]:
                corrects = corrects + 1
            total = total + 1

        t_train.append(finish_train-start_train)
        t_test.append(finish_classify-start_classify)
        prec.append(float(corrects)/total)
        new_y_testcv = []
        new_out = []

        for i in range(len(out)):
            if out[i] == 'malware':
                new_out.append(1)
            else:
                new_out.append(0)

        for i in range(len(y_test)):
            if y_test[i] == 'malware':
                new_y_testcv.append(1)
            else:
                new_y_testcv.append(0)
       
        f1.append(f1_score(new_y_testcv, new_out))
        acc.append(precision_score(new_y_testcv, new_out))
        recall.append(recall_score(new_y_testcv, new_out))



    print("Mean accuracy for " + str(addressSize) + " address size: "+str(np.mean(prec)))
    print("Std dev accuracy for "+str(addressSize)+" address size: "+str(np.std(prec)))
    print("------------------------------------------------------------------------------")
    print("Mean precision for "+str(addressSize)+" address size: "+str(np.mean(acc)))
    print("Std dev precision for "+str(addressSize)+" address size: "+str(np.std(acc)))
    print("------------------------------------------------------------------------------")
    print("Mean recall for "+str(addressSize)+" address size: "+str(np.mean(recall)))
    print("Std dev recall for "+str(addressSize)+" address size: "+str(np.std(recall)))
    print("------------------------------------------------------------------------------")
    print("Mean f1 score for "+str(addressSize)+" address size: "+str(np.mean(f1)))
    print("Std dev f1 score for "+str(addressSize)+" address size: "+str(np.std(f1)))
    print("------------------------------------------------------------------------------")
    print("Mean train time for "+str(addressSize)+" address size: "+str(np.mean(train)))
    print("Std train time for "+str(addressSize)+" address size: "+str(np.std(train)))
    print("------------------------------------------------------------------------------")
    print("Mean test time for "+str(addressSize)+" address size: "+str(np.mean(test)))
    print("Std test time for "+str(addressSize)+" address size: "+str(np.std(test)))
    print("================================================================================")


