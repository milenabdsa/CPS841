# =========================================================================================================#
# ====================================== EXPERIMENT PARAMETERS ============================================#
# =========================================================================================================#

# ==================================== #
# TRAINING AND TESTING INFO
SPLIT_SIZE = 0.3
numberOfRuns = 1
addressSize = 20

# Limite de amostras para não explodir o tempo de treino
MAX_SAMPLES = 500  # você pode reduzir para 500 se ainda ficar pesado

allClasses = {'adware': ['Adposhel', 'Amonetize','BrowseFox','InstallCore','MultiPlug','Neoreklami'],
              'Trojan': ['Agent', 'Dinwod', 'Elex', 'HackKMS', 'Injector','Regrun','Snarasite','VBKrypt','Vilsel'],
              'Worm': ['Allaple', 'Autorun', 'Fasong', 'Hlux'],
              'Backdoor': ['Androm','Stantinko'],
              'Virus': ['Expiro','Neshta','Sality','VBA']
             }
              
clus = ''
wisard = True
if wisard:
    bleachingActivated=True
    ignoreZero=False

cluswisard = False
if cluswisard:
    minScore = 0.1
    threshold = 350
    discriminatorLimit = 20
    clus = 'clus'

confusionPlotFilename = clus + 'thermometer12_6classes.png'

# ==================================== #
# FILE TO SAVE RESULTS
filename = clus + 'thermometer12_6classes.txt'

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
import gc
gc.enable()

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

# Load training data
with open("thermometer12_train_X.p", "rb") as input_file:
  X_train = pd.read_pickle(input_file, compression=None)

print("X_train imported")

with open("thermometer12_train_y.p", "rb") as input_file:
  y_train = pd.read_pickle(input_file, compression=None)

print("y_train imported")

# Load validation data
with open("thermometer12_val_X.p", "rb") as input_file:
  X_val = pd.read_pickle(input_file, compression=None)

print("X_val imported")

with open("thermometer12_val_y.p", "rb") as input_file:
  y_val = pd.read_pickle(input_file, compression=None)

print("y_val imported")

gc.collect()

# # ===================== LIMITAR NÚMERO DE AMOSTRAS =====================
# print("Dataset completo: ", len(X_train), "amostras")
# if len(X_train) > MAX_SAMPLES:
#     from sklearn.model_selection import train_test_split
#     print(f"Reduzindo para {MAX_SAMPLES} amostras para acelerar o experimento...")
#     _, X_sample, _, y_sample = model_selection.train_test_split(
#         X_train, y_train,
#         test_size=MAX_SAMPLES,
#         stratify=y_train,
#         random_state=42
#     )
#     X_train = X_sample
#     y_train = y_sample
#     print("Novo tamanho do dataset:", len(X_train))
# else:
#     print("Mantendo todas as amostras.")
# # ======================================================================

# Map training labels to broader categories
for i in range(len(y_train)):
  if y_train[i] in allClasses['adware']:
    y_train[i] = 'adware'
  elif y_train[i] in allClasses['Trojan']:
    y_train[i] = 'Trojan'
  elif y_train[i] in allClasses['Worm']:
    y_train[i] = 'Worm'
  elif y_train[i] in allClasses['Backdoor']:
    y_train[i] = 'Backdoor'
  elif y_train[i] in allClasses['Virus']:
    y_train[i] = 'Virus'
  gc.collect()

# Map validation labels to broader categories
for i in range(len(y_val)):
  if y_val[i] in allClasses['adware']:
    y_val[i] = 'adware'
  elif y_val[i] in allClasses['Trojan']:
    y_val[i] = 'Trojan'
  elif y_val[i] in allClasses['Worm']:
    y_val[i] = 'Worm'
  elif y_val[i] in allClasses['Backdoor']:
    y_val[i] = 'Backdoor'
  elif y_val[i] in allClasses['Virus']:
    y_val[i] = 'Virus'
  gc.collect()

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

# Get unique classes from training data
unique_classes = list(dict.fromkeys(y_train))

for i in range(numberOfRuns):
    # Use pre-split train and validation data
    X_traincv = X_train
    y_traincv = y_train
    X_testcv = X_val
    y_testcv = y_val
   
    if wisard:
        print("WiSARD")
        wsd = wp.Wisard(addressSize, bleachingActivated=bleachingActivated, ignoreZero=ignoreZero, verbose=False)

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

        clf_eval(confusionPlotFilename, y_testcv, out, classes = unique_classes)

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

        clf_eval(confusionPlotFilename, y_testcv, out, classes = unique_classes)
        
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
f.write("Training size: "+str(len(X_traincv)))
f.write('\n')
f.write("Time to train mean: "+str(np.mean(train)))
f.write('\n')
f.write("Time to train std: "+str(np.std(train)))
f.write('\n')
f.write("Testing size: "+str(len(X_testcv)))
f.write('\n')
f.write("Time to test mean: "+str(np.mean(test)))
f.write('\n')
f.write("Time to test std: "+str(np.std(test)))
f.write('\n')

f.write("==============================")
f.close()
