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
import gist

import itertools
from sklearn import model_selection
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

def plot_confusion_matrix(name, cm, classes,
                          normalize=False,
                          title=' ',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print("Chegou 3")
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    #plt.colorbar()
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
    print("Chegou 4")
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.0f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.savefig(name, pad_inches=0.2, bbox_inches='tight', dpi=300)
    plt.close()
    
def clf_eval(name, y_true, y_pred, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
    clf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    print(clf_matrix)
    print("Chegou 2")
    #print('Classification Report')
    # print(classification_report(y_true, y_pred, target_names=classes))

    #plot_confusion_matrix(name, clf_matrix, classes=classes)
    
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            #images.append(gist.extract(img))
            images.append(img)
    return images

print("Importing data")

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Adposhel')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = img
  #y = ['Adposhel']*length
  #y = ['Malign']*length
  y = ['adware']*length
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Agent')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #data = img
  #y = np.concatenate((y,['Agent']*length),axis=0)
  #y = ['Agent']*length
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Trojan']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Allaple')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #data = img
  #y = np.concatenate((y,['Allaple']*length),axis=0)
  #y = ['Allaple']*length
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Worm']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Amonetize')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Amonetize']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['adware']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Androm')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #data = img
  #y = np.concatenate((y,['Androm']*length),axis=0)
  #y = ['Androm']*length
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Backdoor']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Autorun')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Autorun']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Worm']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/BrowseFox')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['BrowseFox']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['adware']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Dinwod')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Dinwod']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Trojan']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Elex')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Elex']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Trojan']*length),axis=0)
  del img
  gc.collect()

if True: #### VIRUS
  img = load_images_from_folder('malevis_train_val_224x224/train/Expiro')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  #data = np.concatenate((data, img), axis=0)
  data = img
  #y = np.concatenate((y,['Expiro']*length),axis=0)
  y = ['Expiro']*length
  #y = np.concatenate((y,['Malign']*length),axis=0)
  #y = np.concatenate((y,['Virus']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Fasong')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Fasong']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Worm']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/HackKMS')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['HackKMS']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Trojan']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Hlux')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Hlux']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Worm']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Injector')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Injector']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Trojan']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/InstallCore')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['InstallCore']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['adware']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/MultiPlug')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['MultiPlug']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['adware']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Neoreklami')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Neoreklami']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['adware']*length),axis=0)
  del img
  gc.collect()

if True: #### VIRUS
  img = load_images_from_folder('malevis_train_val_224x224/train/Neshta')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  #data = np.concatenate((data, img), axis=0)
  data = img
  #y = np.concatenate((y,['Neshta']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  #y = np.concatenate((y,['Virus']*length),axis=0)
  y = ['Neshta']*length
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Other')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  #data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Benign']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Regrun')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Regrun']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Trojan']*length),axis=0)
  del img
  gc.collect()

if True: #### VIRUS
  img = load_images_from_folder('malevis_train_val_224x224/train/Sality')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  y = np.concatenate((y,['Sality']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  #y = np.concatenate((y,['Virus']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Snarasite')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Snarasite']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Trojan']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Stantinko')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Stantinko']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Backdoor']*length),axis=0)
  del img
  gc.collect()

if False: #### VIRUS
  img = load_images_from_folder('malevis_train_val_224x224/train/VBA')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  y = np.concatenate((y,['VBA']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  #y = np.concatenate((y,['Virus']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/VBKrypt')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['VBKrypt']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Trojan']*length),axis=0)
  del img
  gc.collect()

if False:
  img = load_images_from_folder('malevis_train_val_224x224/train/Vilsel')
  length = len(img)
  #data = np.concatenate((data, np.array(img).reshape(length,224*224*3)), axis=0)
  data = np.concatenate((data, img), axis=0)
  #y = np.concatenate((y,['Vilsel']*length),axis=0)
  #y = np.concatenate((y,['Malign']*length),axis=0)
  y = np.concatenate((y,['Trojan']*length),axis=0)
  del img
  gc.collect()

import imblearn

print("Shape data: "+str(np.array(data).shape))
print("Shape data[0]: "+str(np.array(data[0]).shape))

data = np.resize(data, (len(data),224*224))

print("Shape data: "+str(np.array(data).shape))
print("Shape data[0]: "+str(np.array(data[0]).shape))

# Dynamic
for i in range(len(data)):
    med = np.median(data[i])
    data[i] = np.where(data[i] > med, 1, 0)
X = data
del data
gc.collect()

# Termometer
#X = np.zeros((len(data),RESIZE*RESIZE*4))
#for j in range(len(data)):
#    for i in range(RESIZE*RESIZE):
#        if data[j][i] < 255/4:
#            X[j][4*i] = 1
#        elif data[j][i] > 255/4 and data[j][i] < 255/2:
#            X[j][4*i+1] = 1
#        elif data[j][i] > 255/2 and data[j][i] < 3*255/4:
#            X[j][4*i+2] = 1
#        else:
#            X[j][4*i+3] = 1


print("Shape X: "+str(X.shape))
print("Shape y: "+str(np.array(y).shape))

#undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority')
#X_over, y_over = undersample.fit_resample(X,y)
#X = X_over
#y = y_over

print("Done!")

SPLIT_SIZE = 0.3

f1 = []
precision = []
recall = []
accuracy = []
train = []
test = []

X = list(X)
y = list(y)

for i in range(1):

    print("Splitting...")

    X_traincv, X_testcv, y_traincv, y_testcv = model_selection.train_test_split(X,
                                                                            y,
                                                                            test_size=SPLIT_SIZE,
                                                                            random_state=0)
    print("Split done")

    addressSize = 20

    print("=================")

    wisard = True
    cluswisard = False

### WiSARD

    if wisard:

        print("WiSARD")

        bleachingActivated=True
        ignoreZero=False

        wsd = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = True)

        print("Training...")

        wsd.train(X_traincv, y_traincv)
        
        # wsd.leaveOneOut(X_traincv[i], y_traincv[i])
            
        print("Netword trained")

        print("Testing...")
        # classify some data
        out = wsd.classify(X_testcv)

        print("Tests done")

        # the output of classify is a string list in the same sequence as the input
        total = 0
        corrects = 0
        for count in range(len(y_testcv)):
            if y_testcv[count] == out[count]:
                corrects = corrects + 1
            total = total + 1
        print("Chegou 1")
        print("Keys: " + str(list(dict.fromkeys(y))))
        clf_eval('wisard_virus_forget', y_testcv, out, classes = list(dict.fromkeys(y)))
        new_y_testcv = y_testcv
        new_out = out

        #for i in range(len(out)):
        #    if out[i] == 'Malign':
        #        new_out.append(1)
        #    else:
        #        new_out.append(0)

        #for i in range(len(y_testcv)):
        #    if y_testcv[i] == 'Malign':
        #        new_y_testcv.append(1)
        #    else:
        #        new_y_testcv.append(0)
       
        f1.append(f1_score(new_y_testcv, new_out,average='micro'))
        precision.append(precision_score(new_y_testcv, new_out, average='micro'))
        recall.append(recall_score(new_y_testcv, new_out, average='micro'))
        accuracy.append(float(corrects)/total)

        print("f1 mean " + str(np.mean(f1)))
        print("f1 std " + str(np.std(f1)))
        print("Precision mean "+str(np.mean(precision)))
        print("Precision std "+str(np.std(precision)))
        print("Recall mean "+str(np.mean(recall)))
        print("Recall std "+str(np.std(recall)))
        print("Accuracy mean "+str(np.mean(accuracy)))
        print("Accuracy std "+str(np.std(accuracy)))
        print("==============================")

# ClusWiSARD

if cluswisard:
    print("ClusWiSARD")

    minScore = 0.1
    threshold = int(len(y_testcv)/20)
    discriminatorLimit = 20

    clus = wp.ClusWisard(addressSize, minScore, threshold, discriminatorLimit, verbose = True)

    print("Training...")

    start_train = time.time()
    if False:
        print("Supervised")
        clus.train(X_traincv,y_traincv)
    elif False:
        print("Semi-supervised")
        print(list(dict.fromkeys(y)))
        classes_dict = {
            1: "Expiro",
            2: "Neshta",
            3: "Sality"
        }
        clus.train(X_traincv, classes_dict)
    elif True:
        print("Unsupervised")
        clus.trainUnsupervised(X_traincv)
        out = clus.classifyUnsupervised(X_traincv)
        class_numbers = {
            "Expiro": [],
            "Neshta": [],
            "Sality": [],
        }
        for count in range(len(y_traincv)):
            print("Correct: " + str(y_traincv[count]) + "\t" + "Unsuperv.: " + str(out[count]))
            class_numbers[y_traincv[count]].append(out[count])
    finish_train = time.time()
    print(class_numbers)

    print("Netword trained")

    print("Testing...")
    # classify some data
    start_classify = time.time()
    total = 0
    ok = 0
    if False:
        out = clus.classify(X_testcv)
    elif True:
        out = clus.classifyUnsupervised(X_testcv)
        for count in range(len(y_testcv)):
            if out[count] in class_numbers[y_testcv[count]]:
                correct = True
                ok = ok + 1
            else:
                correct = False
            total = total + 1
            print("Correct: " + str(y_testcv[count]) + "\t\t" + "Unsuperv.: " + str(out[count]))
            # class_numbers[y_testcv[count]].append(out[count])
    finish_classify = time.time()
    #print(class_numbers)

    print("Tests done")

    print(total)
    print(ok)

    # the output of classify is a string list in the same sequence as the input
    total = 0
    corrects = 0
    for count in range(len(y_testcv)):
        if y_testcv[count] == out[count]:
            corrects = corrects + 1
        total = total + 1

    print("Keys: " + str(list(dict.fromkeys(y))))
    clf_eval('confusion_clus.png', y_testcv, out, classes = list(dict.fromkeys(y)))
    print("----------------------")
    print(float(corrects)/total)
    print(finish_train-start_train)
    print(finish_classify-start_classify)


