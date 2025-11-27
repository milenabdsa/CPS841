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
from math import ceil
import random

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

def most_frequent(List): 
    return max(set(List), key = List.count) 

def clf_eval(name, y_true, y_pred, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
    clf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    print(clf_matrix)
    
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            #images.append(gist.extract(img))
            images.append(img)
    return images


def import_data_sb(classes):
  first = True
  if "Expiro" in classes: #### VIRUS # EXPIRO
    img = load_images_from_folder('malevis_train_val_224x224/train/Expiro')
    if first: # FIRST?
      length = len(img)
      data = img
      y = ['Expiro']*length
      first = False
    else:
      data = np.concatenate((data, img), axis=0)
      y = np.concatenate((y,['Expiro']*length),axis=0)
    del img
    gc.collect()

  if "Neshta" in classes: #### VIRUS # NESHTA
    img = load_images_from_folder('malevis_train_val_224x224/train/Neshta')
    if first: # FIRST?
      length = len(img)
      data = img
      y = ['Neshta']*length
      first = False
    else:
      data = np.concatenate((data, img), axis=0)
      y = np.concatenate((y,['Neshta']*length),axis=0)
    del img
    gc.collect()

  if "Sality" in classes: #### VIRUS SALITY
    img = load_images_from_folder('malevis_train_val_224x224/train/Sality')
    if first: # FIRST?
      length = len(img)
      data = img
      y = ['Sality']*length
      first = False
    else:
      data = np.concatenate((data, img), axis=0)
      y = np.concatenate((y,['Sality']*length),axis=0)
    del img
    gc.collect()

  if "VBA" in classes: #### VIRUS VBA
    img = load_images_from_folder('malevis_train_val_224x224/train/VBA')
    if first: # FIRST?
      length = len(img)
      data = img
      y = ['VBA']*length
      first = False
    else:
      data = np.concatenate((data, img), axis=0)
      y = np.concatenate((y,['VBA']*length),axis=0)
    del img
    gc.collect()

  data = np.resize(data, (len(data),224*224))  

  data = data.tolist()

  random.shuffle(data)
 
  X_lsb = []
  X_msb = []

  for i in range(len(data)):
    X_lsb_current = []
    X_msb_current = []
    for k in range(len(data[0])):
      binary = list('{0:0b}'.format(data[i][k]))
      total = len(binary)
      for j in range(8-total):
        binary.insert(0,'0')
      for j in range(8):
        binary[j] = int(binary[j])

      X_lsb_current.append(binary[3])
      X_lsb_current.append(binary[2])
      X_lsb_current.append(binary[1])
      X_lsb_current.append(binary[0])
      X_msb_current.append(binary[7])
      X_msb_current.append(binary[6])
      X_msb_current.append(binary[5])
      X_msb_current.append(binary[4])

    X_lsb.append(X_lsb_current)
    X_msb.append(X_msb_current)
    
  del data
  gc.collect()

  split = ceil(len(y)*0.7)
  X_traincv_lsb = X_lsb[0:split]
  X_testcv_lsb = X_lsb[split::]
  X_traincv_msb = X_msb[0:split]
  X_testcv_msb = X_msb[split::]
  y_traincv = y[0:split]
  y_testcv = y[split::]

  return X_traincv_lsb, X_testcv_lsb, X_traincv_msb, X_testcv_msb, y_traincv, y_testcv


def import_data(classes):
  first = True
  if "Expiro" in classes: #### VIRUS # EXPIRO
    img = load_images_from_folder('malevis_train_val_224x224/train/Expiro')
    if first: # FIRST?
      length = len(img)
      data = img
      y = ['Expiro']*length
      first = False
    else:
      data = np.concatenate((data, img), axis=0)
      y = np.concatenate((y,['Expiro']*length),axis=0)
    del img
    gc.collect()

  if "Neshta" in classes: #### VIRUS # NESHTA
    img = load_images_from_folder('malevis_train_val_224x224/train/Neshta')
    if first: # FIRST?
      length = len(img)
      data = img
      y = ['Neshta']*length
      first = False
    else:
      data = np.concatenate((data, img), axis=0)
      y = np.concatenate((y,['Neshta']*length),axis=0)
    del img
    gc.collect()

  if "Sality" in classes: #### VIRUS SALITY
    img = load_images_from_folder('malevis_train_val_224x224/train/Sality')
    if first: # FIRST?
      length = len(img)
      data = img
      y = ['Sality']*length
      first = False
    else:
      data = np.concatenate((data, img), axis=0)
      y = np.concatenate((y,['Sality']*length),axis=0)
    del img
    gc.collect()

  if "VBA" in classes: #### VIRUS VBA
    img = load_images_from_folder('malevis_train_val_224x224/train/VBA')
    if first: # FIRST?
      length = len(img)
      data = img
      y = ['VBA']*length
      first = False
    else:
      data = np.concatenate((data, img), axis=0)
      y = np.concatenate((y,['VBA']*length),axis=0)
    del img
    gc.collect()

  data = np.resize(data, (len(data),224*224))

  # Dynamic
  for i in range(len(data)):
      med = np.median(data[i])
      data[i] = np.where(data[i] > med, 1, 0)
  X = data
  del data
  gc.collect()

  X = list(X)
  y = list(y)
  
  SPLIT_SIZE = 0.3
  X_traincv, X_testcv, y_traincv, y_testcv = model_selection.train_test_split(X,
                                                                              y,
                                                                              test_size=SPLIT_SIZE,
                                                                              random_state=0)

  return X_traincv, X_testcv, y_traincv, y_testcv

f1 = []
precision = []
recall = []
accuracy = []
train = []
test = []

activation = False
mental_images = True
wisard_pairs_voting = False
wisard_pairs = False
wisard_voting = False
cluswisard = False

if activation:
    print("WiSARD")

    bleachingActivated=True
    ignoreZero=False

    wsd_lsb  = wp.Wisard(20, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False, returnActivationDegree=True,returnClassesDegrees=True)
    wsd_msb  = wp.Wisard(20, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False, returnActivationDegree=True,returnClassesDegrees=True)

    print("Training...")

    if True:
        X_traincv_lsb, X_testcv_lsb, X_traincv_msb, X_testcv_msb, y_traincv, y_testcv = import_data_sb(["Expiro","Neshta","Sality","VBA"])
    
        ds_lsb_train = wp.DataSet()
        ds_msb_train = wp.DataSet()
        ds_lsb_test = wp.DataSet()
        ds_msb_test = wp.DataSet()

        for i in len(X_traincv_lsb):
            ds_lsb_train.add(wp.BinInput(X_traincv_lsb[i]), y_traincv[i])
            ds_msb_train.add(wp.BinInput(X_traincv_msb[i]), y_traincv[i])

        for i in len(X_testcv_lsb):
            ds_lsb_test.add(wp.BinInput(X_testcv_lsb[i]), y_testcv[i])
            ds_msb_test.add(wp.BinInput(X_testcv_msb[i]), y_testcv[i])
    
        ds_lsb_train.save('lsb_train')
        ds_msb_train.save('msb_train')
        ds_lsb_test.save('lsb_test')
        ds_msb_test.save('msb_test')

    else:
        print('Carregar')   

    print("Data imported")    

    wsd_lsb.train(X_traincv_lsb, y_traincv)
    wsd_msb.train(X_traincv_msb, y_traincv)
    
    print("Netword trained")

    print("Testing...")

    somePrediction = 0
    correctPrediction = 0
    out = []
    for i in range(len(X_testcv_lsb)):
        out_lsb    = wsd_lsb.classify([X_testcv_lsb[i]])
        out_msb    = wsd_msb.classify([X_testcv_msb[i]])
        expiro = 0
        neshta = 0
        sality = 0
        vba = 0
        for j in range(4): # "Expiro","Neshta","Sality","VBA"
            if out[i]['classesDegrees'][j]['class'] == 'Expiro':
                expiro += out[i]['classesDegrees'][j]['degree']
            elif out[i]['classesDegrees'][j]['class'] == 'Neshta':
                neshta += out[i]['classesDegrees'][j]['degree']
            elif out[i]['classesDegrees'][j]['class'] == 'Sality':
                sality += out[i]['classesDegrees'][j]['degree']
            elif out[i]['classesDegrees'][j]['class'] == 'VBA':
                vba += out[i]['classesDegrees'][j]['degree']
        results = [expiro, neshta, sality, vba]
        largest = max(results)
        if results.index(largest) == 0:
            out.append('Expiro')
        elif results.index(largest) == 1:
            out.append('Neshta')
        elif results.index(largest) == 1:
            out.append('Sality')
        elif results.index(largest) == 1:
            out.append('VBA')

    print("Tests done")

# ClusWiSARD
if mental_images:
    print("ClusWiSARD")

    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","Neshta","Sality","VBA"])
    
    minScore = 0.1
    threshold = 10
    discriminatorLimit = 4

    clus = wp.ClusWisard(20, minScore, threshold, discriminatorLimit, verbose = True)

    print("Training...")

    clus.train(X_traincv,y_traincv)

    print("Getting patterns...")

    patterns = clus.getMentalImages()

    for label, mental_image in patterns.items():
        if label == "Expiro":
            expiro = mental_image[3]
        elif label == "Neshta":
            neshta = mental_image[3]
        elif label == "Sality":
            sality = mental_image[3]
        elif label == "VBA":
            vba = mental_image[0]
  
    expiro_img = np.reshape(expiro, (224,224))
    neshta_img = np.reshape(neshta, (224,224))
    sality_img = np.reshape(sality, (224,224))
    vba_img = np.reshape(vba, (224,224))

    cv2.imwrite("expiro_gray.png", expiro_img)
    cv2.imwrite("neshta_gray.png", neshta_img)
    cv2.imwrite("sality_gray.png", sality_img)
    cv2.imwrite("vba_gray.png", vba_img)


### WiSARD
if wisard_pairs_voting:

    print("WiSARD")

    bleachingActivated=True
    ignoreZero=False

    wsd02  = wp.Wisard( 2, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd04  = wp.Wisard( 4, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd08  = wp.Wisard( 8, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd16  = wp.Wisard(16, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd32  = wp.Wisard(32, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd64  = wp.Wisard(64, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)

    print("Training...")

    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","Neshta","Sality"])
    
    wsd02.train(X_traincv, y_traincv)
    wsd04.train(X_traincv, y_traincv)
    wsd08.train(X_traincv, y_traincv)
    wsd16.train(X_traincv, y_traincv)
    wsd32.train(X_traincv, y_traincv)
    wsd64.train(X_traincv, y_traincv)
    
    print("Netword trained")

    print("Testing...")

    somePrediction = 0
    correctPrediction = 0
    for i in range(len(X_testcv)):
        out02    = wsd02.classify([X_testcv[i]])
        out04    = wsd04.classify([X_testcv[i]])
        out08    = wsd08.classify([X_testcv[i]])
        out16    = wsd16.classify([X_testcv[i]])
        out32    = wsd32.classify([X_testcv[i]])   
        out64    = wsd64.classify([X_testcv[i]])
        finalResult     = [out02[0], out04[0], out08[0], out16[0], out32[0], out64[0]]
        if most_frequent(finalResult) == "Expiro":
            print("Predicted Expiro, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Expiro":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Neshta":
            print("Predicted Neshta, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Neshta":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Sality":
            print("Predicted Sality, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Sality":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "VBA":
            print("Predicted VBA, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "VBA":
                correctPrediction += 1
            somePrediction += 1
        print("=========")
    
    print("Classified " + str(somePrediction) + " out of " + str(len(X_testcv)) + ": " + str(somePrediction/len(X_testcv)))   
    print("Correct " + str(correctPrediction) + " out of " + str(len(X_testcv)) + ": " + str(correctPrediction/len(X_testcv)))   
    print("Tests done")

    # the output of classify is a string list in the same sequence as the input
    #total = 0
    #corrects = 0
    #for count in range(len(y_testcv)):
    #    if y_testcv[count] == out[count]:
    #        corrects = corrects + 1
    #    total = total + 1
    #print("Chegou 1")
    #print("Keys: " + str(list(dict.fromkeys(y))))
    #clf_eval('wisard_virus_forget', y_testcv, out, classes = list(dict.fromkeys(y)))
    #new_y_testcv = y_testcv
    #new_out = out

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
   
    #f1.append(f1_score(new_y_testcv, new_out,average='micro'))
    #precision.append(precision_score(new_y_testcv, new_out, average='micro'))
    #recall.append(recall_score(new_y_testcv, new_out, average='micro'))
    #accuracy.append(float(corrects)/total)

   # print("f1 mean " + str(np.mean(f1)))
   # print("f1 std " + str(np.std(f1)))
   # print("Precision mean "+str(np.mean(precision)))
   # print("Precision std "+str(np.std(precision)))
   # print("Recall mean "+str(np.mean(recall)))
   # print("Recall std "+str(np.std(recall)))
   # print("Accuracy mean "+str(np.mean(accuracy)))
   # print("Accuracy std "+str(np.std(accuracy)))
   # print("==============================")

if wisard_pairs:

    print("WiSARD")

    addressSize = 20
    bleachingActivated=True
    ignoreZero=False

    ExpiroVBA     = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    ExpiroNeshta  = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    ExpiroSality  = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    NeshtaSality  = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    NeshtaVBA     = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    SalityVBA     = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)

    print("Training...")

    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","VBA","Neshta","Sality"])
    for i in range(len(X_traincv)):
        if y_traincv[i] == "Expiro":
            ExpiroVBA.    train([X_traincv[i]], [y_traincv[i]])      
            ExpiroNeshta. train([X_traincv[i]], [y_traincv[i]])      
            ExpiroSality. train([X_traincv[i]], [y_traincv[i]])
        elif y_traincv[i] == "Neshta":
            ExpiroNeshta. train([X_traincv[i]], [y_traincv[i]])
            NeshtaSality. train([X_traincv[i]], [y_traincv[i]])
            NeshtaVBA.    train([X_traincv[i]], [y_traincv[i]])
        elif y_traincv[i] == "Sality":
            ExpiroSality. train([X_traincv[i]], [y_traincv[i]])
            NeshtaSality. train([X_traincv[i]], [y_traincv[i]])
            SalityVBA.    train([X_traincv[i]], [y_traincv[i]])
        elif y_traincv[i] == "VBA":
            ExpiroVBA.    train([X_traincv[i]], [y_traincv[i]])
            NeshtaVBA.    train([X_traincv[i]], [y_traincv[i]])
            SalityVBA.    train([X_traincv[i]], [y_traincv[i]])

    print("Netword trained")

    print("Testing...")

    somePrediction = 0
    correctPrediction = 0
    for i in range(len(X_testcv)):
        outExpiroVBA    = ExpiroVBA.    classify([X_testcv[i]])
        outExpiroNeshta = ExpiroNeshta. classify([X_testcv[i]])
        outExpiroSality = ExpiroSality. classify([X_testcv[i]])
        outNeshtaSality = NeshtaSality. classify([X_testcv[i]])
        outNeshtaVBA    = NeshtaVBA.    classify([X_testcv[i]])   
        outSalityVBA    = SalityVBA.    classify([X_testcv[i]])
        finalResult     = [outExpiroVBA[0], outExpiroNeshta[0], outExpiroSality[0], outNeshtaSality[0], outNeshtaVBA[0], outSalityVBA[0]]
        if finalResult.count("Expiro") == 3:
            print("Predicted Expiro, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Expiro":
                correctPrediction += 1
            somePrediction += 1
        if finalResult.count("Neshta") == 3:
            print("Predicted Neshta, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Neshta":
                correctPrediction += 1
            somePrediction += 1
        if finalResult.count("Sality") == 3:
            print("Predicted Sality, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Sality":
                correctPrediction += 1
            somePrediction += 1
        if finalResult.count("VBA") == 3:
            print("Predicted VBA, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "VBA":
                correctPrediction += 1
            somePrediction += 1
        print("=========")
    
    print("Classified " + str(somePrediction) + " out of " + str(len(X_testcv)) + ": " + str(somePrediction/len(X_testcv)))   
    print("Correct " + str(correctPrediction) + " out of " + str(len(X_testcv)) + ": " + str(correctPrediction/len(X_testcv)))   
    print("Tests done")

    # the output of classify is a string list in the same sequence as the input
    #total = 0
    #corrects = 0
    #for count in range(len(y_testcv)):
    #    if y_testcv[count] == out[count]:
    #        corrects = corrects + 1
    #    total = total + 1
    #print("Chegou 1")
    #print("Keys: " + str(list(dict.fromkeys(y))))
    #clf_eval('wisard_virus_forget', y_testcv, out, classes = list(dict.fromkeys(y)))
    #new_y_testcv = y_testcv
    #new_out = out

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
   
    #f1.append(f1_score(new_y_testcv, new_out,average='micro'))
    #precision.append(precision_score(new_y_testcv, new_out, average='micro'))
    #recall.append(recall_score(new_y_testcv, new_out, average='micro'))
    #accuracy.append(float(corrects)/total)

    #print("f1 mean " + str(np.mean(f1)))
    #print("f1 std " + str(np.std(f1)))
    #print("Precision mean "+str(np.mean(precision)))
    #print("Precision std "+str(np.std(precision)))
    #print("Recall mean "+str(np.mean(recall)))
    #print("Recall std "+str(np.std(recall)))
    #print("Accuracy mean "+str(np.mean(accuracy)))
    #print("Accuracy std "+str(np.std(accuracy)))
    #print("==============================")

if wisard_voting:

    print("WiSARD")

    bleachingActivated=True
    ignoreZero=False

    wsd02  = wp.Wisard( 2, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd04  = wp.Wisard( 4, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd08  = wp.Wisard( 8, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd16  = wp.Wisard(16, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd32  = wp.Wisard(32, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)
    wsd64  = wp.Wisard(64, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)

    print("Training...")

    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","VBA","Neshta","Sality"])
    
    wsd02.train(X_traincv, y_traincv)
    wsd04.train(X_traincv, y_traincv)
    wsd08.train(X_traincv, y_traincv)
    wsd16.train(X_traincv, y_traincv)
    wsd32.train(X_traincv, y_traincv)
    wsd64.train(X_traincv, y_traincv)
    
    print("Netword trained")

    print("Testing...")

    somePrediction = 0
    correctPrediction = 0
    for i in range(len(X_testcv)):
        out02    = wsd02.classify([X_testcv[i]])
        out04    = wsd04.classify([X_testcv[i]])
        out08    = wsd08.classify([X_testcv[i]])
        out16    = wsd16.classify([X_testcv[i]])
        out32    = wsd32.classify([X_testcv[i]])   
        out64    = wsd64.classify([X_testcv[i]])
        finalResult     = [out02[0], out04[0], out08[0], out16[0], out32[0], out64[0]]
        if most_frequent(finalResult) == "Expiro":
            print("Predicted Expiro, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Expiro":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Neshta":
            print("Predicted Neshta, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Neshta":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Sality":
            print("Predicted Sality, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "Sality":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "VBA":
            print("Predicted VBA, \t\t\t real is " + str(y_testcv[i]))
            if y_testcv[i] == "VBA":
                correctPrediction += 1
            somePrediction += 1
        print("=========")
    
    print("Classified " + str(somePrediction) + " out of " + str(len(X_testcv)) + ": " + str(somePrediction/len(X_testcv)))   
    print("Correct " + str(correctPrediction) + " out of " + str(len(X_testcv)) + ": " + str(correctPrediction/len(X_testcv)))   
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


