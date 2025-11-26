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
from sklearn import decomposition
from sklearn import tree

def most_frequent(List): 
    return max(set(List), key = List.count) 

def clf_eval(name, y_true, y_pred, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
    clf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    print(clf_matrix)
    
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            #images.append(gist.extract(img))
            images.append(img)
    return images

def import_data(classes, therm = False):
  print("Importing...")
  first = True
  if "Expiro" in classes: #### VIRUS # EXPIRO
    img = load_images_from_folder('malevis_train_val_224x224/train/Expiro')
    if first: # FIRST?
      length = len(img)
      data = img
      if therm:
        y = [[1,0,0]]*length
      else:
        y = ['Expiro']*length
      first = False
    else:
      data = np.concatenate((data, img), axis=0)
      if therm:
        y = np.concatenate((y,[[1,0,0]]*length),axis=0)
      else:
        y = np.concatenate((y,["Expiro"]*length),axis=0)
    del img
    gc.collect()
  print("OK")

  if "Neshta" in classes: #### VIRUS # NESHTA
    img = load_images_from_folder('malevis_train_val_224x224/train/Neshta')
    if first: # FIRST?
      length = len(img)
      data = img
      if therm:
        y = [[0,1,0]]*length
      else:
        y = ['Neshta']*length
      first = False
    else:
      data = np.concatenate((data, img), axis=0)
      if therm:
        y = np.concatenate((y,[[0,1,0]]*length),axis=0)
      else:
        y = np.concatenate((y,["Neshta"]*length),axis=0)
    del img
    gc.collect()
  print("OK")

  if "Sality" in classes: #### VIRUS SALITY
    img = load_images_from_folder('malevis_train_val_224x224/train/Sality')
    if first: # FIRST?
      length = len(img)
      data = img
      if therm:
        y = [[0,0,1]]*length
      else:
        y = ['Sality']*length
      first = False
    else:
      data = np.concatenate((data, img), axis=0)
      if therm:
        y = np.concatenate((y,[[0,0,1]]*length),axis=0)
      else:
        y = np.concatenate((y,["Sality"]*length),axis=0)
    del img
    gc.collect()
  print("OK")

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
  print("OK")
  data = np.resize(data, (len(data),224*224*3))
  # data = np.resize(data, (len(data),32*32*3))
  
  print("Preprocessing...")
  # Dynamic
  for i in range(len(data)):
      med = np.median(data[i])
      data[i] = np.where(data[i] > med, 1, 0)
  X = data
  del data
  gc.collect()

  print("Done!")
  X = list(X)
  #X = list(data)
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

resize = False
fft = False
dt_voting = False
dt = True
pca = False
mental_images = False
wisard_pairs_voting = False
wisard_pairs = False
wisard_voting = False
cluswisard = False

if resize:
    print("WiSARD")

    bleachingActivated=True
    ignoreZero=False

    wsd32  = wp.Wisard(32, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)

    print("Training...")

    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","VBA","Neshta","Sality"])
    
    wsd32.train(X_traincv, y_traincv)
    
    print("Netword trained")

    print("Testing...")

    somePrediction = 0
    correctPrediction = 0
    for i in range(len(X_testcv)):
        out32    = wsd32.classify([X_testcv[i]])   
        finalResult     = [out32[0], out32[0]]
        if most_frequent(finalResult) == "Expiro":
            if y_testcv[i] == "Expiro":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Neshta":
            if y_testcv[i] == "Neshta":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Sality":
            if y_testcv[i] == "Sality":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "VBA":
            if y_testcv[i] == "VBA":
                correctPrediction += 1
            somePrediction += 1
        print("=========")
    
    print("Classified " + str(somePrediction) + " out of " + str(len(X_testcv)) + ": " + str(somePrediction/len(X_testcv)))   
    print("Correct " + str(correctPrediction) + " out of " + str(len(X_testcv)) + ": " + str(correctPrediction/len(X_testcv)))   
    print("Tests done")

    patterns = wsd32.getMentalImages()

    expiro = patterns["Expiro"]
    neshta = patterns["Neshta"]
    sality = patterns["Sality"]
    vba = patterns["VBA"]
	
    max_expiro = max(expiro)
    max_neshta = max(neshta)
    max_sality = max(sality)
    max_vba = max(vba)


    for i in range(len(expiro)):
        expiro[i] = 255*expiro[i]/max_expiro
        neshta[i] = 255*neshta[i]/max_neshta
        sality[i] = 255*sality[i]/max_sality
        vba[i] = 255*vba[i]/max_vba

    expiro_img = np.reshape(expiro, (224,224,3))
    neshta_img = np.reshape(neshta, (224,224,3))
    sality_img = np.reshape(sality, (224,224,3))
    vba_img = np.reshape(vba, (224,224,3))

    cv2.imwrite("expiro.png", expiro_img)
    cv2.imwrite("neshta.png", neshta_img)
    cv2.imwrite("sality.png", sality_img)
    cv2.imwrite("vba.png", vba_img)

if fft:
    print("FFT")
    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro"])
    a = np.fft.fft(X_traincv[0]/255)
    a = np.abs(a)
    a = a/max(a)
    a = a*255
    
    a = np.reshape(np.array(a), (224,224,3))
    cv2.imwrite("fft.png",a)

if dt_voting:
    print("WiSARD")

    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","Neshta","Sality"])
    clf_en = tree.DecisionTreeClassifier(criterion='gini', splitter='random') # BEST
    clf_ns = tree.DecisionTreeClassifier(criterion='gini', splitter='random') # BEST
    clf_es = tree.DecisionTreeClassifier(criterion='gini', splitter='random') # BEST
    
    for i in range(len(X_traincv)):
        if y_traincv[i] == "Expiro":
            clf_en = clf_en.fit([X_traincv[i]], [y_traincv[i]])
            clf_es = clf_es.fit([X_traincv[i]], [y_traincv[i]])
        elif y_traincv[i] == "Neshta":
            clf_en = clf_en.fit([X_traincv[i]], [y_traincv[i]])
            clf_ns = clf_ns.fit([X_traincv[i]], [y_traincv[i]])
        elif y_traincv[i] == "Sality":
            clf_es = clf_es.fit([X_traincv[i]], [y_traincv[i]])
            clf_ns = clf_ns.fit([X_traincv[i]], [y_traincv[i]])
 
    correctPrediction = 0
    somePrediction = 0

    for i in range(len(X_testcv)):  
        finalResult = [clf_en.predict([X_testcv[i]])[0],clf_es.predict([X_testcv[i]])[0],clf_ns.predict([X_testcv[i]])[0]]
        if most_frequent(finalResult) == "Expiro":
            if y_testcv[i] == "Expiro":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Neshta":
            if y_testcv[i] == "Neshta":
                correctPrediction += 1
            somePrediction += 1
        if most_frequent(finalResult) == "Sality":
            if y_testcv[i] == "Sality":
                correctPrediction += 1
            somePrediction += 1
        print("=========")

    print(correctPrediction/somePrediction)

if dt:
    print("WiSARD")

    bleachingActivated=True
    ignoreZero=False

    wsd  = wp.Wisard(20, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = False)

    print("Training...")

    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","Neshta","Sality"])
    
    wsd.train(X_traincv, y_traincv)
    
    print("Netword trained")

    patterns = wsd.getMentalImages()

    expiro = patterns["Expiro"]
    neshta = patterns["Neshta"]
    sality = patterns["Sality"]
	
    max_expiro = max(expiro)
    max_neshta = max(neshta)
    max_sality = max(sality)

    for i in range(len(expiro)):
        expiro[i] = 255*expiro[i]/max_expiro
        neshta[i] = 255*neshta[i]/max_neshta
        sality[i] = 255*sality[i]/max_sality

    clf = tree.DecisionTreeClassifier(criterion='gini', splitter='random') # BEST
    clf = clf.fit([expiro, neshta, sality], ["Expiro","Neshta","Sality"])
    
    correct = 0
    total = 0
    out = clf.predict(X_testcv)
    for i in range(len(y_testcv)):
        total += 1
        if out[i] == y_testcv[i]:
            correct += 1

    print(correct/total)

    tree.export_graphviz(clf, out_file="decision.dot")

if pca:
    print("WiSARD")
    saveImage = False
    wsd = wp.Wisard(20, bleachingActivated = True, ignoreZero = False, verbose = True)
    n_components = 1

    # ===================================================================================================== #    

    X_traincv, X_testcv_expiro, y_traincv, y_testcv_expiro = import_data(["Expiro"])
    
    X_traincv = np.array(X_traincv)

    print("PCA")
    n_samples, n_features = X_traincv.shape
    estimator = decomposition.PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    digits_recons = estimator.inverse_transform(estimator.fit_transform(X_traincv))
    print("Done!")

    digits_recons = list(digits_recons)

    for i in range(len(digits_recons)):
        med = np.median(digits_recons[i])
        digits_recons[i] = list(np.where(digits_recons[i] > med, 1, 0))
    
    wsd.train(digits_recons, y_traincv)

    if saveImage:
        X_train_img = np.reshape(X_traincv[5], (224,224))
        X_train_pca_img = np.reshape(digits_recons[5], (224,224))
    
        cv2.imwrite("train_expiro.png", X_train_img)
        cv2.imwrite("train_expiro_pca.png", X_train_pca_img)

    # ===================================================================================================== #

    X_traincv, X_testcv_neshta, y_traincv, y_testcv_neshta = import_data(["Neshta"])
    
    X_traincv = np.array(X_traincv)

    print("PCA")
    n_samples, n_features = X_traincv.shape
    estimator = decomposition.PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    digits_recons = estimator.inverse_transform(estimator.fit_transform(X_traincv))
    print("Done!")
    
    digits_recons = list(digits_recons)

    for i in range(len(digits_recons)):
        med = np.median(digits_recons[i])
        digits_recons[i] = list(np.where(digits_recons[i] > med, 1, 0))
    
    wsd.train(digits_recons, y_traincv)

    if saveImage:
        X_train_img = np.reshape(X_traincv[5], (224,224))
        X_train_pca_img = np.reshape(digits_recons[5], (224,224))
    
        cv2.imwrite("train_neshta.png", X_train_img)
        cv2.imwrite("train_neshta_pca.png", X_train_pca_img)

    # ===================================================================================================== #

    X_traincv, X_testcv_sality, y_traincv, y_testcv_sality = import_data(["Sality"])
    
    X_traincv = np.array(X_traincv)

    print("PCA")
    n_samples, n_features = X_traincv.shape
    estimator = decomposition.PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    digits_recons = estimator.inverse_transform(estimator.fit_transform(X_traincv))
    print("Done!")
    
    digits_recons = list(digits_recons)

    for i in range(len(digits_recons)):
        med = np.median(digits_recons[i])
        digits_recons[i] = list(np.where(digits_recons[i] > med, 1, 0))
    
    wsd.train(digits_recons, y_traincv)

    if saveImage:
        X_train_img = np.reshape(X_traincv[5], (224,224))
        X_train_pca_img = np.reshape(digits_recons[5], (224,224))
    
        cv2.imwrite("train_sality.png", X_train_img)
        cv2.imwrite("train_sality_pca.png", X_train_pca_img)

    correct = 0
    total = 0

    for i in range(len(X_testcv_expiro)):
        med = np.median(X_testcv_expiro[i])
        X_testcv_expiro[i] = list(np.where(X_testcv_expiro[i] > med, 1, 0))
    
    for i in range(len(X_testcv_neshta)):
        med = np.median(X_testcv_neshta[i])
        X_testcv_neshta[i] = list(np.where(X_testcv_neshta[i] > med, 1, 0))
    
    for i in range(len(X_testcv_sality)):
        med = np.median(X_testcv_sality[i])
        X_testcv_sality[i] = list(np.where(X_testcv_sality[i] > med, 1, 0))

    out_expiro = wsd.classify(X_testcv_expiro)
    for i in range(len(out_expiro)):
        if out_expiro[i] == "Expiro":
            correct += 1
        total += 1
    out_neshta = wsd.classify(X_testcv_neshta)
    for i in range(len(out_neshta)):
        if out_neshta[i] == "Neshta":
            correct += 1
        total += 1
    out_sality = wsd.classify(X_testcv_sality)
    for i in range(len(out_sality)):
        if out_sality[i] == "Sality":
            correct += 1
        total += 1

    print(correct/total)




if False:

    print("Training...")

    wsd.train(X_traincv,y_traincv)



if mental_images:
    print("WiSARD")

    X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","Neshta","Sality","VBA"])
    
    minScore = 0.1
    threshold = 10
    discriminatorLimit = 4

    wsd = wp.Wisard(20, bleachingActivated = True, ignoreZero = False, verbose = True)

    print("Training...")

    wsd.train(X_traincv,y_traincv)

    print("Getting patterns...")

    patterns = wsd.getMentalImages()

    expiro = patterns["Expiro"]
    neshta = patterns["Neshta"]
    sality = patterns["Sality"]
    vba = patterns["VBA"]

if False:
    expiro = np.where(expiro > np.mean(expiro), 1, 0)
    neshta = np.where(neshta > np.mean(neshta), 1, 0)
    sality = np.where(sality > np.mean(sality), 1, 0)
    vba = np.where(vba > np.mean(vba), 1, 0)
    
    wsd2 = wp.Wisard(20, bleachingActivated = True, ignoreZero = False, verbose = True)

    wsd2.train([expiro],["Expiro"])
    wsd2.train([neshta],["Neshta"])
    wsd2.train([sality],["Sality"])
    wsd2.train([vba],["VBA"])

    trained = 0
    used = 0
    for i in range(len(X_traincv)):
        out = wsd2.classify([X_traincv[i]])
        trained += 1
        print(out[0])
        print(y_traincv[i])
        if out[0] == y_traincv[i]:
            wsd2.train([X_traincv[i]], [y_traincv[i]])
            used += 1
        else:
            if random.random() < 0.3:
                wsd2.train([X_traincv[i]], [y_traincv[i]])
                used += 1
    
    out = wsd2.classify(X_testcv)
    print("Tried to train "+str(trained)+", but trained "+str(used)+"\n")
    clf_eval('forget.png', y_testcv, out, classes = list(dict.fromkeys(y_testcv)))


if False:

    for i in range(len(expiro)):
        expiro[i] = 255*expiro[i]
        neshta[i] = 255*neshta[i]
        sality[i] = 255*sality[i]
        vba[i] = 255*vba[i]

    expiro_img = np.reshape(expiro, (224,224))
    neshta_img = np.reshape(neshta, (224,224))
    sality_img = np.reshape(sality, (224,224))
    vba_img = np.reshape(vba, (224,224))

    cv2.imwrite("expiro.png", expiro_img)
    cv2.imwrite("neshta.png", neshta_img)
    cv2.imwrite("sality.png", sality_img)
    cv2.imwrite("vba.png", vba_img)


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


