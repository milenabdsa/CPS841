import matplotlib.pyplot as plt
import wisardpkg as wp
import numpy as np
import pandas as pd
import time
import cv2
import os
from unary_coding import inverted_unary

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def clf_eval(name, y_true, y_pred, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
    clf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    print(clf_matrix)
    
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            #images.append(gist.extract(img))
            images.append(img)
    return images

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

  data = np.resize(data, (len(data),224*224*3))

  X = list(data)
  y = list(y)
  
  SPLIT_SIZE = 0.3
  X_traincv, X_testcv, y_traincv, y_testcv = model_selection.train_test_split(X,
                                                                              y,
                                                                              test_size=SPLIT_SIZE,
                                                                              random_state=0)

  return X_traincv, X_testcv, y_traincv, y_testcv


X_traincv, X_testcv, y_traincv, y_testcv = import_data(["Expiro","Neshta","Sality"])
wsd = wp.Wisard(20, bleachingActivated = True, ignoreZero = False, verbose = True)

for i in range(len(X_traincv)):
    X = np.zeros(224*224*3*256, dtype=int)
    for j in range(len(X_traincv[i])):
        for k in range(1,X_traincv[i][j]):
            X[256*(j+1)-k] = 1
    wsd.train([list(X)],[y_traincv[i]])

out = []
for i in range(len(X_testcv)):
    X = np.zeros(224*224*3*256)
    for j in range(len(X_testcv[i])):
        for k in range(1,X_testcv[i][j]):
            X[256*(j+1)-k] = 1
    out.append(wsd.classify([list(X)],[y_testcv[i]]))

print(out)
