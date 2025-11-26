# Import

import numpy as np
import pickle
import os
import cv2
import gc

def load_data():
    img = load_images_from_folder('malevis_train_val_224x224/train/Adposhel')
    data = np.array(img).reshape(350,224*224*3)
    y = ['Adposhel']*350
    img = load_images_from_folder('malevis_train_val_224x224/train/Agent')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Agent']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Allaple')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Allaple']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Amonetize')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Amonetize']*350),axis=0)

    img = load_images_from_folder('malevis_train_val_224x224/train/Androm')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Androm']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Autorun')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Autorun']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/BrowseFox')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['BrowseFox']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Dinwod')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Dinwod']*350),axis=0)

    img = load_images_from_folder('malevis_train_val_224x224/train/Elex')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Elex']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Expiro')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Expiro']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Fasong')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Fasong']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/HackKMS')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['HackKMS']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Hlux')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Hlux']*350),axis=0)
    
    img = load_images_from_folder('malevis_train_val_224x224/train/Injector')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Injector']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/InstallCore')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['InstallCore']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/MultiPlug')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['MultiPlug']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Neoreklami')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Neoreklami']*350),axis=0)
    
    img = load_images_from_folder('malevis_train_val_224x224/train/Neshta')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Neshta']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Other')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Other']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Regrun')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Regrun']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Sality')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Sality']*350),axis=0)
    
    img = load_images_from_folder('malevis_train_val_224x224/train/Snarasite')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Snarasite']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Stantinko')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Stantinko']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/VBA')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['VBA']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/VBKrypt')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['VBKrypt']*350),axis=0)
    img = load_images_from_folder('malevis_train_val_224x224/train/Vilsel')
    data = np.concatenate((data, np.array(img).reshape(350,224*224*3)), axis=0)
    y = np.concatenate((y,['Vilsel']*350),axis=0)
    return data,y

def threshold(data, start, end):
    X = []
    for count in range(int(start),int(end)):
        atual = []
        for i in range(len(data[count])):
            if data[count][i] > THRESHOLD:
                atual.append(1)
            else:
                atual.append(0)
        X.append(atual)
    return X

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

THRESHOLD = 127
data, y = load_data()
length = len(data)
pickle.dump(y, open( "y.p", "wb" ))
del y
gc.collect()

f = open( "X.p", "wb" )
blocks = 100
for k in range(blocks):
    X = threshold(data, k*length/blocks, (k+1)*length/blocks)
    print(str(k+1)+" out of "+str(blocks))
    pickle.dump(X, f)
    del X
    gc.collect()

f.close()
