import numpy as np
import pandas as pd
import pickle
import os
import cv2
import gc

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

print("Importing data")

img = load_images_from_folder('malevis_train_val_224x224/train/Adposhel')
length = len(img)
data = img
y_26 = ['Adposhel']*length
y_2 = ['Malign']*length
y_6 = ['adware']*length
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Agent')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Agent']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Trojan']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Allaple')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Allaple']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Worm']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Amonetize')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Amonetize']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['adware']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Androm')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Androm']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Backdoor']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Autorun')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Autorun']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Worm']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/BrowseFox')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['BrowseFox']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['adware']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Dinwod')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Dinwod']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Trojan']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Elex')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Elex']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Trojan']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Expiro')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Expiro']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Virus']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Fasong')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Fasong']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Worm']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/HackKMS')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['HackKMS']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Trojan']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Hlux')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Hlux']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Worm']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Injector')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Injector']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Trojan']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/InstallCore')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['InstallCore']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['adware']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/MultiPlug')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['MultiPlug']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['adware']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Neoreklami')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Neoreklami']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['adware']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Neshta')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Neshta']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Virus']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Other')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Benign']*length),axis=0)
y_2 = np.concatenate((y_2,['Benign']*length),axis=0)
y_6 = np.concatenate((y_6,['Benign']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Regrun')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Regrun']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Trojan']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Sality')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Sality']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Virus']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Snarasite')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Snarasite']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Trojan']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Stantinko')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['Stantinko']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Backdoor']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/VBA')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['VBA']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Virus']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/VBKrypt')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['VBKrypt']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Trojan']*length),axis=0)
del img
gc.collect()

img = load_images_from_folder('malevis_train_val_224x224/train/Vilsel')
length = len(img)
data = np.concatenate((data, img), axis=0)
y_26 = np.concatenate((y_26,['MultiPlug']*length),axis=0)
y_2 = np.concatenate((y_2,['Malign']*length),axis=0)
y_6 = np.concatenate((y_6,['Trojan']*length),axis=0)
del img
gc.collect()

pickle.dump( data, open( "data.p", "wb" ) )
pickle.dump( y_26, open( "y26.p", "wb" ) )
pickle.dump( y_2, open( "y2.p", "wb" ) )
pickle.dump( y_6, open( "y6.p", "wb" ) )
