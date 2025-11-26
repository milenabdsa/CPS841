import numpy as np
import os
import cv2
import wisardpkg as wp
import sys

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

class_to_process = sys.argv[1]
nBits = int(sys.argv[2])

img = load_images_from_folder('malevis_train_val_224x224/train/' + class_to_process)
length = len(img)
y_26 = [class_to_process]*length
y_6 = ['virus']*length
y_2 = ['Malign']*length

data = np.array(img).reshape(len(img), 224*224*3)
new_data = []

if False:
    X = [[0 for i in range(224*224*3*nBits)] for j in range(len(data))]

    for j in range(len(data)):
        for i in range(224*224*3):
            for k in range(nBits):
                if data[j][i] >= k*256/nBits and data[j][i] <= (k+1)*256/nBits:
                    for l in range(int(nBits/2)):
                        X[j][nBits*i + (k+l) % nBits] = 1
                    break
else:
    for i in range(len(data)):
        new_data.append(data[i][75*224*3:150*224*3])
    data = new_data
    X = [[0 for i in range(75*224*3*nBits)] for j in range(len(data))]

    for j in range(len(data)):
        for i in range(75*224*3):
            for k in range(nBits):
                if data[j][i] >= k*256/nBits and data[j][i] <= (k+1)*256/nBits:
                    for l in range(int(nBits/2)):
                        X[j][nBits*i + (k+l) % nBits] = 1
                    break

ds_all = wp.DataSet()
ds_classes = wp.DataSet()
ds_bin = wp.DataSet()

for i in range(len(X)):
  ds_all.add(wp.BinInput(X[i]),y_26[i])
  ds_classes.add(wp.BinInput(X[i]),y_6[i])
  ds_bin.add(wp.BinInput(X[i]),y_2[i])

ds_all.save(class_to_process + "_crop_all")
ds_classes.save(class_to_process + "_crop_classes")
ds_bin.save(class_to_process + "_crop_bin")
