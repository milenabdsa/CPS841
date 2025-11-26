print("Importing modules...")

import numpy as np
import pickle
import cv2
from sklearn.preprocessing import binarize
import sys
import time
import os
from multiprocessing import Pool

print("Done!")

# =========================================================================================================#
# ====================================== IMPORT DATA ======================================================#
# =========================================================================================================#
print("Importing data...")

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

nBits = int(sys.argv[1])

classes = ["Adposhel","Agent","Allaple","Amonetize",
			"Androm","Autorun","BrowseFox","Dinwod",
			"Elex","Expiro","Fasong","HackKMS","Hlux",
			"Injector","InstallCore","MultiPlug","Neoreklami",
			"Neshta","Other","Regrun","Sality","Snarasite",
			"Stantinko","VBA","VBKrypt","Vilsel"]

total_time = 0
total_samples = 0

def preprocess(class_to_process):
	print(class_to_process)
	img = load_images_from_folder('../../malevis_train_val_224x224/train/' + class_to_process)
	data = np.array(img).reshape(len(img), 224*224*3)
	tamanho = len(img)

#	start = time.time()
	# --------------------
	
    X = [[0 for i in range(224*224*nBits)] for j in range(tamanho)]

    for j in range(tamanho):
        print(j)
        for i in range(224*224):
            for k in range(nBits):
                if data[j][i] >= k*256/nBits and data[j][i] <= (k+1)*256/nBits:
                    for l in range(int(nBits/2)):
                        X[j][nBits*i + (k+l) % nBits] = 1
                    break

	# --------------------
#	end = time.time()

	pickle.dump ( X, open("cr/" + class_to_process + "_cr_" + str(nBits) + ".p", "wb" ) )

#	total_time += (end-start)
#	total_samples += len(img)

pool = Pool()
pool.map(preprocess, classes)

#for class_to_process in classes:
#	preprocess(class_to_process)

#print("Mean time: " + str(1000*total_time/total_samples))

print("Done")
