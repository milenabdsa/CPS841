print("Importing modules...")

import numpy as np
import pickle
import cv2
from sklearn.preprocessing import binarize
import sys
import time
import os

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

THRESHOLD = int(sys.argv[1])

classes = ["Adposhel","Agent","Allaple","Amonetize",
			"Androm","Autorun","BrowseFox","Dinwod",
			"Elex","Expiro","Fasong","HackKMS","Hlux",
			"Injector","InstallCore","MultiPlug","Neoreklami",
			"Neshta","Other","Regrun","Sality","Snarasite",
			"Stantinko","VBA","VBKrypt","Vilsel"]

total_time = 0
total_samples = 0

for class_to_process in classes:
	print(class_to_process)
	img = load_images_from_folder('../../malevis_train_val_224x224/train/' + class_to_process)
	data = np.array(img).reshape(len(img), 224*224*3)
	
	start = time.time()
	X = binarize(data, threshold = THRESHOLD)
	end = time.time()

	pickle.dump ( X, open("bt/" + class_to_process + "_bt_" + str(THRESHOLD) + ".p", "wb" ) )

	total_time += (end-start)
	total_samples += len(img)

print("Mean time: " + str(1000*total_time/total_samples))

print("Done")
