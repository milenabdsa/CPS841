import matplotlib
import matplotlib.pyplot as plt
import wisardpkg as wp
import numpy as np
import cv2
import gist

img = cv2.imread('malevis_train_val_224x224/control.png')
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
