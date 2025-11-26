# =========================================================================================================#
# ======================================== IMPORT MODULES =================================================#
# =========================================================================================================#
print("Importing modules...")

import numpy as np
import pickle
import multiprocessing
import cv2
from sklearn.preprocessing import binarize

print("Done!")
# =========================================================================================================#
# ====================================== EXPERIMENT PARAMETERS ============================================#
# =========================================================================================================#

# ==================================== #
# HOW DO YOU WANT TO IMPORT THE DATA?
binaryClassification = False
multiClassification = True
# ==================================== #
# DO YOU WANT TO INTERPOLATE?
interpolate = False
FACTOR = 224
# ==================================== #
# WHICH BINARIZATION DO YOU WANT TO USE?
simple = False
THRESHOLD = 127

dynamic = False

thermometer = True
N = 12

circularThermometer = False
nBits = 4

# Process both train and validation datasets
for dataset_type in ['train', 'val']:
    for i in range(0,10000,1000):
        # ==================================== #
        # DEBUG OPTIONS
        inicio = i
        fim = i + 1000 if i + 1000 < 9101 else 9100
        tamanho = fim-inicio

        # ==================================== #
        # FILE TO SAVE
        filename = f'thermometer12_{dataset_type}_' + str(inicio) + '_' + str(fim-1)


        # =========================================================================================================#
        # ====================================== IMPORT DATA ======================================================#
        # =========================================================================================================#
        print(f"Importing {dataset_type} data...")

        # Load appropriate dataset
        if dataset_type == 'train':
            data = pickle.load( open( "data.p", "rb" ) )
            if binaryClassification:
                y = pickle.load( open( "y2.p", "rb" ) )
            if multiClassification:
                y = pickle.load( open( "y26.p", "rb" ) )
        else:  # val
            data = pickle.load( open( "data_val.p", "rb" ) )
            if binaryClassification:
                y = pickle.load( open( "y2_val.p", "rb" ) )
            if multiClassification:
                y = pickle.load( open( "y26_val.p", "rb" ) )

        #y_6 = pickle.load( open( "y6.p", "rb" ) )
        print("Done!")

        data = data[inicio:fim]
        y = y[inicio:fim]

        # =========================================================================================================#
        # ========================================== INTERPOLATION ================================================#
        # =========================================================================================================#
        if interpolate:
            print("Starting interpolation...")
            new_data = []

            for i in range(len(data)):
                new_data.append(cv2.resize(data[i], dsize=(FACTOR, FACTOR), interpolation=cv2.INTER_CUBIC))
            data = new_data
            print("Done!")

        # =========================================================================================================#
        # ========================================== RESHAPING ====================================================#
        # =========================================================================================================#
        print("Reshaping...")   
        data = np.array(data).reshape(tamanho, 224*224*3)
        print("Done!")

        # =========================================================================================================#
        # ========================================= BINARIZATION ==================================================#
        # =========================================================================================================#
        print("Binarization...")

        if simple:
            X = binarize(data, threshold = THRESHOLD)

        if dynamic:
            for i in range(len(data)):
                med = np.median(data[i])
                data[i] = np.where(data[i] > med, 1, 0)
            X = data

        if thermometer:
            X = [[0 for i in range(224*224*N)] for j in range(tamanho)]

            for j in range(tamanho):
                # print(j)
                for i in range(224*224):
                    for k in range(N):
                        if data[j][i] >= k*256/N and data[j][i] < (k+1)*256/N:
                            X[j][N*i+k] = 1
                            break

        if circularThermometer:
            X = [[0 for i in range(224*224*nBits)] for j in range(tamanho)]

            for j in range(tamanho):
                print(j)
                for i in range(224*224):
                    for k in range(nBits):
                        if data[j][i] >= k*256/nBits and data[j][i] <= (k+1)*256/nBits:
                            for l in range(int(nBits/2)):
                                X[j][nBits*i + (k+l) % nBits] = 1
                            break

        pickle.dump ( X, open( filename + "X.p", "wb" ) )
        pickle.dump ( y, open( filename + "y.p", "wb" ) )

        print(f"Done - {filename}")
