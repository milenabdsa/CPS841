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

# Choose encoding style
THERMOMETER_STYLE = "one-hot"  # "one-hot" (old style) or "cumulative" (true thermometer)

circularThermometer = False
nBits = 4

LIGHT= False

range_max= 100 if LIGHT else 10000 
range_increase=100 if LIGHT else 1000
# Process both train and validation datasets
for dataset_type in ['train', 'val']:
    if dataset_type == 'val':
        range_max = 6000            
    for i in range(0,range_max,range_increase):

        # ==================================== #
        # DEBUG OPTIONS
        inicio = i
        fim = i + range_increase if i + range_increase < 9101 else 9100
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

        # Ajustar fim para não ultrapassar o tamanho real dos dados
        fim_real = min(fim, len(data))
        data = data[inicio:fim_real]
        y = y[inicio:fim_real]
        tamanho = len(data)  # Atualizar tamanho com o valor real

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
        data = np.array(data).reshape(-1, 224*224*3)
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
            if THERMOMETER_STYLE == "one-hot":
                print("One-hot encoding")
                X = np.zeros((tamanho, 224*224*N), dtype=np.uint8)
                for j in range(tamanho):
                    if j % 100 == 0:
                        print(f"  Processing image {j}/{tamanho}...")
                    
                    img = data[j][:224*224]  
                    levels = np.floor(img * N / 256).astype(int)
                    levels = np.clip(levels, 0, N-1)
                    
                    indices = N * np.arange(len(img)) + levels
                    X[j, indices] = 1
                
                print("  One-hot encoding complete!")
                
            elif THERMOMETER_STYLE == "cumulative":
                print("Thermometer encoding (optimized - true thermometer)...")
                X = np.zeros((tamanho, 224*224*3*N), dtype=np.uint8)

                for j in range(tamanho):
                    if j % 100 == 0:
                        print(f"  Processing image {j}/{tamanho}...")
                    
                    img = data[j]
                    levels = np.floor(img * N / 256).astype(int)
                    levels = np.clip(levels, 0, N-1)

                    for i in range(len(img)):
                        level = levels[i]
                        X[j, N*i:N*i + level + 1] = 1
                
                print("  Thermometer encoding complete!")

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
