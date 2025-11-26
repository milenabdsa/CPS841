import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback

#X_traincv, X_testcv, y_traincv, y_testcv = train_test_split(X,
#                                                            y,
#                                                            test_size=0.3)

from tensorflow.keras.applications import DenseNet201

nn = DenseNet201(
    include_top=True,
    weights=None
)

nn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

stats_accu = []
stats_loss = []

batch_size = 64
epochs = 60

class NBatchLogger(Callback):
    "A Logger that log average performance per `display` steps."

    def __init__(self, display=100):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_train_batch_end(self, batch, logs={}):
        self.step += 1
        #print(logs)
        stats_accu.append(logs['accuracy'])
        stats_loss.append(logs['loss'])
        #for k in self.params['metrics']:
         #   print(k)


