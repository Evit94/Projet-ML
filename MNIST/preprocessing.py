import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

#Chargement des données
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

#Reshape 
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

#Normalisation entre 0 et 1
X_train = X_train / 255.0
X_test = X_test / 255.0

#One-hot encoding des étiquettes
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)