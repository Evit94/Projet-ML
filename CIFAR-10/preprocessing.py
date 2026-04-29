import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import numpy as np
from tensorflow.keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

R_train = X_train[:,:,:,0]
G_train = X_train[:,:,:,1]
B_train = X_train[:,:,:,2]
X_gris_train = 0.299*R_train + 0.587*G_train + 0.114*B_train
X_gris_train = X_gris_train.reshape(50000, 1024)

R_test = X_test[:,:,:,0]
G_test = X_test[:,:,:,1]
B_test = X_test[:,:,:,2]
X_gris_test = 0.299*R_test + 0.587*G_test + 0.114*B_test
X_gris_test = X_gris_test.reshape(10000,1024)

X_train = X_train.reshape(50000,3072)
X_test = X_test.reshape(10000,3072)

#Normalisation 
X_gris_train = X_gris_train/255.0
X_gris_test = X_gris_test/255.0

X_train = X_train/255.0
X_test = X_test/255.0

#One-hot encoding
Y_train = to_categorical(Y_train.flatten())
Y_test = to_categorical(Y_test.flatten())
