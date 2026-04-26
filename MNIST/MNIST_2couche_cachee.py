#MODELE AVEC DEUX COUCHES CACHEES

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

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

def softmax(o):
    exp_o = np.exp(o-np.max(o))
    return exp_o/np.sum(exp_o)

def cross_entropy(P, y):
    L = -np.sum(y * np.log(P))
    return L

def ReLU(x):
    return np.maximum(0,x)

def relu_derivee(x):
    return (x>0).astype(float)

np.random.seed(42)
p1= 128
p2 = 64
W1 = np.random.randn(p1, 784)*0.01
b1 = np.zeros((p1,1))
W2 = np.random.randn(p2, p1)*0.01
b2 = np.zeros((p2, 1))
W3 = np.random.randn(10, p2) * 0.01
b3 = np.zeros((10, 1))

def forward(x, W1, b1, W2, b2, W3, b3):
    o1 = W1 @ x + b1
    z1 = ReLU(o1)
    o2 = W2 @ z1 + b2
    z2 = ReLU(o2)
    o3 = W3 @ z2 + b3
    P = softmax(o3)
    return P, z2, o2, z1, o1

def gradient(x, y, P, z2, o2, W3, z1, o1, W2):
    dL_do3 = P - y
    dW3 = dL_do3 @ z2.T  #règle de la chaine : do3/dW3 = z2
    db3 = dL_do3 

    dL_dz2 = W3.T @ dL_do3

    dL_do2 = dL_dz2 * relu_derivee(o2)
    dW2 = dL_do2 @ z1.T
    db2 = dL_do2

    dL_dz1 = W2.T @ dL_do2

    dL_do1 = dL_dz1 * relu_derivee(o1)
    dW1 = dL_do1 @ x.T
    db1 = dL_do1

    return dW1, db1, dW2, db2, dW3, db3

def train(x, y, W1, b1, W2, b2, W3, b3, lr = 0.01, epochs = 10):
    n = len(x)
    for epoch in range(epochs):
        total_loss = 0
        for image in range(n):
            X = x[image].reshape(784,1)
            Y = y[image].reshape(10,1)

            (P, z2, o2, z1, o1) = forward(X, W1, b1, W2, b2, W3, b3)
            total_loss += cross_entropy(P, Y)
            (dW1, db1, dW2, db2, dW3, db3) = gradient(X, Y, P, z2, o2, W3, z1, o1, W2)
            
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2
            W3 = W3 - lr * dW3
            b3 = b3 - lr * db3
        print(f"epoch {epoch+1}, loss_moyenne = {total_loss/n:.4f}")
    return W1, b1, W2, b2, W3, b3

W1, b1, W2, b2, W3, b3 = train(X_train, Y_train, W1, b1, W2, b2, W3, b3)

def evaluate(x, y, W1, b1, W2, b2, W3, b3):
    errors = 0
    n = len(x)
    for image in range(n):
        X = x[image].reshape(784,1)
        Y = y[image].reshape(10,1)
        (P, z2, o2, z1, o1) = forward(X, W1, b1, W2, b2, W3, b3)
        y_pred = np.argmax(P)
        y_true = np.argmax(Y)
        if y_pred != y_true:
            errors += 1
    return (errors/n)*100

print(f"taux d'erreur (valeur train) : {evaluate(X_train, Y_train, W1, b1, W2, b2, W3, b3):.2f}%") #1,29% 
#moins bon qu'avec seulement une couche -> sûrement du au learning rate ou nombre d'epochs

print(f"taux d'erreur (valeur test) : {evaluate(X_test, Y_test, W1, b1, W2, b2, W3, b3):.2f}%") #2,61%