#MODELE AVEC UNE COUCHE CACHEE

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
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

#softmax et cross_entropy ne vont pas changer
def softmax(o):
    exp_o = np.exp(o - np.max(o)) #plante si o contient des très grands nombres (on soustrait le max de o)
    return exp_o / np.sum(exp_o)

def cross_entropy(P, y):
    L = -np.sum(y*np.log(P))
    return L


#Fonction d'activation choisie
def ReLU(x):
    return np.maximum(0,x)



#Nombre de neurones dans la couche cachée 
p1 = 128
np.random.seed(42)
W1 = np.random.randn(p1,784)*0.01   #(p1, 784) car il y a 784 entrées et p1 neurones
b1 = np.zeros((p1,1))
W2 = np.random.randn(10, p1)*0.01  #relie la couche cachée à la sortie, 10 classes (0...9) et p1 neurones 
b2 = np.zeros((10, 1))

def forward(x, W1, b1, W2, b2):
    o1 = W1 @ x + b1
    z = ReLU(o1)
    o2 = W2 @ z + b2
    P = softmax(o2)
    return P, z, o1 #on retourne o1, z et P pour la rétropropagation (descente de gradient)

#dérivée de ReLU
def relu_derivee(x):
    return (x>0).astype(float)

def gradient(x, y, P, z, o1, W2):
    dL_do2 = P - y
    dW2 = dL_do2 @ z.T #règle de la chaine
    db2 = dL_do2 #pareil

    dL_dz = W2.T @ dL_do2
    dL_do1 = dL_dz * relu_derivee(o1)
    dW1 = dL_do1 @ x.T
    db1 = dL_do1

    return dW1, db1, dW2, db2

#même principe que pour la couche linéaire
def train(x, y, W1, b1, W2, b2, lr = 0.01, epochs = 10):
    n = len(x)
    for epoch in range(epochs):
        total_loss = 0
        for image in range(n):
            X = x[image].reshape(784,1)
            Y = y[image].reshape(10,1)
            (P, z, o1) = forward(X, W1, b1, W2, b2)
            total_loss += cross_entropy(P, Y)
            (dW1, db1, dW2, db2) = gradient(X, Y, P, z, o1, W2)
            W1 = W1 - lr*dW1
            b1 = b1 - lr*db1
            W2 = W2 - lr*dW2
            b2 = b2 - lr*db2
        print(f"epoch {epoch+1} : loss_moyenne = {total_loss/n:.4f}")
    return W1, b1, W2, b2


W1, b1, W2, b2 = train(X_train, Y_train, W1, b1, W2, b2) #peut changer en [:5000] si trop long

def evaluate(x, y, W1, b1, W2, b2):
    errors = 0
    n = len(x)
    for image in range(n):
        X = x[image].reshape(784,1)
        Y = y[image].reshape(10,1)
        P, z, o1 = forward(X, W1, b1, W2, b2)
        y_pred = np.argmax(P)
        y_true = np.argmax(Y)
        if y_pred != y_true :
            errors += 1
    return (errors/n)*100

print(f"taux d'erreur (valeur train) : {evaluate(X_train, Y_train, W1, b1, W2, b2):.2f}%") #peut changer en [:5000] si trop long
#0,66%
print(f"taux d'erreur (valeur test): {evaluate(X_test, Y_test, W1, b1, W2, b2):.2f}%") #2,43%

#matrice de confusion
def matrice_confusion(x, y, W1, b1, W2, b2):
    matrice = np.zeros((10,10), dtype=int)
    n = len(x)
    for image in range(n):
        X = x[image].reshape(784,1)
        Y = y[image].reshape(10,1)
        P, z, o1 = forward(X, W1, b1, W2, b2)
        y_pred = np.argmax(P)
        y_true = np.argmax(Y)
        matrice[y_true, y_pred] += 1
    return matrice

mc = matrice_confusion(X_test, Y_test, W1, b1, W2, b2)

plt.figure(figsize=(8,6))
plt.imshow(mc, cmap='Blues')
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.title('Matrice de confusion')
for i in range(10):
    for j in range(10):
        plt.text(j, i, mc[i,j], ha='center', va='center', fontsize=8)
plt.show()

def afficher_erreurs(x, y, W1, b1, W2, b2, nb_erreurs=20):
    images_erreurs = []
    preds_erreurs = []
    vrais_erreurs = []

    n = len(x)
    for image in range(n):
        X = x[image].reshape(784,1)
        Y = y[image].reshape(10,1)
        P, z, o1 = forward(X, W1, b1, W2, b2)
        y_pred = np.argmax(P)
        y_true = np.argmax(Y)
        if y_pred != y_true:
            images_erreurs.append(x[image].reshape(28, 28))  # image en 2D pour affichage
            preds_erreurs.append(y_pred)
            vrais_erreurs.append(y_true)
        
        if len(images_erreurs) == nb_erreurs:  # on s'arrête après nb_erreurs
            break
    # Affichage
    fig, axes = plt.subplots(4, 5, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images_erreurs[i], cmap='gray')
        ax.set_title(f"Vrai: {vrais_erreurs[i]} | Prédit: {preds_erreurs[i]}", fontsize=8)
        ax.axis('off')
    plt.suptitle("Images mal classées")
    plt.show()

afficher_erreurs(X_test, Y_test, W1, b1, W2, b2)