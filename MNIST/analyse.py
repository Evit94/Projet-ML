import numpy as np
import matplotlib.pyplot as plt
from preprocessing import X_train, Y_train, X_test, Y_test

# Chargement des paramètres
A    = np.load('MNIST/params/A.npy')
b    = np.load('MNIST/params/b.npy')

W1_1 = np.load('MNIST/params/W1_1couche.npy')
b1_1 = np.load('MNIST/params/b1_1couche.npy')
W2_1 = np.load('MNIST/params/W2_1couche.npy')
b2_1 = np.load('MNIST/params/b2_1couche.npy')

W1_2 = np.load('MNIST/params/W1_2couches.npy')
b1_2 = np.load('MNIST/params/b1_2couches.npy')
W2_2 = np.load('MNIST/params/W2_2couches.npy')
b2_2 = np.load('MNIST/params/b2_2couches.npy')
W3_2 = np.load('MNIST/params/W3_2couches.npy')
b3_2 = np.load('MNIST/params/b3_2couches.npy')

print("Chargement réussi.")


# Fonctions communes
def softmax(o):
    exp_o = np.exp(o - np.max(o))
    return exp_o / np.sum(exp_o)

def ReLU(x):
    return np.maximum(0, x)

# Forward des trois modèles
def forward_lineaire(x, A, b):
    o = A @ x + b
    return softmax(o)

def forward_1couche(x, W1, b1, W2, b2):
    o1 = W1 @ x + b1
    z1 = ReLU(o1)
    o2 = W2 @ z1 + b2
    P = softmax(o2)
    return P, z1, o1

def forward_2couches(x, W1, b1, W2, b2, W3, b3):
    o1 = W1 @ x + b1
    z1 = ReLU(o1)
    o2 = W2 @ z1 + b2
    z2 = ReLU(o2)
    o3 = W3 @ z2 + b3
    P = softmax(o3)
    return P, z2, o2, z1, o1

def evaluate(x, y, forward_func, *params):
    errors = 0
    n = len(x)
    for image in range(n):
        X = x[image].reshape(784, 1)
        Y = y[image].reshape(10, 1)
        result = forward_func(X, *params)
        P = result[0] if isinstance(result, tuple) else result
        y_pred = np.argmax(P)
        y_true = np.argmax(Y)
        if y_pred != y_true:
            errors += 1
    return (errors / n) * 100

print("=== Comparaison des modèles ===")
print(f"Linéaire        - Test : {evaluate(X_test, Y_test, forward_lineaire, A, b):.2f}%")
print(f"1 couche cachée - Test : {evaluate(X_test, Y_test, forward_1couche, W1_1, b1_1, W2_1, b2_1):.2f}%")
print(f"2 couches       - Test : {evaluate(X_test, Y_test, forward_2couches, W1_2, b1_2, W2_2, b2_2, W3_2, b3_2):.2f}%")

def matrice_confusion(x, y, W1, b1, W2, b2):
    matrice = np.zeros((10, 10), dtype=int)
    n = len(x)
    for image in range(n):
        X = x[image].reshape(784, 1)
        Y = y[image].reshape(10, 1)
        P, z, o1 = forward_1couche(X, W1, b1, W2, b2)
        y_pred = np.argmax(P)
        y_true = np.argmax(Y)
        matrice[y_true, y_pred] += 1
    return matrice

mc = matrice_confusion(X_test, Y_test, W1_1, b1_1, W2_1, b2_1)

plt.figure(figsize=(8, 6))
plt.imshow(mc, cmap='Blues')
plt.colorbar()
plt.xlabel('Prédit')
plt.ylabel('Vrai')
plt.title('Matrice de confusion - 1 couche cachée')
for i in range(10):
    for j in range(10):
        plt.text(j, i, mc[i, j], ha='center', va='center', fontsize=8)
plt.show()


def afficher_erreurs(x, y, W1, b1, W2, b2, nb_erreurs=20):
    images_erreurs = []
    preds_erreurs = []
    vrais_erreurs = []

    n = len(x)
    for image in range(n):
        X = x[image].reshape(784, 1)
        Y = y[image].reshape(10, 1)
        P, z, o1 = forward_1couche(X, W1, b1, W2, b2)
        y_pred = np.argmax(P)
        y_true = np.argmax(Y)
        if y_pred != y_true:
            images_erreurs.append(x[image].reshape(28, 28))
            preds_erreurs.append(y_pred)
            vrais_erreurs.append(y_true)

        if len(images_erreurs) == nb_erreurs:
            break

    fig, axes = plt.subplots(4, 5, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images_erreurs[i], cmap='gray')
        ax.set_title(f"Vrai: {vrais_erreurs[i]} | Prédit: {preds_erreurs[i]}", fontsize=8)
        ax.axis('off')
    plt.suptitle("Images mal classées - 1 couche cachée")
    plt.show()

afficher_erreurs(X_test, Y_test, W1_1, b1_1, W2_1, b2_1)