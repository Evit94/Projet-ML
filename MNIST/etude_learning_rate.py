#ETUDE DE L'INFLUENCE DU LEARNING RATE
#On teste plusieurs learning rates sur les trois modèles (linéaire, 1 couche, 2 couches)
#et on regarde l'impact sur l'erreur et l'accuracy (train et test).

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import X_train, Y_train, X_test, Y_test

#Pour que l'étude reste rapide, on entraîne sur un sous-ensemble des images
#(7 learning rates x 3 modèles, sinon c'est très long sur les 60000 images)
N_TRAIN = 10000
N_TEST = 5000
X_train = X_train[:N_TRAIN]
Y_train = Y_train[:N_TRAIN]
X_test = X_test[:N_TEST]
Y_test = Y_test[:N_TEST]

#Liste des learning rates à tester
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
epochs = 5


#Fonctions communes (mêmes que dans les modèles)
def softmax(o):
    exp_o = np.exp(o - np.max(o))
    return exp_o / np.sum(exp_o)

def cross_entropy(P, y):
    L = -np.sum(y * np.log(P))
    return L

def ReLU(x):
    return np.maximum(0, x)

def relu_derivee(x):
    return (x > 0).astype(float)


#---------- MODELE LINEAIRE ----------
def forward_lineaire(x, A, b):
    o = A @ x + b
    return softmax(o)

def train_lineaire(x, y, lr, epochs):
    np.random.seed(42)
    A = np.random.randn(10, 784) * 0.01
    b = np.zeros((10, 1))
    n = len(x)
    for epoch in range(epochs):
        for i in range(n):
            X = x[i].reshape(784, 1)
            Y = y[i].reshape(10, 1)
            P = forward_lineaire(X, A, b)
            dL_do = P - Y
            A = A - lr * (dL_do @ X.T)
            b = b - lr * dL_do
    return A, b

def evaluate_lineaire(x, y, A, b):
    errors = 0
    n = len(x)
    for i in range(n):
        X = x[i].reshape(784, 1)
        P = forward_lineaire(X, A, b)
        if np.argmax(P) != np.argmax(y[i]):
            errors += 1
    return (errors / n) * 100


#---------- MODELE 1 COUCHE CACHEE ----------
def forward_1couche(x, W1, b1, W2, b2):
    o1 = W1 @ x + b1
    z = ReLU(o1)
    o2 = W2 @ z + b2
    P = softmax(o2)
    return P, z, o1

def train_1couche(x, y, lr, epochs):
    np.random.seed(42)
    p1 = 128
    W1 = np.random.randn(p1, 784) * 0.01
    b1 = np.zeros((p1, 1))
    W2 = np.random.randn(10, p1) * 0.01
    b2 = np.zeros((10, 1))
    n = len(x)
    for epoch in range(epochs):
        for i in range(n):
            X = x[i].reshape(784, 1)
            Y = y[i].reshape(10, 1)
            P, z, o1 = forward_1couche(X, W1, b1, W2, b2)
            dL_do2 = P - Y
            dW2 = dL_do2 @ z.T
            db2 = dL_do2
            dL_dz = W2.T @ dL_do2
            dL_do1 = dL_dz * relu_derivee(o1)
            dW1 = dL_do1 @ X.T
            db1 = dL_do1
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2
    return W1, b1, W2, b2

def evaluate_1couche(x, y, W1, b1, W2, b2):
    errors = 0
    n = len(x)
    for i in range(n):
        X = x[i].reshape(784, 1)
        P, z, o1 = forward_1couche(X, W1, b1, W2, b2)
        if np.argmax(P) != np.argmax(y[i]):
            errors += 1
    return (errors / n) * 100


#---------- MODELE 2 COUCHES CACHEES ----------
def forward_2couches(x, W1, b1, W2, b2, W3, b3):
    o1 = W1 @ x + b1
    z1 = ReLU(o1)
    o2 = W2 @ z1 + b2
    z2 = ReLU(o2)
    o3 = W3 @ z2 + b3
    P = softmax(o3)
    return P, z2, o2, z1, o1

def train_2couches(x, y, lr, epochs):
    np.random.seed(42)
    p1 = 128
    p2 = 64
    W1 = np.random.randn(p1, 784) * 0.01
    b1 = np.zeros((p1, 1))
    W2 = np.random.randn(p2, p1) * 0.01
    b2 = np.zeros((p2, 1))
    W3 = np.random.randn(10, p2) * 0.01
    b3 = np.zeros((10, 1))
    n = len(x)
    for epoch in range(epochs):
        for i in range(n):
            X = x[i].reshape(784, 1)
            Y = y[i].reshape(10, 1)
            P, z2, o2, z1, o1 = forward_2couches(X, W1, b1, W2, b2, W3, b3)
            dL_do3 = P - Y
            dW3 = dL_do3 @ z2.T
            db3 = dL_do3
            dL_dz2 = W3.T @ dL_do3
            dL_do2 = dL_dz2 * relu_derivee(o2)
            dW2 = dL_do2 @ z1.T
            db2 = dL_do2
            dL_dz1 = W2.T @ dL_do2
            dL_do1 = dL_dz1 * relu_derivee(o1)
            dW1 = dL_do1 @ X.T
            db1 = dL_do1
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2
            W3 = W3 - lr * dW3
            b3 = b3 - lr * db3
    return W1, b1, W2, b2, W3, b3

def evaluate_2couches(x, y, W1, b1, W2, b2, W3, b3):
    errors = 0
    n = len(x)
    for i in range(n):
        X = x[i].reshape(784, 1)
        P, z2, o2, z1, o1 = forward_2couches(X, W1, b1, W2, b2, W3, b3)
        if np.argmax(P) != np.argmax(y[i]):
            errors += 1
    return (errors / n) * 100


#---------- BOUCLE PRINCIPALE ----------
#On stocke les résultats dans des dictionnaires : une liste par modèle
resultats = {
    "lineaire":  {"err_train": [], "err_test": [], "acc_train": [], "acc_test": []},
    "1couche":   {"err_train": [], "err_test": [], "acc_train": [], "acc_test": []},
    "2couches":  {"err_train": [], "err_test": [], "acc_train": [], "acc_test": []},
}

for lr in learning_rates:
    print(f"\n===== Learning rate = {lr} =====")

    #Modèle linéaire
    A, b = train_lineaire(X_train, Y_train, lr, epochs)
    et = evaluate_lineaire(X_train, Y_train, A, b)
    ev = evaluate_lineaire(X_test, Y_test, A, b)
    resultats["lineaire"]["err_train"].append(et)
    resultats["lineaire"]["err_test"].append(ev)
    resultats["lineaire"]["acc_train"].append(100 - et)
    resultats["lineaire"]["acc_test"].append(100 - ev)
    print(f"Lineaire   : erreur train = {et:.2f}%, erreur test = {ev:.2f}%")

    #Modèle 1 couche cachée
    W1, b1, W2, b2 = train_1couche(X_train, Y_train, lr, epochs)
    et = evaluate_1couche(X_train, Y_train, W1, b1, W2, b2)
    ev = evaluate_1couche(X_test, Y_test, W1, b1, W2, b2)
    resultats["1couche"]["err_train"].append(et)
    resultats["1couche"]["err_test"].append(ev)
    resultats["1couche"]["acc_train"].append(100 - et)
    resultats["1couche"]["acc_test"].append(100 - ev)
    print(f"1 couche   : erreur train = {et:.2f}%, erreur test = {ev:.2f}%")

    #Modèle 2 couches cachées
    W1, b1, W2, b2, W3, b3 = train_2couches(X_train, Y_train, lr, epochs)
    et = evaluate_2couches(X_train, Y_train, W1, b1, W2, b2, W3, b3)
    ev = evaluate_2couches(X_test, Y_test, W1, b1, W2, b2, W3, b3)
    resultats["2couches"]["err_train"].append(et)
    resultats["2couches"]["err_test"].append(ev)
    resultats["2couches"]["acc_train"].append(100 - et)
    resultats["2couches"]["acc_test"].append(100 - ev)
    print(f"2 couches  : erreur train = {et:.2f}%, erreur test = {ev:.2f}%")


#---------- SAUVEGARDE DES RESULTATS ----------
os.makedirs("results/mnist", exist_ok=True)

#CSV
with open("results/mnist/learning_rate_study.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["modele", "learning_rate", "err_train", "err_test", "acc_train", "acc_test"])
    for modele in resultats:
        for i, lr in enumerate(learning_rates):
            writer.writerow([
                modele, lr,
                f"{resultats[modele]['err_train'][i]:.2f}",
                f"{resultats[modele]['err_test'][i]:.2f}",
                f"{resultats[modele]['acc_train'][i]:.2f}",
                f"{resultats[modele]['acc_test'][i]:.2f}",
            ])
print("\nCSV sauvegardé : results/mnist/learning_rate_study.csv")


#---------- GRAPHIQUES ----------
noms = {"lineaire": "Linéaire", "1couche": "1 couche", "2couches": "2 couches"}

#1) Erreur (train et test) en fonction du learning rate
plt.figure(figsize=(10, 6))
for modele in resultats:
    plt.plot(learning_rates, resultats[modele]["err_train"], "o--", label=f"{noms[modele]} (train)")
    plt.plot(learning_rates, resultats[modele]["err_test"], "o-", label=f"{noms[modele]} (test)")
plt.xscale("log")
plt.xlabel("Learning rate (échelle log)")
plt.ylabel("Taux d'erreur (%)")
plt.title("Erreur en fonction du learning rate")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("results/mnist/error_vs_learning_rate.png")
plt.close()

#2) Accuracy (train et test) en fonction du learning rate
plt.figure(figsize=(10, 6))
for modele in resultats:
    plt.plot(learning_rates, resultats[modele]["acc_train"], "o--", label=f"{noms[modele]} (train)")
    plt.plot(learning_rates, resultats[modele]["acc_test"], "o-", label=f"{noms[modele]} (test)")
plt.xscale("log")
plt.xlabel("Learning rate (échelle log)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy en fonction du learning rate")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("results/mnist/accuracy_vs_learning_rate.png")
plt.close()

#3) Comparaison des trois modèles (erreur test)
plt.figure(figsize=(10, 6))
for modele in resultats:
    plt.plot(learning_rates, resultats[modele]["err_test"], "o-", label=noms[modele])
plt.xscale("log")
plt.xlabel("Learning rate (échelle log)")
plt.ylabel("Taux d'erreur test (%)")
plt.title("Comparaison des trois modèles - erreur test")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig("results/mnist/learning_rate_comparison.png")
plt.close()

print("Graphiques sauvegardés dans results/mnist/")
