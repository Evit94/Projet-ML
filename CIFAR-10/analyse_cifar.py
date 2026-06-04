#ANALYSE ET COMPARAISON DES MODELES CIFAR-10
#On réentraîne rapidement les 4 modèles NumPy (gris/couleur, linéaire/1 couche)
#sur un sous-ensemble, puis on les compare avec le CNN (résultats lus depuis le json).

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import X_train, X_test, X_gris_train, X_gris_test, Y_train, Y_test

#Sous-ensemble pour garder un temps de calcul raisonnable
N_TRAIN = 20000
N_TEST = 5000


#Fonctions communes (mêmes que dans les modèles)
def softmax(o):
    exp_o = np.exp(o - np.max(o))
    return exp_o / np.sum(exp_o)

def ReLU(x):
    return np.maximum(0, x)

def relu_derivee(x):
    return (x > 0).astype(float)


#---------- MODELE LINEAIRE (image par image) ----------
def train_lineaire(x, y, dim, lr, epochs):
    np.random.seed(42)
    A = np.random.randn(10, dim) * 0.01
    b = np.zeros((10, 1))
    n = len(x)
    for epoch in range(epochs):
        for i in range(n):
            X = x[i].reshape(dim, 1)
            Y = y[i].reshape(10, 1)
            P = softmax(A @ X + b)
            dL_do = P - Y
            A = A - lr * (dL_do @ X.T)
            b = b - lr * dL_do
    return A, b

def evaluate_lineaire(x, y, A, b, dim):
    errors = 0
    n = len(x)
    for i in range(n):
        X = x[i].reshape(dim, 1)
        P = softmax(A @ X + b)
        if np.argmax(P) != np.argmax(y[i]):
            errors += 1
    return (errors / n) * 100


#---------- MODELE 1 COUCHE CACHEE (mini-batch) ----------
def forward_1couche(x, W1, b1, W2, b2):
    O1 = x @ W1.T + b1.T
    Z = ReLU(O1)
    O2 = Z @ W2.T + b2.T
    P = np.array([softmax(o) for o in O2])
    return P, Z, O1

def train_1couche(x, y, dim, lr, epochs, batch_size=64):
    np.random.seed(42)
    p1 = 128
    W1 = np.random.randn(p1, dim) * 0.01
    b1 = np.zeros((p1, 1))
    W2 = np.random.randn(10, p1) * 0.01
    b2 = np.zeros((10, 1))
    n = len(x)
    for epoch in range(epochs):
        for i in range(0, n, batch_size):
            X_batch = x[i:i+batch_size]
            Y_batch = y[i:i+batch_size]
            bs = len(X_batch)
            P, Z, O1 = forward_1couche(X_batch, W1, b1, W2, b2)
            dL_dO2 = (P - Y_batch) / bs
            dW2 = dL_dO2.T @ Z
            db2 = np.mean(dL_dO2, axis=0).reshape(-1, 1)
            dL_dZ = dL_dO2 @ W2
            dL_dO1 = dL_dZ * relu_derivee(O1)
            dW1 = dL_dO1.T @ X_batch
            db1 = np.mean(dL_dO1, axis=0).reshape(-1, 1)
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2
    return W1, b1, W2, b2

def evaluate_1couche(x, y, W1, b1, W2, b2):
    n = len(x)
    errors = 0
    for i in range(n):
        P, _, _ = forward_1couche(x[i:i+1], W1, b1, W2, b2)
        if np.argmax(P) != np.argmax(y[i]):
            errors += 1
    return (errors / n) * 100


#---------- ENTRAINEMENT DES 4 MODELES ----------
Xc_tr, Xc_te = X_train[:N_TRAIN], X_test[:N_TEST]
Xg_tr, Xg_te = X_gris_train[:N_TRAIN], X_gris_test[:N_TEST]
Yt, Yv = Y_train[:N_TRAIN], Y_test[:N_TEST]

metrics = {}

print("Entraînement linéaire gris...")
A, b = train_lineaire(Xg_tr, Yt, 1024, lr=0.01, epochs=8)
et = evaluate_lineaire(Xg_tr, Yt, A, b, 1024)
ev = evaluate_lineaire(Xg_te, Yv, A, b, 1024)
metrics["lineaire_gris"] = {"acc_train": round(100-et, 2), "acc_test": round(100-ev, 2),
                            "err_train": round(et, 2), "err_test": round(ev, 2)}

print("Entraînement 1 couche gris...")
W1, b1, W2, b2 = train_1couche(Xg_tr, Yt, 1024, lr=0.05, epochs=20)
et = evaluate_1couche(Xg_tr, Yt, W1, b1, W2, b2)
ev = evaluate_1couche(Xg_te, Yv, W1, b1, W2, b2)
metrics["1couche_gris"] = {"acc_train": round(100-et, 2), "acc_test": round(100-ev, 2),
                           "err_train": round(et, 2), "err_test": round(ev, 2)}

print("Entraînement linéaire couleur...")
A, b = train_lineaire(Xc_tr, Yt, 3072, lr=0.001, epochs=8)
et = evaluate_lineaire(Xc_tr, Yt, A, b, 3072)
ev = evaluate_lineaire(Xc_te, Yv, A, b, 3072)
metrics["lineaire_couleur"] = {"acc_train": round(100-et, 2), "acc_test": round(100-ev, 2),
                               "err_train": round(et, 2), "err_test": round(ev, 2)}

print("Entraînement 1 couche couleur...")
W1, b1, W2, b2 = train_1couche(Xc_tr, Yt, 3072, lr=0.05, epochs=20)
et = evaluate_1couche(Xc_tr, Yt, W1, b1, W2, b2)
ev = evaluate_1couche(Xc_te, Yv, W1, b1, W2, b2)
metrics["1couche_couleur"] = {"acc_train": round(100-et, 2), "acc_test": round(100-ev, 2),
                              "err_train": round(et, 2), "err_test": round(ev, 2)}

#Le CNN a été entraîné séparément (CNN.py), on lit ses résultats si disponibles
if os.path.exists("results/cifar10/metrics_cnn.json"):
    with open("results/cifar10/metrics_cnn.json") as f:
        metrics["CNN"] = json.load(f)
    print("Métriques CNN chargées.")
else:
    print("metrics_cnn.json absent : lancez CNN.py pour inclure le CNN.")


#---------- SAUVEGARDE ----------
os.makedirs("results/cifar10", exist_ok=True)
with open("results/cifar10/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Métriques sauvegardées : results/cifar10/metrics.json")

#Tableau récapitulatif
print("\n=== Comparaison des modèles CIFAR-10 ===")
print(f"{'Modèle':<20}{'acc train':>12}{'acc test':>12}")
for nom, m in metrics.items():
    print(f"{nom:<20}{m['acc_train']:>11.2f}%{m['acc_test']:>11.2f}%")


#---------- GRAPHIQUE DE COMPARAISON ----------
noms = list(metrics.keys())
acc_train = [metrics[n]["acc_train"] for n in noms]
acc_test = [metrics[n]["acc_test"] for n in noms]

x = np.arange(len(noms))
largeur = 0.35
plt.figure(figsize=(11, 6))
plt.bar(x - largeur/2, acc_train, largeur, label="train")
plt.bar(x + largeur/2, acc_test, largeur, label="test")
plt.xticks(x, noms, rotation=30, ha="right")
plt.ylabel("Accuracy (%)")
plt.title("Comparaison des modèles CIFAR-10")
plt.legend()
plt.grid(True, axis="y", ls=":")
plt.tight_layout()
plt.savefig("results/cifar10/model_comparison.png")
plt.close()
print("Graphique sauvegardé : results/cifar10/model_comparison.png")
