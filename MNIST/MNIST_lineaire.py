#MODELE LINEAIRE

import numpy as np
from preprocessing import X_train, Y_train, X_test, Y_test

#Fonction softmax
def softmax(o):
    exp_o = np.exp(o - np.max(o))
    return exp_o / np.sum(exp_o)


#Initialisation des paramètres pour le forward
np.random.seed(42)
A = np.random.randn(10,784)*0.01 #*0.01 pour pas que les scores initiaux soient trop grande dès le début 
b = np.zeros((10,1))

def forward(x, A, b):
    o = A @ x + b
    P = softmax(o)
    return P

def cross_entropy(P, y):
    L = -np.sum(y*np.log(P))
    return L

def gradient(x, y, P):
    dL_do = P - y
    dA = dL_do @ x.T 
    db = dL_do
    return dA, db



def train(A,b,x,y, lr = 0.01, epochs=10):
    n = len(x)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0,n):
            X = x[i].reshape(784,1) #on prend un par un les 60000 lignes qui représentent un chiffre, et on le reshape sous forme de vecteur colonne
            Y = y[i].reshape(10,1)
            P = forward(X, A, b) #(10,1)
            total_loss += cross_entropy(P,Y)
            dA, dB = gradient(X, Y, P)
            A = A - lr*dA
            b = b - lr*dB
        print(f"Epoch {epoch+1}, Loss moyenne: {total_loss/n:.4f}")
    return A, b

A, b = train(A, b, X_train, Y_train)

def evaluate(x,y, A, b):
    errors = 0
    for image in range(len(x)):
        X = x[image].reshape(784,1)
        Y = y[image].reshape(10,1)
        P = forward(X, A, b)
        y_pred = np.argmax(P)
        y_true = np.argmax(Y)
        if y_pred != y_true:
            errors +=1
    return (errors*100)/len(x)

print(f"taux d'erreur (train) = {evaluate(X_train, Y_train, A, b)}%") #8,19%
print(f"taux d'erreur (test) = {evaluate(X_test, Y_test, A, b)}%") #8,77%


np.save('MNIST/params/A.npy', A)
np.save('MNIST/params/b.npy', b)