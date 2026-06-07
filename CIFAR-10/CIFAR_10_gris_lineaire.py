import numpy as np
from preprocessing import X_gris_train, X_gris_test, Y_test, Y_train
def softmax(o):
    exp_o = np.exp(o-np.max(o))
    return exp_o/np.sum(exp_o)



def cross_entropy(P, y):
    L = -np.sum(y*np.log(P))
    return L

np.random.seed(42)
A = np.random.randn(10,1024)*0.01
b = np.zeros([10,1])

def forward(x, A, b):
    o = A @ x + b
    P = softmax(o)
    return P

def gradient(P, x, y):
    dL_do = P - y
    dA = dL_do @ x.T
    db = dL_do
    return dA, db

def train(x, y, A, b, lr = 0.01, epochs = 10):
    n = len(x)
    for epoch in range(epochs):
        total_loss = 0
        for image in range(n):
            X = x[image].reshape(1024,1)
            Y = y[image].reshape(10,1)

            P = forward(X, A, b)

            total_loss += cross_entropy(P, Y)

            dA, dB = gradient(P, X, Y)

            A = A - lr * dA
            b = b - lr * dB
        print(f"epoch {epoch+1}, loss moyenne = {total_loss/n:.4f}")
    return A, b

A, b = train(X_gris_train, Y_train, A, b)

def evaluate(x, y, A, b):
    n = len(x)
    errors = 0
    for image in range(n):
        X = x[image].reshape(1024,1)
        Y = y[image].reshape(10,1)
        P = forward(X, A, b)
        y_pred = np.argmax(P)
        y_true = np.argmax(Y)
        if y_pred != y_true:
            errors += 1
    return (errors/n)*100

print(f"taux d'erreur (train) : {evaluate(X_gris_train, Y_train, A, b):.2f}%") #84.67%
print(f"taux d'erreur (test) : {evaluate(X_gris_test, Y_test, A, b):.2f}%") 
#85.7%


