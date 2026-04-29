import numpy as np
from preprocessing import X_test, X_train, Y_test, Y_train


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

p1 = 128 #nombre de neurones dans la couche cachée

np.random.seed(42)
W1 = np.random.randn(p1, 3072)*0.01
b1 = np.zeros([p1,1])
W2 = np.random.randn(10,p1)*0.01
b2 = np.zeros([10,1]) 

def forward(x, W1, b1, W2, b2):
    # x : (batch_size, 3072)
    O1 = x @ W1.T + b1.T        # (batch_size, p1)
    Z = ReLU(O1)                 # (batch_size, p1)
    O2 = Z @ W2.T + b2.T        # (batch_size, 10)
    P = np.array([softmax(o) for o in O2])  # (batch_size, 10)
    return P, Z, O1

def gradient(x, y, P, Z, O1, W2, batch_size):
    # x : (batch_size, 3072)
    dL_dO2 = (P - y) / batch_size      # (batch_size, 10)
    dW2 = dL_dO2.T @ Z                 # (10, p1)
    db2 = np.mean(dL_dO2, axis=0).reshape(-1, 1)

    dL_dZ = dL_dO2 @ W2                # (batch_size, p1)
    dL_dO1 = dL_dZ * relu_derivee(O1)  # (batch_size, p1)
    dW1 = dL_dO1.T @ x                 # (p1, 3072)
    db1 = np.mean(dL_dO1, axis=0).reshape(-1, 1)

    return dW1, db1, dW2, db2

def train(x, y, W1, b1, W2, b2, lr=0.01, epochs=10, batch_size=64):
    n = len(x)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, n, batch_size):
            X_batch = x[i:i+batch_size]
            Y_batch = y[i:i+batch_size]

            P, Z, O1 = forward(X_batch, W1, b1, W2, b2)

            P_clip = np.clip(P, 1e-10, 1.0)
            total_loss += np.mean(-np.sum(Y_batch * np.log(P_clip), axis=1))

            dW1, db1, dW2, db2 = gradient(X_batch, Y_batch, P, Z, O1, W2, batch_size)

            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2

        print(f"epoch {epoch+1}, loss = {total_loss/(n//batch_size):.4f}")
    return W1, b1, W2, b2

W1, b1, W2, b2 = train(X_train, Y_train, W1, b1, W2, b2)

def evaluate(x, y, W1, b1, W2, b2):
    n = len(x)
    errors = 0
    for image in range(n):
        X = x[image:image+1]  # (1, 3072) — batch de taille 1
        Y = y[image:image+1]  # (1, 10)
        P, _, _ = forward(X, W1, b1, W2, b2)
        y_pred = np.argmax(P)
        y_true = np.argmax(Y)
        if y_pred != y_true:
            errors += 1
    return (errors / n) * 100

print(f"taux d'erreur (train) = {evaluate(X_train, Y_train, W1, b1, W2, b2):.2f}%")
#55.03%
print(f"taux d'erreur (test) : {evaluate(X_test, Y_test, W1, b1, W2, b2):.2f}%")
#55.76%