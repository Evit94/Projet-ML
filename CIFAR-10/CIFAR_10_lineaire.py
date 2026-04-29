import numpy as np
from preprocessing import X_gris_train, X_gris_test, Y_test, Y_train
def softmax(o):
    exp_o = np.exp(o-np.max(o))
    return exp_o/np.sum(exp_o)



def cross_entropy(P, y):
    return -np.sum(y*np.log(P))

np.random.seed(42)
A = np.random.randn(10,1024)*0.01
b = np.zeros([10,1])

def forward(x, A, b):
    o = A @ x + b
    P = softmax(o)
    return P

def gradient():
    