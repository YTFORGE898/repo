import numpy as np

def linear(X,Z):
    K = X @ Z.T
    return K

def poly(X,Z):
    K = (X @ Z.T + 1.0)**2
    return K

def rbf(X,Z):
    gamma = 0.1
    X_norm = np.sum(X**2,axis=1)
    Z_norm = np.sum(Z**2,axis=1)
    dist = X_norm[:,None] + Z_norm[None,:] - 2*X@Z.T
    K = np.exp(-gamma*dist)
    return K