import numpy as np

def linear(X, Z):
    K = X @ Z.T
    return K

def poly(X, Z):
    K = (X @ Z.T + 1)**2
    return K

def rbf(X, Z, gamma=0.1):
    X_norm = np.sum(X**2, axis=1)
    Z_norm = np.sum(Z**2, axis=1)
    dist = X_norm[:, None] + Z_norm[None, :] - 2*X @ Z.T
    dist = np.maximum(dist, 0)
    K = np.exp(-gamma*dist)
    return K

def make_kernel_matrix(data, kernel):
    n = data.shape[0]
    K = np.empty((n, n))

    for i in range(n):
        values = kernel(data[i:i+1], data[i:]).reshape(-1)
        K[i, i:] = values
        K[i:, i] = values

    return K