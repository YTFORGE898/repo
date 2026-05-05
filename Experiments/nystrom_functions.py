import numpy as np
from scipy.linalg import cholesky, solve_triangular, svd

def random_nystrom(A, k, l, rng):
    n = A.shape[0]
    Omega = rng.standard_normal((n, l))
    Y = A @ Omega
    nu = np.sqrt(n) * np.spacing(np.linalg.norm(Y, 2))
    Y_nu = Y + nu * Omega
    C = cholesky(Omega.T @ Y_nu, lower=False)
    B = solve_triangular(C.T, Y_nu.T, lower=True).T
    U, sigma, V = svd(B, full_matrices=False)
    lam_all = np.maximum(0.0, sigma**2 - nu)
    U1 = U[:, :k]
    lam = lam_all[:k]
    Ahat = U1 @ np.diag(lam) @ U1.T
    return U1, lam, Ahat

