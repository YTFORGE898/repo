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

def uniform_sampling(n, l, rng):
    # least informed
    sample_ind = rng.choice(n, size=l, replace=False)
    return sample_ind

def nystrom_CWCT(K, sample_ind):
    # C W^dagger C^T
    C = K[:, sample_ind]
    W = K[np.ix_(sample_ind, sample_ind)]
    K_tilde = C @ np.linalg.pinv(W) @ C.T
    return K_tilde

def sample_from_scores(scores, l, rng):
    # turn the scores into probabilities then sample
    probs = scores / scores.sum()
    sample_ind = rng.choice(len(scores), size=l, replace=False, p=probs)
    return sample_ind

def full_leverage(K, r, l, rng):
    # therotical max
    val, vec = np.linalg.eigh(K)
    U1 = vec[:, -r:]
    n = U1.shape[0]
    scores = np.zeros(n)
    
    for i in range(n):
        row = U1[i, :]
        scores[i] = np.sum(row ** 2)

    sample_ind = sample_from_scores(scores, l, rng)
    return sample_ind, scores

def approximate_leverage(K, ini_l, extra_l, rng):
    # assumed formation of all K need to optimise later
    n = K.shape[0]
    ini_ind = uniform_sampling(n, ini_l, rng)
    K1 = K[:, ini_ind]
    U, sigma, VT = np.linalg.svd(K1, full_matrices=False)
    n = U.shape[0]
    scores = np.zeros(n)
    
    for i in range(n):
        row = U[i, :]
        scores[i] = np.sum(row ** 2)

    scores[ini_ind] = 0
    extra_ind = sample_from_scores(scores, extra_l, rng)
    sample_ind = np.concatenate([ini_ind, extra_ind])
    return sample_ind, scores, ini_ind

def recursive_nystrom(X, kernel, lam, delta, rng, base_size=None):
    m = X.shape[0]

    if base_size is None:
        threshold = 192 * np.log(1 / delta)
    else:
        threshold = base_size

    #base case: if m below threshold, return all indices with weight 1 and score 1
    if m <= threshold:
        sample_ind = np.arange(m)
        weight = np.ones(m)
        score = np.ones(m)
        return sample_ind, weight

    half_ind = np.where(rng.random(m) <= 0.5)[0]
    #if no points are selected select the first point
    if len(half_ind) == 0:
        half_ind = np.array([0])

    small_ind, small_weight = recursive_nystrom(
        X[half_ind],
        kernel,
        lam,
        delta / 3,
        rng,
        base_size
    )

    inner_ind = half_ind[small_ind]
    X_inner = X[inner_ind]

    C = kernel(X, X_inner)
    C = C * small_weight[None, :]
    W = kernel(X_inner, X_inner)
    W = small_weight[:, None] * W * small_weight[None, :]
    A = W + lam * np.eye(len(inner_ind))
    Z = np.linalg.solve(A, C.T)
    diag_part = np.sum(C * Z.T, axis=1)
    K_diag = np.empty(m)

    for i in range(m):
        K_diag[i] = kernel(X[i:i+1], X[i:i+1])[0, 0]

    score = (3 / (2 * lam)) * (K_diag - diag_part)
    score = np.maximum(score, 0)
    score_sum = score.sum()

    if score_sum == 0:
        raise ValueError("all score are zero")

    p = np.minimum(1, score * 16 * np.log(score_sum / delta))
    random_values = rng.random(m)
    sample_ind = np.where(random_values < p)[0]
    #if no points are selected select the point with the highest p
    if len(sample_ind) == 0:
        sample_ind = np.array([np.argmax(p)])

    weight = 1 / np.sqrt(p[sample_ind])

    return sample_ind, weight