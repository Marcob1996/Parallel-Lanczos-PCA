import numpy as np


def lanczosSVD(A, k, trunc):
    m = A.shape[0]
    T, V = lanczos(A, k)
    U, D, Vt = approx_svd(T, V, m, trunc)
    projData = np.matmul(A, Vt)
    return projData, U, D, Vt


def lanczos(A, k):
    r, c = A.shape
    tot = r + c
    V = np.zeros((tot, k))
    alphas = np.zeros(k)
    betas = np.zeros(k)
    v = np.random.rand(tot)
    v = v / np.linalg.norm(v)
    b = 0
    v_previous = np.zeros(tot).T
    for i in range(k):
        V[:, i] = v
        w = np.concatenate((np.dot(A.T, v[-r:]), np.dot(A, v[0:c])))
        a = np.dot(v, w)
        alphas[i] = a
        w = w - b * v_previous - a * v
        # Re-orthogonalization
        w = reorthogonalization(V, w, i)
        b = np.linalg.norm(w)
        betas[i] = b
        if b < np.finfo(float).eps:
            break
        v_previous = v
        v = (1 / b) * w

    T = np.diag(alphas) + np.diag(betas[0:-1], k=1) + np.diag(betas[0:-1], k=-1)
    return T, V


def approx_svd(T, V, m, c):
    # Compute Eigenvalues and Eigenvectors of Tridiagonal Matrix from Lanczos
    Eig_val, Eig_vec = np.linalg.eigh(T)
    tempY = V @ Eig_vec
    r = tempY.shape[0]
    Y_l = tempY[-m:, -c:] / np.linalg.norm(tempY[-m:, -c:], axis=0, keepdims=True)
    Y_r = tempY[0:(r - m), -c:] / np.linalg.norm(tempY[0:(r - m), -c:], axis=0, keepdims=True)
    return np.fliplr(Y_l), Eig_val, np.fliplr(Y_r)


def reorthogonalization(V, w, i):
    for t in range(i):
        adj = cp.dot(V[:, t], w)
        if adj == 0.0:
            continue
        w -= adj * V[:, t]
    return w