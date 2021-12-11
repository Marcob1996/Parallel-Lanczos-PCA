import numpy as np


def lanczosSVD(A, k, trunc):
    m = A.shape[0]
    X = A
    if A.shape[0] != A.shape[1]:
        A = sym_data(A)
    else:
        if not np.allclose(A, A.T, rtol=1e-05, atol=1e-08):
            A = sym_data(A)
    T, V = lanczos(A, k)
    U, D, Vt = approx_svd(T, V, m, trunc)
    projData = np.matmul(X, Vt)
    return projData, U, D, Vt


def lanczos(A, k):
    r = A.shape[0]
    V = np.zeros((r, k))
    alphas = np.zeros(k)
    betas = np.zeros(k)
    v = np.random.rand(r)
    v = v / np.linalg.norm(v)
    b = 0
    v_previous = np.zeros(r).T
    for i in range(k):
        V[:, i] = v
        w = A.dot(v)
        a = np.dot(v, w)
        alphas[i] = a
        w = w - b * v_previous - a * v

        # Re-orthogonalization
        for t in range(i):
            adj = np.dot(V[:, t], w)
            if adj == 0.0:
                continue
            w -= adj * V[:, t]

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
    Y_r = tempY[0:r - m, -c:] / np.linalg.norm(tempY[0:r - m, -c:], axis=0, keepdims=True)
    print(leftY[0:10, :])
    return np.fliplr(Y_l), Eig_val, np.fliplr(Y_r)


def sym_data(X):
    # Create symmetric matrix S
    r, c = X.shape
    S = np.zeros((r+c, r+c))
    S[0:c, c:r+c] = X.T
    S[c:r+c, 0:c] = X
    return S
