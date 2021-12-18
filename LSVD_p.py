import cupy as cp


def lanczosSVDp(A, k, trunc):
    m = A.shape[0]
    A = cp.asarray(A)
    T, V = lanczosP(A, k)
    U, D, Vt = approx_svdP(T, V, m, trunc)
    projData = U[:, 0:trunc]*D[0:trunc]
    return projData, U, D, Vt


def lanczosP(A, k):
    r,c = A.shape
    tot = r+c
    V = cp.zeros((tot, k))
    alphas = cp.zeros(k)
    betas = cp.zeros(k)
    v = cp.random.rand(tot)
    v = v / cp.linalg.norm(v)
    b = 0
    v_previous = cp.zeros(tot).T
    for i in range(k):
        V[:, i] = v
        w = cp.concatenate((cp.dot(A.T, v[-r:]), cp.dot(A, v[0:c])))
        a = cp.dot(v, w)
        alphas[i] = a
        w = w - b * v_previous - a * v

        # Re-orthogonalization
        w = reorthogonalization(V, w, i)

        b = cp.linalg.norm(w)
        betas[i] = b
        if b < cp.finfo(float).eps:
            break
        v_previous = v
        v = (1 / b) * w
    T = cp.diag(alphas) + cp.diag(betas[0:-1], k=1) + cp.diag(betas[0:-1], k=-1)
    return T, V


def reorthogonalization(V, w, i):
    for t in range(i):
        adj = cp.dot(V[:, t], w)
        if adj == 0.0:
            continue
        w -= adj * V[:, t]
    return w


def approx_svdP(T, V, m, c):
    # Compute Eigenvalues and Eigenvectors of Tridiagonal Matrix from Lanczos
    E_val, Evec = cp.linalg.eigh(T)
    tempY = cp.matmul(V, Evec)
    r = tempY.shape[0]
    leftY = tempY[-m:, -c:] / cp.linalg.norm(tempY[-m:, -c:], axis=0, keepdims=True)
    rightY = tempY[0:(r-m), -c:] / cp.linalg.norm(tempY[0:(r-m), -c:], axis=0, keepdims=True)
    return cp.fliplr(leftY), E_val, cp.fliplr(rightY)
