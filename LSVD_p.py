import cupy as cp


def lanczosSVDp(A, k, trunc):
    m = A.shape[0]
    A = cp.asarray(A)
    X = A
    if A.shape[0] != A.shape[1]:
        A = sym_dataP(A)
    else:
        if not cp.allclose(A, A.T, rtol=1e-05, atol=1e-08):
            A = sym_dataP(A)

    T, V = lanczosP(A, k)
    U, D, Vt = approx_svdP(T, V, m, trunc)
    projData = cp.matmul(X, Vt)
    return projData, U, D, Vt


def lanczosP(A, k):
    r = A.shape[0]
    V = cp.zeros((r, k))
    alphas = cp.zeros(k)
    betas = cp.zeros(k)
    v = cp.random.rand(r)
    v = v / cp.linalg.norm(v)
    b = 0
    v_previous = cp.zeros(r).T
    for i in range(k):
        V[:, i] = v
        w = cp.dot(A, v)
        a = cp.dot(v, w)
        alphas[i] = a
        w = w - b * v_previous - a * v

        # Re-orthogonalization
        for t in range(i):
            adj = cp.dot(V[:, t], w)
            if adj == 0.0:
                continue
            w -= adj * V[:, t]

        b = cp.linalg.norm(w)
        betas[i] = b
        if b < cp.finfo(float).eps:
            break
        v_previous = v
        v = (1 / b) * w
    T = cp.diag(alphas) + cp.diag(betas[0:-1], k=1) + cp.diag(betas[0:-1], k=-1)
    return T, V


def approx_svdP(T, V, m, c):
    E_val, Evec = cp.linalg.eigh(T)
    tempY = V@Evec
    r = tempY.shape[0]
    leftY = tempY[-m:, -c:]/cp.linalg.norm(tempY[-m:, -c:], axis=0, keepdims=True)
    rightY = tempY[0:r-m, -c:]/cp.linalg.norm(tempY[0:r-m, -c:], axis=0, keepdims=True)
    return cp.fliplr(leftY), E_val, cp.fliplr(rightY)


def sym_dataP(X):
    # Create symmetric matrix S
    r, c = X.shape
    S = cp.zeros((r+c, r+c))
    S[0:c, c:r+c] = X.T
    S[c:r+c, 0:c] = X
    return S
