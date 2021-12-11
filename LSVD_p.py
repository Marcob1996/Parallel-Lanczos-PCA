import cupy as cp


def lanczosSVDp(A, k, trunc, v):
    m = A.shape[0]
    A = cp.asarray(A)
    X = A
    if A.shape[0] != A.shape[1]:
        A = sym_dataP(A)
    else:
        if not cp.allclose(A, A.T, rtol=1e-05, atol=1e-08):
            A = sym_dataP(A)

    T, V = lanczosP(A, k, v)
    U, D, Vt = approx_svdP(T, V, m, trunc)
    projData = cp.matmul(X, Vt)
    return projData, U, D, Vt


def lanczosP(A, k, v):
    r = A.shape[0]
    V = cp.zeros((r, k))
    alphas = cp.zeros(k)
    betas = cp.zeros(k)
    #v = cp.random.rand(r)
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
    print(alphas)
    print(betas)
    T = cp.diag(alphas) + cp.diag(betas[0:-1], k=1) + cp.diag(betas[0:-1], k=-1)
    return T, V


def approx_svdP(T, V, m, c):

    E_val, Evec = cp.linalg.eigh(T)
    print(E_val)
    tempY = V @ Evec
    r = tempY.shape[0]
    count = 0
    leftY = cp.zeros((m, c))
    rightY = cp.zeros((r - m, c))

    for i in range(len(E_val)):
        if E_val[i] > 1e-12:
            leftY[:, count] = tempY[-m:, i]/cp.linalg.norm(tempY[-m:,i])
            rightY[:, count] = tempY[0:r - m, i]/cp.linalg.norm(tempY[0:r-m, i])
            count += 1
            if count == c:
                break

    #leftY = normalize(leftY.T, norm="l2").T
    #rightY = normalize(rightY.T, norm="l2").T
    return leftY, E_val, rightY


def sym_dataP(X):
    # Create symmetric matrix S
    r, c = X.shape
    S = cp.zeros((r+c, r+c))
    S[0:c, c:r+c] = X.T
    S[c:r+c, 0:c] = X
    return S
