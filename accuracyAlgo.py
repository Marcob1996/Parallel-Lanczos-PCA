import numpy as np
import time
from keras.datasets import mnist
from LSVD_s import lanczosSVD
from LSVD_p import lanczosSVDp
from sklearn.preprocessing import StandardScaler
import cupy as cp

if __name__ == '__main__':

    # Load MNIST
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_samples = train_X.shape[0]
    test_samples = test_X.shape[0]
    pixels = train_X.shape[1] * train_X.shape[2]
    X = train_X.reshape(train_samples, pixels)
    X = np.concatenate((X, test_X.reshape(test_samples, pixels)))
    label = np.concatenate(train_y.reshape(train_samples, 1), test_y.reshape(test_samples, 1))
    trunc = 3

    # Compare values of k (to determine how varying k affects accuracy)
    k_vals = np.linspace(0, 150, 16).astype(int)
    k_vals[0] = 1

    times_s = np.zeros((k_vals.shape[0], 1))
    times_p = np.zeros((k_vals.shape[0], 1))
    errors_inf = np.zeros((k_vals.shape[0], 1))
    errors_2norm = np.zeros((k_vals.shape[0], 1))
    errors_inf_p = np.zeros((k_vals.shape[0], 1))
    errors_2norm_p = np.zeros((k_vals.shape[0], 1))
    dim = 3

    # Time the difference between computing true SVD and approximate SVD via Lanczos
    tsvd = time.time()
    Ux, Sx, Vx = np.linalg.svd(X)
    elapsed_tsvd = time.time() - tsvd

    # Compute projected data
    trueProjX = Ux[:, 0:dim] * Sx[0:dim]

    count = 0
    for k in k_vals:
        # Compute serial time
        t3dk = time.time()
        projX, Uk, Dk, Vtk = lanczosSVD(X, k, trunc)
        elapsed_t3dk = time.time() - t3dk
        # Compute parallel time
        t3dkp = time.time()
        projXp, Ukp, Dkp, Vtkp = lanczosSVDp(X, k, trunc)
        elapsed_t3dkp = time.time() - t3dkp

        print('Run for k = %d is Complete!' % k)
        # Store times
        times_s[count] = elapsed_t3dk
        times_p[count] = elapsed_t3dk

        # Compute projected data
        projXk = Uk[:, 0:dim] * Dk[0:dim]
        projXkp = Ukp[:, 0:dim] * Dkp[0:dim]

        # Compute and store error
        errors_inf[count] = np.linalg.norm(abs(projXk) - abs(trueProjX), np.inf)
        errors_2norm[count] = np.linalg.norm(abs(projXk) - abs(trueProjX))

        errors_inf_p[count] = np.linalg.norm(abs(projXkp) - abs(trueProjX), np.inf)
        errors_2norm_p[count] = np.linalg.norm(abs(projXkp) - abs(trueProjX))

        count += 1

