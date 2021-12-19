import numpy as np
import time
from keras.datasets import mnist
from LSVD_s import lanczosSVD
from LSVD_p import lanczosSVDp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cupy as cp

if __name__ == '__main__':

    # Load MNIST
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_samples = train_X.shape[0]
    test_samples = test_X.shape[0]
    pixels = train_X.shape[1] * train_X.shape[2]
    X = train_X.reshape(train_samples, pixels)
    X = np.concatenate((X, test_X.reshape(test_samples, pixels)))
    label = np.concatenate((train_y.reshape(train_samples, 1), test_y.reshape(test_samples, 1)))
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

    X = StandardScaler().fit_transform(X)

    # Time the difference between computing true SVD and approximate SVD via Lanczos
    pca_t = time.time()
    #Ux, Sx, Vx = np.linalg.svd(X)
    pca = PCA(n_components=dim, svd_solver='arpack')
    pca.fit(X)
    trueProjX = pca.transform(X)
    elapsed_pca = time.time() - pca_t

    # Flush out parallel overhead by running once
    projXp, Ukp, Dkp, Vtkp = lanczosSVDp(X, 1, trunc)

    count = 0
    for k in k_vals:
        # Compute serial time
        t3dk = time.time()
        projX, Uk, Dk, Vtk = lanczosSVD(X, k, trunc)
        elapsed_t3dk = time.time() - t3dk
        # Compute parallel time
        t3dkp = time.time()
        projXp, Ukp, Dkp, Vtkp = lanczosSVDp(X, k.item(), trunc)
        elapsed_t3dkp = time.time() - t3dkp

        print('Run for k = %d is Complete!' % k)
        # Store times
        times_s[count] = elapsed_t3dk
        times_p[count] = elapsed_t3dkp

        # Compute projected data
        projXk = Uk[:, 0:dim] * Dk[0:dim]
        projXkp = Ukp[:, 0:dim] * Dkp[0:dim]

        # Compute and store error
        errors_inf[count] = np.linalg.norm(abs(projXk) - abs(trueProjX), np.inf)
        errors_2norm[count] = np.linalg.norm(abs(projXk) - abs(trueProjX))

        errors_inf_p[count] = np.linalg.norm(abs(cp.asnumpy(projXkp)) - abs(trueProjX), np.inf)
        errors_2norm_p[count] = np.linalg.norm(abs(cp.asnumpy(projXkp)) - abs(trueProjX))

        count += 1

    fig1 = plt.figure()
    plt.plot(k_vals, errors_inf)
    plt.xlabel('K Values')
    plt.ylabel('Low Rank Approximation Error (Infinity Norm)')
    plt.yscale('log')
    plt.ylim([1e-13, 5000])
    fig1.savefig('inf_error_serial.png')

    fig2 = plt.figure()
    plt.plot(k_vals, errors_2norm)
    plt.xlabel('K Values')
    plt.ylabel('Low Rank Approximation Error (2-Norm)')
    plt.yscale('log')
    plt.ylim([1e-13, 5000])
    fig2.savefig('2norm_error_serial.png')

    fig3 = plt.figure()
    plt.plot(k_vals, times_s)
    plt.xlabel('K Values')
    plt.ylabel('Runtime (seconds)')
    fig3.savefig('runtime_serial.png')

    fig4 = plt.figure()
    plt.plot(k_vals, errors_inf_p)
    plt.xlabel('K Values')
    plt.ylabel('Low Rank Approximation Error (Infinity Norm)')
    plt.yscale('log')
    plt.ylim([1e-13, 5000])
    fig4.savefig('inf_error_parallel.png')

    fig5 = plt.figure()
    plt.plot(k_vals, errors_2norm_p)
    plt.xlabel('K Values')
    plt.ylabel('Low Rank Approximation Error (2-Norm)')
    plt.yscale('log')
    plt.ylim([1e-13, 5000])
    fig5.savefig('2norm_error_parallel.png')

    fig6 = plt.figure()
    plt.plot(k_vals, times_p)
    plt.xlabel('K Values')
    plt.ylabel('Runtime (seconds)')
    fig6.savefig('runtime_parallel.png')