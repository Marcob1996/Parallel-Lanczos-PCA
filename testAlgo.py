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

    # Take smaller subset of examples to test
    num_vals = [10000, 30000, 60000]

    # Hyperparameters
    k = 100
    trunc = 3
    runs = 4

    for num in num_vals:

        print('Accuracy and Runtime for %d samples and k = %d' %(num, k))

        Data = X[0:num, :]
        labels = train_y[0:num].reshape(num, 1)
        m, n = Data.shape

        # Standardize data
        Data = StandardScaler().fit_transform(Data)

        times_s = cp.zeros(runs)
        # Perform approximate SVD algo (serial)
        for j in range(runs):
            t1 = time.time()
            projX, U, D, Vt = lanczosSVD(Data, k, trunc)
            ts = time.time()-t1
            times_s[j] = ts

        times_p = cp.zeros(runs)
        # Perform approximate SVD algo (parallel)
        for j in range(runs):
            t2 = time.time()
            projXp, Up, Dp, Vtp = lanczosSVDp(Data, k, trunc)
            tp = time.time() - t2
            times_p[j] = tp

        if num != 60000:
            # Perform true SVD algo
            Ux, Sx, Vx = cp.linalg.svd(cp.array(Data))
            trueProjX = Ux[:, 0:trunc]*Sx[0:trunc]

            # Compare accuracy
            print('PCA Error Between Lanczos Serial SVD and True SVD:')
            print(np.linalg.norm(abs(projX) - abs(cp.asnumpy(trueProjX))))
            print('PCA Error Between Lanczos Parallel SVD and True SVD:')
            print(cp.linalg.norm(abs(projXp) - abs(trueProjX)))

        # Compare runtime
        print('Serial Runtime (Minimum):')
        print(cp.amin(times_s))
        print('Serial Runtime (Average):')
        print(cp.average(times_s))
        print('All Serial Runtimes:')
        print(times_s)
        print('Parallel Runtime (Minimum):')
        print(cp.amin(times_p))
        print('Parallel Runtime (Average):')
        print(cp.average(times_p))
        print('All Parallel Runtimes:')
        print(times_p)

        print('==================================')
        print('==================================')
        print('==================================')
