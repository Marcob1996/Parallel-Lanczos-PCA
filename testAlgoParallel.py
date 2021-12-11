import numpy as np
import time
from keras.datasets import mnist
from LSVD_s import lanczosSVD
from LSVD_p import lanczosSVDp
from LSVD_pe import lanczosSVDpe
from sklearn.preprocessing import StandardScaler
import cupy as cp

if __name__ == '__main__':

    # Load MNIST
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_samples = train_X.shape[0]
    test_samples = test_X.shape[0]
    pixels = train_X.shape[1] * train_X.shape[2]
    X = train_X.reshape(train_samples, pixels)

    # Take smaller subset of examples to test
    num_vals = [10000, 30000, 50000]

    # Hyperparameters
    k = 150
    trunc = 3
    runs = 3

    for num in num_vals:

        print('Accuracy and Runtime for %d samples and k = %d' %(num, k))

        Data = X[0:num, :]
        labels = train_y[0:num].reshape(num, 1)
        m, n = Data.shape

        # Standardize data
        Data = StandardScaler().fit_transform(Data)

        times_s = np.zeros(runs)
        # Perform approximate SVD algo (serial)
        for j in range(runs):
            t1 = time.time()
            projX, U, D, Vt = lanczosSVD(Data, k, trunc)
            ts = time.time()-t1
            times_s[j] = ts

        times_p = np.zeros(runs)
        # Perform approximate SVD algo (parallel)
        for j in range(runs):
            t2 = time.time()
            projXpe, Upe, Dpe, Vtpe = lanczosSVDpe(Data, k, trunc)
            tpe = time.time() - t2
            times_p[j] = tpe

        if num != 50000:
            # Perform true SVD algo
            Ux, Sx, Vx = cp.linalg.svd(cp.array(Data))

            # Compare accuracy
            print('Error of Lanczos Serial SVD vs True SVD:')
            print(np.linalg.norm(abs(Vt) - abs(cp.asnumpy(Vx.T[:, 0:trunc]))))
            print('Error of Lanczos Parallel Efficient SVD vs True SVD:')
            print(cp.linalg.norm(abs(Vtpe) - abs(Vx.T[:, 0:trunc])))

        # Compare runtime
        print('Serial Runtime (Min):')
        print(np.minimum(times_s))
        print('Serial Runtime (Average):')
        print(np.average(times_s))
        print('Parallel Runtime (Min):')
        print(np.minimum(times_p))
        print('Parallel Runtime (Average):')
        print(np.average(times_p))

        print('==================================')
        print('==================================')
        print('==================================')
