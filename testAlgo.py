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

    # Take smaller subset of examples to test
    num_vals = [5000]
    # Hyperparameters
    k = 20
    trunc = 3

    for num in num_vals:


        # MAYBE split data up onto multiple GPUs here
        Data = X[0:num, :]
        labels = train_y[0:num].reshape(num, 1)
        m, n = Data.shape

        # Standardize data
        Data = StandardScaler().fit_transform(Data)

        v = cp.random.rand(Data.shape[0] + Data.shape[1])

        # Perform approximate SVD algo (serial)
        t1 = time.time()
        projX, U, D, Vt = lanczosSVD(Data, k, trunc, cp.asnumpy(v))
        ts = time.time()-t1

        print('parallel:')
        # Perform approximate SVD algo (parallel)
        t2 = time.time()
        projXp, Up, Dp, Vtp = lanczosSVDp(Data, k, trunc, v)
        tp = time.time() - t2

        # Perform true SVD algo
        Ux, Sx, Vx = np.linalg.svd(Data)

        # Compare accuracy
        print('Error of approximate SVD vs True SVD:')
        print(np.linalg.norm(abs(cp.asnumpy(Vtp)) - abs(Vx.T[:, 0:trunc])))
        print(np.linalg.norm(abs(Vt) - abs(Vx.T[:, 0:trunc])))

        # Compare runtime
        print('Serial Runtime:')
        #print(ts)
        print('Parallel Runtime:')
        #print(tp)

