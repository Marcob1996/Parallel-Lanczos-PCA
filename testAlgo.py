import numpy as np
from keras.datasets import mnist
from LanczosSVD_Serial import lanczosSVD
from LanczosSVD_Parallel import lanczosSVDp
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':

    # Load MNIST
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_samples = train_X.shape[0]
    test_samples = test_X.shape[0]
    pixels = train_X.shape[1] * train_X.shape[2]
    X = train_X.reshape(train_samples, pixels)

    # Take smaller subset of examples to test
    num = 5000
    X = X[0:num, :]
    labels = train_y[0:num].reshape(num, 1)
    m, n = X.shape

    # Hyperparameters
    k = 60
    trunc = 3

    # Standardize data
    X = StandardScaler().fit_transform(X)

    # Perform approximate SVD algo (serial)
    projX, U, D, Vt = lanczosSVD(X, k, trunc)

    # Perform approximate SVD algo (serial)
    projXp, Up, Dp, Vtp = lanczosSVDp(X, k, trunc)

    # Perform true SVD algo
    Ux, Sx, Vx = np.linalg.svd(X)

    # Compare accuracy
    print('Error of approximate SVD vs True SVD:')
    print(np.linalg.norm(abs(Vt) - abs(Vx.T[:, 0:trunc])))
    print(np.linalg.norm(abs(Vtp) - abs(Vx.T[:, 0:trunc])))

