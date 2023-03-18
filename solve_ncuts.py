import numpy as np

def n_cuts(W, regularization):
    if regularization is True:
        # Regularization
        offset = 5e-1
        D = np.sum(np.abs(W), axis=1) + offset * 2
        W += np.diag(0.5 * (D - np.sum(W, axis=1)) + offset)
    else:
        # Without Regularization
        D = np.sum(np.abs(W), axis=1)

    # Three ways to calculate T
    # T = np.linalg.inv(D ** (1 / 2))
    T = np.power(D, -1 / 2)
    # T = 1. / np.sqrt(D)

    # Change D to a Diagonal one
    D = np.diag(D)

    # Calculate eigenvalues and eigenvectors
    # eigenvalues, eigenvectors = np.linalg.eig(T * (D - W) * T)
    (_, eigenvalues, eigenvectors) = np.linalg.svd(T * (D - W) * T)
    return eigenvalues, eigenvectors