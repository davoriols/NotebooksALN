import numpy as np


def cholesky(A, tol=1.e-10):

    n, r = A.shape

    L = np.zeros((n, n))

    if (n != r):
        raise ValueError("La matriu no és quadrada!")

    for k in range(0, n):
        L[k, k] = A[k, k]

        if (abs(L[k, k]) < tol):
            raise ValueError("Element diagonal massa petit! Matriu (pròxima a) singular!")

        for r in range(0, k):
            L[k, k] = L[k, k] - L[k, r]**2

        L[k, k] = np.sqrt(L[k, k])


        for i in range(k + 1, n):
            L[i, k] = A[i, k]

            for r in range(k):
                L[i, k] = L[i, k] - L[i, r]*L[k, r]

            L[i, k] = L[i, k] / L[k, k]

    return L

