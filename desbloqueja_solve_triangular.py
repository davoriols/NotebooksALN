import numpy as np

def factLU(A, tol=1.e-10):

    n, r = A.shape

    #comprovem que les dimensions siguin correctes
    if (n != r):
        raise ValueError(f"La matriu no és quadrada!")


    for k in range(0, n - 1):
        for i in range(k + 1, n):

            if (abs(A[k, k]) < tol):
                raise ValueError("Pivot massa petit! Matriu (pròxima a) singular!")

            m = A[i, k]/A[k, k]

            A[i, k] = m
            for j in range(k + 1, n): # Si comencem j = k fem els 0's a la columna k
                A[i, j] -= m * A[k, j]

    return A


def triL(L, b, ones=False, tol=1.e-10):

    n, m = L.shape

    #comprovem que les dimensions siguin correctes
    if (n != m):
        raise ValueError(f"La matriu és {n}x{m} i ha de ser quadrada!")
    elif(n != len(b)):
        raise ValueError(f"Dimensions incompatibles! (files matriu) {n} != {len(b)} (elements vector)")


    x = np.zeros((n))

    for i in range(n):
        x[i] = b[i]

        for j in range(i):
            x[i] -= L[i, j]*x[j]

        if (abs(L[i, i]) < tol):
            raise ValueError("Element diagonal massa petit!")

        if (not ones):
            x[i] /= L[i, i]

    return x

def triU(U, b, tol=1e-10):

    n, m = U.shape

    #comprovem que les dimensions siguin correctes
    if (n != m):
        raise ValueError(f"La matriu és {n}x{m} i ha de ser quadrada!")
    elif(n != len(b)):
        raise ValueError(f"Dimensions incompatibles! (files matriu) {n} != {len(b)} (elements vector)")



    x = np.zeros((n))

    for i in range(n - 1, -1, -1):
        x[i] = b[i]

        for j in range(i + 1, n):
            x[i] -= U[i][j]*x[j]

        if (abs(U[i][i]) < tol):
            raise ValueError("Element diagonal massa petit!")

        x[i] /= U[i][i]

    return x
