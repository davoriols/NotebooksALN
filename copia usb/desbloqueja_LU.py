import numpy as np

def LUpivpar(A, tol=1.e-10):

    n, r = A.shape

    #comprovem que les dimensions siguin correctes
    if (n != r):
        raise ValueError(f"La matriu no és quadrada!")


    #inicialitzem vector de permutacions
    p = np.arange(n)

    for k in range(0, n - 1):

        #seleccionem el pivot
        indexPivot = k + np.argmax(abs(A[k:, k]))

        A[[indexPivot, k]] = A[[k, indexPivot]]

        #actualitzem vector permutacions
        p[indexPivot], p[k] = p[k], p[indexPivot]

        # eliminació Gauss normal
        for i in range(k + 1, n):

            if (abs(A[k, k]) < tol):
                raise ValueError("Pivot massa petit! Matriu (pròxima a) singular!")

            m = A[i, k]/A[k, k]

            A[i, k] = m
            for j in range(k + 1, n): # Si comencem j = k fem els 0's a la columna k
                A[i, j] -= m * A[k, j]

    return A, p




def LUpivesg(A, tol=1.e-10):

    n, r = A.shape

    #comprovem que les dimensions siguin correctes
    if (n != r):
        raise ValueError(f"La matriu no és quadrada!")


    #inicialitzem vector de permutacions
    p = np.arange(n)

    for k in range(0, n - 1):

        #seleccionem el pivot
        indexPivot = k + np.argmax(abs(A[k:, k]) / np.max(abs(A), axis=1)[k:])

        # permutem files
        A[[indexPivot, k]] = A[[k, indexPivot]]

        # actualitzem vector permutacions
        p[indexPivot], p[k] = p[k], p[indexPivot]

        # eliminació Gauss normal
        for i in range(k + 1, n):

            # Mirem que el pivot no sigui massa petit
            if (abs(A[k, k]) < tol):
                raise ValueError("Pivot massa petit! Matriu (pròxima a) singular!")

            m = A[i, k]/A[k, k]

            A[i, k] = m
            for j in range(k + 1, n): # Si comencem j = k fem els 0's a la columna k
                A[i, j] -= m * A[k, j]

    return A, p
