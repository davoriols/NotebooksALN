import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg import lu_factor

# Funció pel mètode de la potència
def potIter(A, z0, tol=1.e-10, maxIter=100):

    z = z0.copy()
    sigma0 = z0.T@A@z0

    for i in range(maxIter):
        Az0 = A@z0
        z = Az0 / np.linalg.norm(Az0)

        sigma = z.T@A@z

        if (np.abs(sigma - sigma0) < tol):
            return z, sigma, i + 1

        z0 = z.copy()
        sigma0 = sigma

    return z, sigma, -(i + 1)



# Funció per recuperar el vector de permutacions
def pivot_to_permutation(piv):
    perm = np.arange(len(piv))
    for i in range(len(piv)):
        perm[i], perm[piv[i]] = perm[piv[i]], perm[i]
    return perm


# Funció pel mètode de la potència inversa
def potInv(A, z0, tol=1.e-10, maxIter=10000):

    # Definim les matrius per resoldre el sistema
    M, piv = lu_factor(A)
    p = pivot_to_permutation(piv)
    U = np.triu(M)
    L = np.tril(M, -1) + np.eye(len(M))

    z = z0.copy()
    sigma0 = z0.T@A@z0
    y = z0 / np.linalg.norm(z0)

    for i in range(maxIter):

        sol1 = solve_triangular(L, y[p[:]], lower=True)
        z = solve_triangular(U, sol1)

        # Normalitzem el vector
        y = z / np.linalg.norm(z)

        # Mirem si hem assolit la precisió desitjada
        sigma = y.T@A@y

        if (np.abs(sigma - sigma0) < tol):
            return y, sigma, i + 1

        sigma0 = sigma.copy()

    return z, sigma, -(i + 1)
