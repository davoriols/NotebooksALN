import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg import lu_factor
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import eigs

def pivot_to_permutation(piv):
    perm = np.arange(len(piv))
    for i in range(len(piv)):
        perm[i], perm[piv[i]] = perm[piv[i]], perm[i]
    return perm


A = np.array([[1,1,1,1],
              [1,2,1,1],
              [1,1,3,1],
              [1,1,1,4]], dtype=np.float64)

L, V = eigs(A, k=1, which='LM', tol=1.e-10)
l, v = eigs(A, k=1, which='SM', tol=1.e-10)


print(L)
print(l)

q = 1

L2 , V2 = eigs((A - q*np.eye(len(A))), k=1, which='LM', tol=1.e-10)
l2 , v2 = eigs((A - q*np.eye(len(A))), k=1, which='SM', tol=1.e-10)

print(L2 + q)
print(l2 + q)

