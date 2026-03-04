import numpy as np

# Exercici 1

def matHouseholder(u):
    n = len(u)
    H = np.eye(n) - 2*np.outer(u,u)/np.dot(u,u)
    return H

def qrHouseholder(A):

    m, n = A.shape

    Q = np.eye(m)
    R = A.copy()

    for k in range(0, n):
        u = R[k:, k].copy()

        # evitem el cas on sign(x) = 0, ja que aleshores multipliquem per 0
        if (np.sign(u[0]) == 0):
            u[0] -= np.linalg.norm(R[k:,k])
        else:
            u[0] += np.sign(u[0])*np.linalg.norm(R[k:, k])

        H = np.eye(m)
        H[k:, k:] = matHouseholder(u)

        # Acumulem les matrius de householder en Q i R correponentment
        R = H@R
        Q = Q@H.T

    return Q, R # Q és una matriu quadrada

"""
A = np.array([[1, 0, 1],
              [0, 0, 1],
              [1,-2, 1],
              [0, 1, 0]],dtype = np.float64)

Q, R = qrHouseholder(A)

print(R)
print(Q@R) # retorna A

"""


# Exercici 2

def qrsys(A,b):
    m, n = A.shape

    R = A.copy()

    for k in range(0, n):
        u = R[k:, k].copy()

        R[k:, k] = np.zeros(m - k)

        # distingim igual que abans quan sign(x) = 0
        # Per evitar multiplicacions de matrius, podem calcular 
        # la primera columna de R directament
        if (np.sign(u[0]) == 0):
            R[k, k] += np.linalg.norm(u)
        else:
            R[k, k] -= np.sign(u[0])*np.linalg.norm(u)

        # Trobem Qtb (la variable b) de manera optimitzada
        u = u - R[k:, k]

        beta = 2/np.dot(u,u)

        b[k:] = b[k:] - u*(beta*(np.dot(u,b[k:])))

        # Trobem la resta de les columnes de R sense multiplicar matrius
        for i in range(k + 1, n):
            R[k:, i] = R[k:, i] - u*(beta*(np.dot(u,R[k:, i])))


    return R, b # R matriu quadrada, Qtb vector

"""
A = np.array([[1, 0, 1],
              [0, 0, 1],
              [1,-2, 1],
              [0, 1, 0]],dtype = np.float64)

b = np.array([1, 2, 1, 3], dtype = np.float64)

R, Qtb = qrsys(A, b)

print(R) # retorna la mateixa R que en el exercici 1
print(Qtb)
"""


