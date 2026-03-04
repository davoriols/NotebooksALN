import numpy as np

def jacobi(A, tol=1.e-10,maxIter=1.e5):

    # Definim la matriu on guardarem les rotacions:
    Q = np.eye(len(A))

    for i in range(int(maxIter)):
        # Trobem l'element més gran en mòdul
        (p, q) = np.unravel_index(np.argmax(abs(np.triu(A, 1))), A.shape)

        # Triem l'angle de la rotació
        n = (A[p, p] - A[q, q])/(2*A[p, q])

        if (n > 0):
            t = -n - np.sqrt(n**2 + 1)
            cos = 1/np.sqrt(1 + t**2)
            sin = t*cos

        elif (n < 0):
            t = -n + np.sqrt(n**2 + 1)
            cos = 1/np.sqrt(1 + t**2)
            sin = t*cos

        # Si n == 0, sobreescrivim cos i sen
        if (n == 0):
            cos = sin = np.sqrt(2)/2

        # Definim la matriu de rotació i la guardem
        R = cos*np.eye(2)
        R[0, 1] = -sin
        R[1, 0] = sin

        Q[:, [p, q]] = Q[:, [p, q]]@R

        # Apliquem la rotació
        A[[p,q], :] = R.T@A[[p,q], :]
        A[:, [p,q]] = A[:, [p, q]]@R

        # Mirem si hem assolit la precisió desitjada
        if (np.max(np.triu(A, 1)) < tol):
            return np.diag(A), Q, i + 1

    # Si no trobem cap solució retornem les iteracions negatives
    return np.diag(A), Q, -(i + 1)
