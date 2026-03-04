import numpy as np

# Codi pel mètode del gradient
def grad(A,b,x0,atol=1.e-10,rtol=1.e-10,maxIter=100):
    r = b - A@x0
    x = x0.copy()

    # Iterem maxIter vegades, si assolim la precisió desitjada abans, sortim del bucle amb el return
    for k in range(maxIter):
        Ark = A@r #guardem aquest valor per estalviar productes de matrius

        alpha = np.dot(r, r) / np.dot(r,Ark)
        x = x + alpha*r

        r = r - alpha*Ark

        # Mirem si hem assolit la presició desitjada
        absolut = np.linalg.norm(r)
        relatiu = absolut/np.linalg.norm(b)

        if (absolut < atol and relatiu < rtol):
            return x, k + 1

    if (absolut >= atol and relatiu >= rtol):
        return x, -3

    elif (relatiu >= rtol):
        return x, -2

    else:
        return x, -1


# Codi pel mètode del gradient conjugat
def gradConj(A,b,x0,atol=1.e-10,rtol=1.e-10,maxIter=100):

    r = b - A@x0
    p = r.copy()
    x = x0.copy()
    gamma = np.dot(r, r)

    # Iterem maxIter vegades, si assolim la precisió desitjada abans, sortim del bucle amb el return
    for k in range(maxIter):
        # Fem el mètode del gradient conjugat
        y = A@p

        alpha = gamma / np.dot(p, y)

        x = x + alpha*p

        r = r - alpha*y

        beta = np.dot(r, r) / gamma

        gamma = np.dot(r, r)
        p = r + beta*p

        # Mirem si hem assolit la precisió desitjada
        absolut = np.linalg.norm(r)
        relatiu = absolut/np.linalg.norm(b)

        if (absolut < atol and relatiu < rtol):
            return x, k + 1

    if (absolut >= atol and relatiu >= rtol):
        return x, -3

    elif (relatiu >= rtol):
        return x, -2

    else:
        return x, -1
