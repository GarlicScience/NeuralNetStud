import numpy as np


def sgn(x, alpha=0):
    return 1 if x > 0 else (0 if alpha else -1)


def dsgn(x, alpha=0):
    return 1


def logistic(x, alpha=0.01):
    assert alpha != 0, 'ERR: wrong alpha ' + str(alpha)
    return 1/(1 + np.exp(-alpha * x))


def dlogistic(x, alpha):
    return alpha * logistic(x, alpha) * (1 - logistic(x, alpha))