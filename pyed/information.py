from math import log, pow
import numpy as np
from scipy import stats


def shannon_entropy(p):
    return stats.entropy(p)


def kl_divergence(p, q):
    """Standard KL divergence."""
    return stats.entropy(p, q)


def q_divergence(q):
    """Returns the q-divergence function corresponding to the parameter value q."""
    if q == 0:
        def d(x, y):
            return 0.5 * np.dot((x - y), (x - y))
        return d

    if q == 1:
        return kl_divergence

    if q == 2:
        def d(x,y):
            s = 0.
            for i in range(len(x)):
                s += log(x[i] / y[i]) + 1 - x[i] / y[i]
            return -s
        return d

    q = float(q)

    def d(x, y):
        s = 0.
        for i in range(len(x)):
            s += (pow(y[i], 2 - q) - pow(x[i], 2 - q)) / (2 - q)
            s -= pow(y[i], 1 - q) * (y[i] - x[i])
        s = -s / (1 - q)
        return s
    return d
