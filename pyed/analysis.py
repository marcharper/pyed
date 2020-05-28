import numpy as np
from .geometries import DEFAULT_ESCORT, DEFAULT_METRIC


def product(xs):
    s = 1.
    for x in xs:
        s *= x
    return s


def relative_prediction_power(new, old):
    return product(new) / product(old)


def compute_iss_diff(e, x, incentive):
    """Computes the difference of the LHS and RHS of the ISS condition."""
    i = incentive(x)
    s = np.sum(incentive(x))
    lhs = sum(e[j] / x[j] * i[j] for j in range(x.size))
    return lhs - s


def eiss_diff_func(e, incentive, escort=None):
    if not escort:
        escort = DEFAULT_ESCORT

    def f(x):
        es = escort(x)
        inc = incentive(x)
        s = sum((e[i] - x[i]) * inc[i] / es[i] for i in range(len(x)))
        return s
    return f


def G_iss_diff_func(e, incentive, G=None):
    if not G:
        G = DEFAULT_METRIC

    def f(x):
        g = G(x)
        return np.dot((e - x), np.dot(G(x), incentive(x)))
    return f

