from math import pow
import numpy as np


def DEFAULT_ESCORT(x):
    """Gives Shahshahani metric and KL-divergence."""
    return x


def twisted_escort(x):
    l = list(x)
    return np.array([l[1], l[2], l[0]])


def power_escort(q):
    """Returns an escort function for the power q."""

    def g(x):
        y = []
        for i in range(len(x)):
            y.append(pow(x[i], q))
        return np.array(y)

    return g


def projection_escort(x):
    return power_escort(0)


def exponential_escort(x):
    return np.exp(x)


# Can also use metric_from_escort to get the Euclidean metric.
def euclidean_metric(n=3):
    I = np.identity(n)

    def G(x):
        return I
    return G


def metric_from_escort(escort):
    def G(x):
        return np.diag(1. / escort(x))
    return G


def shahshahani_metric():
    """Also known as the Fisher information metric."""
    return metric_from_escort(DEFAULT_ESCORT)


DEFAULT_METRIC = shahshahani_metric()
