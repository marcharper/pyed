"""Fitness landscapes, incentives, and various matrices."""

from math import pow
import numpy as np


def normalize(x):
    """Normalizes a numpy array by dividing by the sum."""
    s = float(np.sum(x))
    return x / s


def uniform_mutation_matrix(n, ep):
    return (1. - ep) * np.eye(n) + ep / (n - 1.) * (np.ones(n) - np.eye(n))


def rock_paper_scissors(a=1, b=1):
    return [[0, a, b], [b, 0, a], [a, b, 0]]


def linear_fitness(m):
    """f(x) = mx for a matrix m."""
    m = np.array(m)

    def f(x):
        return np.dot(m, x)
    return f


def replicator_incentive(fitness):
    def g(x):
        return x * fitness(x)
    return g


def replicator_incentive_power(fitness, q):
    def g(x):
        y = []
        for i in range(len(x)):
            y.append(pow(x[i], q))
        y = np.array(y)
        return y * fitness(x)
    return g


def best_reply_incentive(fitness):
    """Compute best reply to fitness landscape at state."""
    def g(state):
        f = fitness(state)
        try:
            dim = state.size
        except AttributeError:
            state = np.array(state)
            dim = state.size
        replies = []
        for i in range(dim):
            x = np.zeros(dim)
            x[i] = 1
            replies.append(np.dot(x, f))
        replies = np.array(replies)
        i = np.argmax(replies)
        x = np.zeros(dim)
        x[i] = 1
        return x
    return g


def logit_incentive(fitness, eta):
    def f(x):
        return normalize(np.exp(fitness(x) / eta))
    return f


def fermi_incentive(fitness, beta):
    def f(x):
        return normalize(np.exp(fitness(x) * beta))
    return f


def DEFAULT_INCENTIVE(f):
    return replicator_incentive(f)

