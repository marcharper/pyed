import matplotlib.pyplot as plt
import numpy as np

import ternary
import pyed


def divergence_test(a=0, b=4, steps=100):
    """Compare the q-divergences for various values to illustrate that q_1-div > q_2-div if q_1 > q_2."""
    points = []
    x = np.array([1./3, 1./3, 1./3])
    y = np.array([1./2, 1./4, 1./4])
    d = float(b - a) / steps
    for i in range(0, steps):
        q = a + i * d
        div = pyed.information.q_divergence(q)
        points.append((q, div(x, y)))
    plt.plot([x for (x,y) in points], [y for (x,y) in points])
    plt.show()


def basic_example():
    # Projection dynamic.
    initial_state = pyed.normalize(np.array([1, 1, 4]))
    m = pyed.incentives.rock_paper_scissors(a=1., b=-2.)
    fitness = pyed.incentives..linear_fitness(m)
    incentive = pyed.incentives.replicator_incentive_power(fitness, 0)
    mu = pyed.incentives.uniform_mutation_matrix(3, ep=0.2)
    t = pyed.dynamics.compute_trajectory(
        initial_state, incentive, escort=pyed.geometries.power_escort(0), iterations=10000, verbose=True, mu=mu)
    figure, tax = ternary.figure()
    tax.plot(t, linewidth=2, color="black")
    tax.boundary()

    ## Lyapunov Quantities
    plt.figure()
    # Replicator Lyapunov
    e = pyed.normalize(np.array([1, 1, 1]))
    v = [pyed.information.kl_divergence(e, x) for x in t]
    plt.plot(range(len(t)), v, color='b')
    d = pyed.information.q_divergence(0)
    v = [d(e, x) for x in t]
    plt.plot(range(len(t)), v, color='r')
    plt.show()


if __name__ == "__main__":
    divergence_test()
    basic_example()
