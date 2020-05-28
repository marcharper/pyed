import inspect
import warnings

import numpy as np

from . import incentives, geometries


def normalize(x):
    """Normalizes a numpy array by dividing by the sum."""
    s = float(np.sum(x))
    return x / s


### Simulation ###

# Functions to check exit conditions.
def is_uniform(x, epsilon=0.000000001):
    """Determine if the vector is uniform within epsilon tolerance. Useful to
    stop a simulation if the fitness landscape has become essentially uniform."""
    x_0 = x[0]
    for i in range(1, len(x)):
        if abs(x[i] - x_0) > epsilon:
            return False
    return True


def is_in_simplex(x):
    """Checks if a distribution has exited the simplex."""
    stop = True
    for j in range(x.size):
        if x[j] < 0:
            stop = False
            break
    return stop

### Generators for time-scales. ##


def constant_generator(h):
    while True:
        yield h


def fictitious_play_generator(h):
    i = 1
    while True:
        yield float(h) / (i + 1)
        i += 1

## Functions to actually compute trajectories


def dynamics(state, z, incentive=None, G=None, h=1.0, mu=None, momentum=0, nesterov=False):
    """Compute the next iteration of the dynamic."""
    if not incentive:
        incentive = incentives.DEFAULT_INCENTIVE
    if not G:
        G = geometries.DEFAULT_METRIC
    if mu is None:
        mu = np.eye(len(state))
    ones = np.ones(len(state))
    g = np.dot(np.linalg.inv(G(state)), ones)
    i = incentive(state)
    if not nesterov:
        next_z = momentum * z + (np.dot(i, mu) - g / np.dot(g, ones) * np.sum(i))
        next_state = state + h * next_z
    else:
        next_z = state + h * (np.dot(i, mu) - g / np.dot(g, ones) * np.sum(i))
        next_state = next_z + momentum * (next_z - z)
    # next_state = state + h * (np.dot(i, mu) - g / np.dot(g, ones) * np.sum(i))
    return next_state, next_z


def compute_trajectory(
        initial_state, incentive, iterations=2000, h=1/200., G=None,
        escort=None, exit_on_uniform=True, verbose=False, fitness=None,
        project=True, mu=None, initial_z=(), momentum=None, nesterov=False):
    """Computes a trajectory of a dynamic until convergence or other exit
    condition is reached."""
    # Check if the time-scale is constant or not, and if it is, make it into a generator.
    if not len(initial_z):
        n = len(initial_state)
        initial_z = initial_state.copy() - np.array([1. /n] * n)
    if not momentum:
        momentum = 0
    if not inspect.isgenerator(h):
        h_gen = constant_generator(h)
    else:
        h_gen = h
    # If an escort is given, translate to a metric.
    if escort:
        if G:
            warnings.warn("Both an escort and a metric were supplied to the simulation. Proceeding with the metric only.""")
        else:
            G = geometries.metric_from_escort(escort)
    # Make sure we are starting in the simplex.
    x = normalize(initial_state)
    z = initial_z
    t = []
    for j, h in enumerate(h_gen):
        # Record each point for later analysis.
        t.append(x)
        if verbose:
            if fitness:
                print(j, x, incentive(x), fitness(x))
            else:
                print(j, x, incentive(x))
        ## Exit conditions.
        # Is the landscape uniform, indicating convergence?
        if exit_on_uniform:
            if fitness:
                if is_uniform(fitness(x)):
                    break
            if is_uniform(incentive(x)):
                break
        # Are we out of the simplex?
        if not is_in_simplex(x):
            break
        if j >= iterations:
            break
        ## End Exit Conditions.
        # Iterate the dynamic.
        x, z = dynamics(x, z, incentive=incentive, G=G, h=h, mu=mu,
                        momentum=momentum, nesterov=nesterov)
        # Check to make sure that the distribution has not left the simplex
        # due to round-off.
        # May conflict with out of simplex exit condition, but is useful for
        # non-forward-invariant dynamics (such as projection dynamics). Note
        # that this is very similar to Sandholm's projection and may be
        # better handled that way.
        if project:
            for i in range(len(x)):
                x[i] = max(0, x[i])
        # Re-normalize in case any values were rounded to 0.
        x = normalize(x)
    return t


def two_population_trajectory(params, iterations=2000, verbose=False):
    """Multipopulation trajectory -- each population has its own incentive,
    metric, and time-scale. This function only accepts metrics G and
    generators for h."""
    t = [tuple(normalize(p[0]) for p in params)]
    for j in range(iterations):
        current_state = t[-1]
        h = [p[2].next() for p in params]
        i = params[0][1](current_state[1])
        G = params[0][-1]
        ones = np.ones(len(current_state[0]))
        g = np.dot(np.linalg.inv(G(current_state[0])), ones)
        x = current_state[0] + h[0] * (i - g / np.dot(g, ones) * np.sum(i))
        i = params[1][1](current_state[0])
        G = params[1][-1]
        ones = np.ones(len(current_state[1]))
        g = np.dot(np.linalg.inv(G(current_state[1])), ones)
        y = current_state[1] + h[1] * (i - g / np.dot(g, ones) * np.sum(i))

        for i in range(len(x)):
            x[i] = max(0, x[i])
        for i in range(len(y)):
            y[i] = max(0, y[i])
        x = normalize(x)
        y = normalize(y)
        t.append((x, y))
        if verbose:
            print(x, y)
    return t
    
