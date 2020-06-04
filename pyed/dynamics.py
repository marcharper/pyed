import inspect
import warnings

import numpy as np

from . import incentives, geometries, information


def normalize(x):
    """Normalizes a numpy array by dividing by the sum."""
    s = float(np.sum(x))
    return x / s


### Simulation ###

# Functions to check exit conditions.
def is_uniform(x, epsilon=1e-9):
    """Determine if the vector is uniform within epsilon tolerance. Useful to
    stop a simulation if the fitness landscape has become essentially uniform."""
    x_0 = x[0]
    for i in range(1, len(x)):
        if abs(x[i] - x_0) > epsilon:
            return False
    return True


def is_in_simplex(x):
    """Checks if a distribution has exited the simplex. Assumes that the distribution has been normalized."""
    for j in range(x.size):
        if x[j] < 0 or x[j] > 1:
            return False
    return True

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


def potential(state, incentive=None, G=None, mu=None):
    """Compute the potential function for a given state.

    incentive
        function of population vector that incorporates the payoff / fitness landscape
    G
        Riemannian metric, a matrix-valued function of the population vector
    mu, optional
        mutation vector

    """
    if not incentive:
        incentive = incentives.DEFAULT_INCENTIVE
    if not G:
        G = geometries.DEFAULT_METRIC
    if mu is None:
        mu = np.eye(len(state))
    ones = np.ones(len(state))
    g = np.dot(np.linalg.inv(G(state)), ones)
    i = incentive(state)
    return np.dot(i, mu) - g / np.dot(g, ones) * np.sum(i)


def dynamics(state, z, incentive=None, G=None, h=1.0, mu=None, momentum=0):
    """Compute the next iteration of the dynamic."""
    U = potential(state, incentive=incentive, G=G, mu=mu)
    next_z = momentum * z + U
    next_state = state + h * next_z
    return next_state, next_z


def nesterov_dynamics(state, z, incentive=None, G=None, h=1.0, mu=None, momentum=0):
    """Compute the next iteration of the dynamic with Nesterov momentum."""
    lookahead_state = normalize(state + momentum * z)
    U = potential(lookahead_state, incentive=incentive, G=G, mu=mu)
    next_z = momentum * z + U
    next_state = state + h * next_z
    return next_state, next_z


def compute_trajectory(
        initial_state, incentive, iterations=2000, h=1/200., G=None,
        escort=None, exit_on_uniform=True, exit_on_divergence_tol=False,
        divergence_tol=.001, stable_state=None, verbose=False,
        fitness=None, project=True, mu=None, initial_z=(),
        momentum=None, divergence=None, nesterov=False):
    """Computes a trajectory of a dynamic until convergence or other exit condition is reached.

    initial_state
        initial population distribution
    incentive
        function of population vector that incorporates the payoff / fitness landscape
    iterations
        maximum number of iterations to run, if exit conditions are not met
    h, float
        step size aka learning rate
    G
        Riemannian metric, a matrix-valued function of the population vector
    escort
        function which generates a Riemannan metric
    exit_on_uniform, bool
        whether to exit when the population becomes uniform
    exit_on_divergence_tol, bool
        whether to exit when the population reaches near convergence, according to the divergence tolerance
    divergence_tol, float
        tolerance for determining convergence
    stable_state, None
        an ESS (population state) to measure convergence to
    verbose, bool
        whether to report verbose information
    fitness
        a function of the population vector
    project, bool
        if the trajectory exits the simple, project back into it
    mu, optional
        mutation vector
    initial_z
        if momentum is used, supply an initial z vector
    momentum, float
        value of momentum to use
    divergence
        supplied divergence to measure convergence
    nesterov
        whether to use the Nesterov version of momentum
    """
    ## Setup and check inputs
    # Check if the time-scale is constant or not, and if it is, make it into a generator.
    if not len(initial_z):
        n = len(initial_state)
        initial_z = initial_state.copy() - np.array([1. /n] * n)
    if not momentum:
        momentum = 0
    if not divergence:
        divergence = information.kl_divergence
    if not inspect.isgenerator(h):
        h_gen = constant_generator(h)
    else:
        h_gen = h
    # If an escort is given, translate to a metric.
    if escort:
        if G:
            warnings.warn(
                "Both an escort and a metric were supplied to the simulation. Proceeding with the metric only.""")
        else:
            G = geometries.metric_from_escort(escort)
    # Make sure we are starting in the simplex.
    x = normalize(initial_state)
    z = initial_z
    t = []
    if not stable_state:
        stable_state = np.ones_like(initial_state) / initial_state.shape[0]

    ## Iterate the dynamics.
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
                    return t
            if is_uniform(incentive(x)):
                return t
        if exit_on_divergence_tol:
            if divergence(stable_state, x) < divergence_tol:
                return t
        # Are we out of the simplex?
        if not is_in_simplex(x):
            break
        if j >= iterations:
            break
        ## End Exit Conditions.

        # Iterate the dynamic.
        if nesterov:
          x, z = nesterov_dynamics(x, z, incentive=incentive, G=G, h=h, mu=mu, momentum=momentum)
        else:
          x, z = dynamics(x, z, incentive=incentive, G=G, h=h, mu=mu, momentum=momentum)
        # Check to make sure that the distribution has not left the simplex
        # due to round-off.
        # May conflict with out of simplex exit condition, but is useful for
        # non-forward-invariant dynamics (such as projection dynamics). Note
        # that this is very similar to Sandholm's projection and may be
        # better handled that way.
        if project:
            x = np.clip(x, a_min=0, a_max=np.inf)
        # Re-normalize in case any values were rounded to 0.
        x = normalize(x)
    return t


def replicator_trajectory(initial_state, fitness, iterations=2000, h=1/200., verbose=False, momentum=None,
                          exit_on_uniform=True, exit_on_divergence_tol=True, divergence_tol=.001, nesterov=False):
    """Convenience function for replicator dynamics."""
    incentive = incentives.replicator_incentive_power(fitness, 1)
    return compute_trajectory(
        initial_state,
        incentive,
        iterations=iterations,
        h=h,
        verbose=verbose,
        momentum=momentum,
        exit_on_uniform=exit_on_uniform,
        exit_on_divergence_tol=exit_on_divergence_tol,
        divergence_tol=divergence_tol,
        nesterov=nesterov
    )


def projection_trajectory(initial_state, fitness, iterations=2000, h=1/200., verbose=False, momentum=None,
                          exit_on_uniform=True, exit_on_divergence_tol=True, divergence_tol=.001, nesterov=False):
    incentive = incentives.replicator_incentive_power(fitness, 0)
    """Convenience function for projection dynamics."""
    return compute_trajectory(
        initial_state,
        incentive,
        iterations=iterations,
        escort=geometries.power_escort(0),
        h=h,
        verbose=verbose,
        momentum=momentum,
        exit_on_uniform=exit_on_uniform,
        exit_on_divergence_tol=exit_on_divergence_tol,
        divergence_tol=divergence_tol,
        nesterov=nesterov
    )


def two_population_trajectory(params, iterations=2000, verbose=False):
    """Multipopulation trajectory -- each population has its own incentive, metric, and time-scale. This function only
    accepts metrics G and generators for h."""
    t = [tuple(normalize(p[0]) for p in params)]
    for _ in range(iterations):
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

