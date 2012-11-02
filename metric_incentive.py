import inspect
import math
import warnings

import numpy
from matplotlib import pyplot

import ternary

### Globals ##
# Plotting options for matplotlib, color list to maintain colors across plots.
#colors = ['r','g','b','k', 'y']
colors = "bgrcmyk"
## Greyscale
#shade_count = 10
#colors = map(str, [0.5 + x / (2. * shade_count) for x in range(1, shade_count+1)])

### Math helpers ##

def product(xs):
    s = 1.
    for x in xs:
        s *= x
    return s    

def normalize(x):
    """Normalizes a numpy array by dividing by the sum."""
    s = float(numpy.sum(x))
    return x / s

def shannon_entropy(p):
    s = 0.
    for i in range(len(p)):
        try:
            s += p[i] * math.log(p[i])
        except ValueError:
            continue
    return -1.*s

def uniform_mutation_matrix(n, ep):
    return (1. - ep) * numpy.eye(n) + ep / (n - 1.) * (numpy.ones(n) - numpy.eye(n))
    
### Information Divergences ##    

def kl_divergence(p, q):
    s = 0.
    for i in range(len(p)):
        try:
            t = p[i] * math.log(p[i] / q[i])
            s += t
        except ValueError:
            continue
    return s

def q_divergence(q):
    """Returns the divergence function corresponding to the parameter value q."""
    if q == 0:
        def d(x, y):
            return 0.5 * numpy.dot((x-y),(x-y))
        return d
    if q == 1:
        return kl_divergence
    if q == 2:
        def d(x,y):
            s = 0.
            for i in range(len(x)):
                s += math.log(x[i] / y[i]) + 1 - x[i] / y[i]
            return -s
        return d
    q = float(q)
    def d(x, y):
        s = 0.
        for i in range(len(x)):
            s += (math.pow(y[i], 2 - q) - math.pow(x[i], 2 - q)) / (2 - q)
            s -= math.pow(y[i], 1 - q) * (y[i] - x[i])
        s = -s / (1 - q)
        return s
    return d

### Escorts ###    
    
def DEFAULT_ESCORT(x):
    """Gives Shahshahani metric and KL-divergence."""
    return x

def twisted_escort(x):
    l = list(x)
    return numpy.array([l[1],l[2],l[0]])

## Just use power escort with p = 0
#def projection_escort(x):
    #return numpy.ones(len(x))

def power_escort(q):
    """Returns an escort function for the power q."""
    def g(x):
        y = []
        for i in range(len(x)):
            y.append(math.pow(x[i], q))
        return numpy.array(y)
    return g

def exponential_escort(x):
    return numpy.exp(x)

### Metrics ##

# Can also use metric_from_escort to get the Euclidean metric.
def euclidean_metric(n=3):
    I = numpy.identity(3)
    def G(x):
        return I
    return G

def metric_from_escort(escort):
    def G(x):
        #return numpy.linalg.inv(numpy.diag(escort(x)))
        return numpy.diag(1./ escort(x))
    return G

def shahshahani_metric():
    return metric_from_escort(DEFAULT_ESCORT)

DEFAULT_METRIC = shahshahani_metric()    

### Incentives ##

def rock_scissors_paper(a=1, b=1):
    return [[0,-b,a], [a, 0, -b], [-b, a, 0]]

def linear_fitness(m):
    """f(x) = mx for a matrix m."""
    m = numpy.array(m)
    def f(x):
        return numpy.dot(m, x)
    return f

def replicator_incentive(fitness):
    def g(x):
        return x * fitness(x)
    return g    

def DEFAULT_INCENTIVE(f):
    return replicator_incentive(f)
    
def replicator_incentive_power(fitness, q):
    def g(x):
        y = []
        for i in range(len(x)):
            y.append(math.pow(x[i], q))
        y = numpy.array(y)
        return y * fitness(x)
    return g

def best_reply_incentive(fitness):
    """Compute best reply to fitness landscape at state."""
    def g(state):
        f = fitness(state)
        try:
            dim = state.size
        except AttributeError:
            state = numpy.array(state)
            dim = state.size
        replies = []
        for i in range(dim):
            x = numpy.zeros(dim)
            x[i] = 1
            replies.append(numpy.dot(x, f))
        replies = numpy.array(replies)
        i = numpy.argmax(replies)
        x = numpy.zeros(dim)
        x[i] = 1
        return x
    return g 
    
def logit_incentive(fitness, eta):
    def f(x):
        return normalize(numpy.exp(fitness(x) / eta))
    return f
    
### Simulation ###

# Functions to check exit conditions.
def is_uniform(x, epsilon=0.001):
    """Determine if the vector is uniform within epsilon tolerance. Useful to stop a simulation if the fitness landscape has become essentially uniform."""
    x_0 = x[0]
    for i in range(1, len(x)):
        if abs(x[i] - x_0) > epsilon:
            return False
    return True

def is_in_simplex(x):
    """Checks if a distribution has exited the simplex."""
    stop = True
    for j in range(x.size):
        #print x[j]
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

def dynamics(state, incentive=None, G=None, h=1.0, mu=None):
    """Compute the next iteration of the dynamic."""
    if not incentive:
        incentive = DEFAULT_INCENTIVE
    if not G:
        G = shahshahani_metric()
    if mu is None:
        mu = numpy.eye(len(state))
    ones = numpy.ones(len(state))
    g = numpy.dot(numpy.linalg.inv(G(state)), ones)
    i = incentive(state)
    next_state = state + h * (numpy.dot(i, mu) - g / numpy.dot(g, ones) * numpy.sum(i))
    return next_state

def compute_trajectory(initial_state, incentive, iterations=2000, h=1/200., G=None, escort=None, exit_on_uniform=True, verbose=False, fitness=None, project=False, mu=None):
    """Computes a trajectory of a dynamic until convergence or other exit condition is reached."""
    # Check if the time-scale is constant or not, and if it is, make it into a generator.
    if not inspect.isgenerator(h):
        h_gen = constant_generator(h)
    else:
        h_gen = h
    # If an escort is given, translate to a metric.
    if escort:
        if G:
            warnings.warn("Both an escort and a metric were supplied to the simulation. Proceeding with the metric only.""")
        else:
            G = metric_from_escort(escort)
    # Make sure we are starting in the simplex.
    x = normalize(initial_state)
    t = []
    for j, h in enumerate(h_gen):
        # Record each point for later analysis.
        t.append(x)
        if verbose:
            if fitness:
                print j, x, incentive(x), fitness(x)
            else:
                print j, x, incentive(x)
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
        x = dynamics(x, incentive=incentive, G=G, h=h, mu=mu)
        # Check to make sure that the distribution has not left the simplex due to round-off.
        # May conflict with out of simplex exit condition, but is useful for non-forward-invariant dynamics (such as projection dynamics). Note that this is very similar to Sandholm's projection and may be better handled that way.
        if project:
            for i in range(len(x)):
                x[i] = max(0, x[i])
        #Re-normalize in case any values were rounded to 0.
        x = normalize(x)
    return t    

def two_population_trajectory(params, iterations=2000, exit_on_uniform=True, verbose=False):
    """Multipopulation trajectory -- each population has its own incentive, metric, and time-scale. This function only accepts metrics G and generators for h."""
    t = [tuple(normalize(p[0]) for p in params)]
    for j in range(iterations):
        current_state = t[-1]
        h = [p[2].next() for p in params]
        i = params[0][1](current_state[1])
        G = params[0][-1]
        ones = numpy.ones(len(current_state[0]))
        g = numpy.dot(numpy.linalg.inv(G(current_state[0])), ones)
        #print i, g, h[0]
        x = current_state[0] + h[0] * (i - g / numpy.dot(g, ones) * numpy.sum(i))
        i = params[1][1](current_state[0])
        G = params[1][-1]
        ones = numpy.ones(len(current_state[1]))
        g = numpy.dot(numpy.linalg.inv(G(current_state[1])), ones)
        y = current_state[1] + h[1] * (i - g / numpy.dot(g, ones) * numpy.sum(i))

        for i in range(len(x)):
            x[i] = max(0, x[i])
        for i in range(len(y)):
            y[i] = max(0, y[i])
        x = normalize(x)
        y = normalize(y)
        t.append((x, y))
        if verbose:
            print x, y
    return t
    
### Analysis ##    

def relative_prediction_power(new, old):
    return product(new) / product(old)

def compute_iss_diff(e, x, incentive):
    """Computes the difference of the LHS and RHS of the ISS condition."""
    i = incentive(x)
    s = numpy.sum(incentive(x))
    lhs = sum(e[j] / x[j] * i[j] for j in range(x.size))
    return lhs - s

def eiss_diff_func(e, incentive, escort=None):
    if not escort:
        escort = DEFAULT_ESCORT
    def f(x):
        es = escort(x)
        inc = incentive(x)
        s = sum((e[i] - x[i])*inc[i] / es[i] for i in range(len(x)))
        return s
    return f    

def G_iss_diff_func(e, incentive, G=None):
    if not G:
        G = DEFAULT_METRIC
    def f(x):
        g = G(x)
        inc = incentive(x)
        return numpy.dot((e - x), numpy.dot(G(x), incentive(x)))
    return f

### Examples and Tests ##    

def divergence_test(a=0, b=4, steps=100):
    """Compare the q-divergences for various values to illustrate that q_1-div > q_2-div if q_1 > q_2."""
    points = []
    x = numpy.array([1./3, 1./3, 1./3])
    y = numpy.array([1./2, 1./4, 1./4])
    d = float(b - a) / steps
    for i in range(0, steps):
        q = a + i * d
        div = q_divergence(q)
        points.append((q, div(x, y)))
    pyplot.plot([x for (x,y) in points], [y for (x,y) in points])
    pyplot.show()

def basic_example():
    # Projection dynamic.
    initial_state = normalize(numpy.array([1,1,4]))
    m = rock_scissors_paper(a=1., b=-2.)
    fitness = linear_fitness(m)
    incentive = replicator_incentive_power(fitness, 0)
    mu = uniform_mutation_matrix(3, ep=0.2)
    t = compute_trajectory(initial_state, incentive, escort=power_escort(0), iterations=10000, verbose=True, mu=mu)
    ternary.plot(t, linewidth=2)
    ternary.draw_boundary()    

    ## Lyapunov Quantities
    pyplot.figure()
    # Replicator Lyapunov
    e = normalize(numpy.array([1,1,1]))
    v = [kl_divergence(e, x) for x in t]
    pyplot.plot(range(len(t)), v, color='b')
    d = q_divergence(0)
    v = [d(e, x) for x in t]
    pyplot.plot(range(len(t)), v, color='r')    
    pyplot.show()
    
    ## Some example code that may be useful for best reply...
    # Traditional best reply Lyapunov
    #v = [numpy.max(fitness(x)) - numpy.dot(x, fitness(x)) for x in t]
    #pyplot.plot(range(len(t)), v, color='b')
    #pyplot.axhline(y=0)

    #def metric(x):
        #x_1, x_2, x_3 = x[0], x[1], x[2]
        #return numpy.array([[1., 1./x_2, 0], [0, 1./x_2, 1.], [1., 0, 1./x_3]])
    #t = compute_trajectory(initial_state, incentive, G=metric)
    
if __name__ == "__main__":
    #divergence_test()
    basic_example()
    #two_population_test()
    
