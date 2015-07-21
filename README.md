
# metric-incentive-dynamics

This python script computes time-scale dynamics for the metric-incentive dynamic. Software dependencies include numpy, scipy, and python-ternary for graphing (which also requires matplotlib). python-ternary is available on github at https://github.com/marcharper/python-ternary All the others are standard python libraries that are available on most platforms (e.g. through a package manager on linux).

Basic Usage
-----------

The main function is compute_trajectory in the file metric_incentive.py, which takes several parameters:

   def compute_trajectory(initial_state, incentive, iterations=2000, h=1/200., G=None, escort=None, exit_on_uniform=True, verbose=False, fitness=None):

Let us consider each in turn. The initial_state is a numpy array that indicates the starting point of the dynamic. For instance, to start at the center of the simplex in a three-type dynamic, use:

```
    from metric_incentive import *
    import numpy
    initial_state = normalize(numpy.array([1,1,1]))
```

Strictly speaking, the normalization is not necessary (compute_trajectory will do it for you), nevertheless the normalizations function *normalize* is available.

The second parameter, *incentive* is a function that takes a state and produces a vector (a numpy array) of the incentive values. Several incentives are included, such as *replicator_incentive*, which takes a fitness landscape (again a function taking a state to a vector) and produces an incentive:

```
    m = rock_scissors_paper(a=1, b=-2)
    print m
    fitness = linear_fitness(m)
    print fitness(initial_state)
    incentive = replicator_incentive(fitness)
    print incentive(normalize(numpy.array([1,1,4])))
```

This outputs:

```
    [[0, 2, 1], [1, 0, 2], [2, 1, 0]]
    array([ 1.,  1.,  1.])
    array([ 0.16666667,  0.25      ,  0.33333333])
```

*iterations* (optional: default=2000) is the maximum number of iterations that the dynamic will step through unless an exit condition is reached.

*h* (optional but you probably want to change it) is a constant (should be between 0 and 1) corresponding to the time scale, or a generator that produces successful values that are not necessarily the same.

*G* (optional) is a Riemannian metric given as a function of a simplex variable. Again there are several helpers, such as *shahshahani_metric()* included in the code (which is the default). This parameter must return numpy array matrices at input points on the simplex.

*escort* is another optional functional parameter that can be used instead of an entire metric (technically it defines a diagonal metric). If you specific both a metric and an escort, just the metric is used (and a warning is given).

*exit_on_uniform* gives an exit condition to stop iteration early if the incentive vector is uniform, or very nearly so (which indicates convergence).

*verbose* if True outputs each step of the dynamic to standard out.

*fitness* is optional and used for reporting if verbose == True

Example
-------

The function *basic_example* shows how to compute a trajectory, plot it in the simplex (for n=3), and plot candidate Lyapunov functions on the trajectory.
