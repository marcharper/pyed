import numpy as np
import pytest

import pyed
from pyed import dynamics

# Set random seed
np.random.seed(10)


def dynamics_example(a=1, b=2, power=1, initial_state=None, momentum=0., iterations=1000, alpha=1e-2):
	if initial_state is None:
		initial_state = pyed.normalize(np.array([1, 1, 4]))
	m = pyed.incentives.rock_paper_scissors(a=a, b=b)
	fitness = pyed.incentives.linear_fitness(m)
	incentive = pyed.incentives.replicator_incentive_power(fitness, power)
	mu = pyed.incentives.uniform_mutation_matrix(3, ep=0.2)
	t = pyed.dynamics.compute_trajectory(
		initial_state, incentive, escort=pyed.geometries.power_escort(power), iterations=iterations, verbose=False,
		mu=mu, momentum=momentum, h=alpha)

	e = pyed.normalize(np.array([1, 1, 1]))
	d = pyed.information.q_divergence(power)
	v = [d(e, x) for x in t]
	return t[-1], v[-1]


def test_convergence():
	# These examples converge to the interior of the simplex
	parameters = [
		(1, 1, 1),  # replicator with hawk-dove like landscape,
		(2, 1, 1),  # replicator with rock-paper-scissors like landscape,
		(1, 1, 0),  # projection with rock-paper-scissors like landscape,
		(2, 1, 0),  # projection with rock-paper-scissors like landscape,
		(1, 1, 2),  # poincare with hawk-dove like landscape,
	]
	for a, b, power in parameters:
		initial_state = pyed.normalize(np.array([1, 1, 4]))
		x, v = dynamics_example(a, b, power, initial_state=initial_state, iterations=10000, alpha=0.1)
		# assert (dynamics.is_uniform(x) == True)
		assert abs(v) < 1e-8


def test_cycling():
	# These cycle indefinitely
	parameters = [
		(1, -1, 1),  # replicator with hawk-dove like landscape,
		(1, -1, 0),  # projection with rock-paper-scissors like landscape,
	]
	for a, b, power in parameters:
		initial_state = pyed.normalize(np.array([2, 1, 4]))
		x, v = dynamics_example(a, b, power, initial_state=initial_state)
		assert (dynamics.is_uniform(x) == False)
		assert v > 1e-3


def test_divergence():
	# These diverge to the boundary
	parameters = [
		(1, -2, 1),  # replicator with hawk-dove like landscape,
		(1, -2, 0),  # projection with rock-paper-scissors like landscape,
	]
	for a, b, power in parameters:
		initial_state = pyed.normalize(np.array([1, 1, 8]))
		x, v = dynamics_example(a, b, power, initial_state=initial_state, iterations=20000)
		assert (dynamics.is_uniform(x) == False)
		assert abs(np.prod(x)) < 1e-2
		assert v > 1e-3


def test_polyak_momentum():
	# These examples converge to the interior of the simplex
	parameters = [
		(1, 1, 1),  # replicator with hawk-dove like landscape,
		(2, 1, 1),  # replicator with rock-paper-scissors like landscape,
		(1, 1, 0),  # projection with rock-paper-scissors like landscape,
		(2, 1, 0),  # projection with rock-paper-scissors like landscape,
		(1, 1, 2),  # poincare with hawk-dove like landscape,
	]
	for a, b, power in parameters:
		for momentum in [-0.2, -0.1, 0.1, 0.5]:
			initial_state = pyed.normalize(np.array([1, 1, 4]))
			x, v = dynamics_example(
				a, b, power, initial_state=initial_state, momentum=momentum, iterations=20000, alpha=0.01)
			assert abs(v) < 1e-4

	# These examples diverge because of the momentum
	parameters = [
		(1, 1, 1),  # replicator with hawk-dove like landscape,
		(2, 1, 1),  # replicator with rock-paper-scissors like landscape,
		(1, 1, 0),  # projection with rock-paper-scissors like landscape,
		(2, 1, 0),  # projection with rock-paper-scissors like landscape,
		# (1, 1, 2),  # poincare with hawk-dove like landscape,
	]
	for a, b, power in parameters:
		for momentum in [1.1, 1.2]:
			initial_state = pyed.normalize(np.array([1, 1, 4]))
			x, v = dynamics_example(
				a, b, power, initial_state=initial_state, momentum=momentum, iterations=1000, alpha=0.01)
			assert abs(v) > 1e-2


