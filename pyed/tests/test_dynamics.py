from .. import dynamics
import numpy as np
import pytest

def test_normalize():
	assert np.allclose(dynamics.normalize(np.ones(3)), (1/3) * np.ones(3))

def test_is_uniform_false():
	assert dynamics.is_uniform(np.array([1, 1.1, 1.001])) == False

def test_is_uniform():
	assert dynamics.is_uniform(np.array([1, 1.05, 1.001]), epsilon=.1) == True