# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numpy as np


def get_random_array(shape, data_type):
    """helper function creating the same random array in each process"""
    rng = np.random.default_rng(0)
    if np.issubdtype(data_type, np.complexfloating):
        return rng.random(shape) + rng.random(shape) * 1j
    if np.issubdtype(data_type, np.integer):
        return rng.integers(0, 10, size=shape)
    return rng.random(shape)
