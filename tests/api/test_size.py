# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import pytest
from mpi4py.MPI import COMM_WORLD

import numba_mpi as mpi


@pytest.mark.parametrize("sut", [mpi.size, mpi.size.py_func])
def test_size(sut):
    size = sut()
    assert size == COMM_WORLD.Get_size()
