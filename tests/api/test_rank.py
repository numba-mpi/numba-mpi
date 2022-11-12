# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import pytest
from mpi4py.MPI import COMM_WORLD

import numba_mpi as mpi


@pytest.mark.parametrize("sut", [mpi.rank, mpi.rank.py_func])
def test_rank(sut):
    rank = sut()
    assert rank == COMM_WORLD.Get_rank()
