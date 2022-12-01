# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numba
import pytest

import numba_mpi
from tests.common import MPI_SUCCESS


@numba.njit()
def jit_barrier():
    return numba_mpi.barrier()


@pytest.mark.parametrize("barrier", (jit_barrier.py_func, jit_barrier))
def test_barrier(barrier):
    status = barrier()

    assert status == MPI_SUCCESS
