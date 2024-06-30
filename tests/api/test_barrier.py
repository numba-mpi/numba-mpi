# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numba
import pytest

import numba_mpi
from numba_mpi.common import _MPI_Comm_World_ptr
from tests.common import MPI_SUCCESS


@numba.njit()
def jit_barrier(comm_ptr):
    return numba_mpi.barrier(comm_ptr)


@pytest.mark.parametrize("barrier", (jit_barrier.py_func, jit_barrier))
def test_barrier(barrier):
    status = barrier(_MPI_Comm_World_ptr)

    assert status == MPI_SUCCESS
