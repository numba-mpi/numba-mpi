# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

import numba
import pytest

import numba_mpi as mpi
from numba_mpi.common import _MPI_Comm_World_ptr


@numba.njit()
def jit_split(color, key, comm=_MPI_Comm_World_ptr):
    return mpi.comm_split(color, key, comm)


@pytest.mark.parametrize("split", (mpi.comm_split,))
def test_split_barrier(split):
    rank = mpi.rank()
    comm, status = split(rank == 0, rank)

    assert status == 0
    if rank == 0:
        status = mpi.barrier(comm)

        assert status == 0
