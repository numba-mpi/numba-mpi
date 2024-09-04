# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

import numba
import pytest

import numba_mpi as mpi
from numba_mpi.common import _MPI_Comm_World_ptr


@numba.njit()
def jit_split(color, key, comm=_MPI_Comm_World_ptr):
    return mpi.comm_split(color, key, comm)


@pytest.mark.parametrize(
    "split",
    (
        mpi.comm_split,
        pytest.param(jit_split, marks=pytest.mark.xfail(strict=True)),  # FIXME
    ),
)
def test_split_barrier_with_default_comm(split):
    rank = mpi.rank()
    comm, status = split(rank == 0, rank)

    assert status == mpi.SUCCESS
    if rank == 0:
        status = mpi.barrier(comm)

    assert status == mpi.SUCCESS


@pytest.mark.parametrize(
    "split",
    (
        mpi.comm_split,
        pytest.param(jit_split, marks=pytest.mark.xfail(strict=True)),  # FIXME
    ),
)
def test_split_splitted_comm(split):
    rank = mpi.rank()

    comm, status = split(rank == 0, rank)
    assert status == mpi.SUCCESS

    comm, status = split(rank == 0, rank, comm)
    assert status == mpi.SUCCESS

    comm, status = split(rank == 0, rank, comm)
    assert status == mpi.SUCCESS
