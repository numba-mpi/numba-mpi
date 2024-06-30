# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

from mpi4py import MPI

import numba_mpi as mpi
from numba_mpi.common import _MPI_Comm_World_ptr


def test_split_barrier():
    rank = mpi.rank()
    comm, status = mpi.comm_split(_MPI_Comm_World_ptr, rank == 0, rank)

    assert status == 0
    if rank == 0:
        status = mpi.barrier(MPI.Get_address(comm))

        assert status == 0
