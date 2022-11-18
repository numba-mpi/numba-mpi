# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numba_mpi
from tests.common import MPI_SUCCESS


def test_barrier():
    status = numba_mpi.barrier()

    assert status == MPI_SUCCESS
