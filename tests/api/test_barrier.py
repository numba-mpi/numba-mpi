import numba_mpi
from tests.common import MPI_SUCCESS


def test_barrier():
    status = numba_mpi.barrier()

    assert status == MPI_SUCCESS
