import ctypes

from numba_mpi.common import _MPI_Comm_World_ptr, libmpi
from numba_mpi.utils import _mpi_addr, _MpiComm

_MPI_Barrier = libmpi.MPI_Barrier
_MPI_Barrier.restype = ctypes.c_int
_MPI_Barrier.argtypes = [_MpiComm]


def barrier():
    return _MPI_Barrier(_mpi_addr(_MPI_Comm_World_ptr))
