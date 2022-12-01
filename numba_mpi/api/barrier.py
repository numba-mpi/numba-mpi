"""file contains MPI_Barrier() implementation"""
import ctypes

import numba

from numba_mpi.common import _MPI_Comm_World_ptr, libmpi
from numba_mpi.utils import _mpi_addr, _MpiComm

_MPI_Barrier = libmpi.MPI_Barrier
_MPI_Barrier.restype = ctypes.c_int
_MPI_Barrier.argtypes = [_MpiComm]


@numba.njit()
def barrier():
    """wrapper for MPI_Barrier(). Returns integer status code (0 == MPI_SUCCESS)"""
    return _MPI_Barrier(_mpi_addr(_MPI_Comm_World_ptr))
