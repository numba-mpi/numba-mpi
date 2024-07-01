"""MPI_Comm_rank() implementation"""

import ctypes

import numba
import numpy as np

from numba_mpi.common import _MPI_Comm_World_ptr, _MpiComm, libmpi
from numba_mpi.utils import _mpi_addr

_MPI_Comm_rank = libmpi.MPI_Comm_rank
_MPI_Comm_rank.restype = ctypes.c_int
_MPI_Comm_rank.argtypes = [_MpiComm, ctypes.c_void_p]


@numba.njit()
def rank():
    """wrapper for MPI_Comm_rank(), in case of failure returns 0"""
    value = np.empty(1, dtype=np.intc)
    status = _MPI_Comm_rank(_mpi_addr(_MPI_Comm_World_ptr), value.ctypes.data)
    if status != 0:
        value[0] = 0
    return value[0]
