"""MPI_Comm_rank() implementation"""
import ctypes

import numba
import numpy as np

from ..common import _MPI_Comm_World_ptr, _MpiComm, libmpi
from ..utils import _mpi_addr

_MPI_Comm_rank = libmpi.MPI_Comm_rank
_MPI_Comm_rank.restype = ctypes.c_int
_MPI_Comm_rank.argtypes = [_MpiComm, ctypes.c_void_p]


@numba.njit()
def rank():
    """wrapper for MPI_Comm_rank()"""
    value = np.empty(1, dtype=np.intc)
    status = _MPI_Comm_rank(_mpi_addr(_MPI_Comm_World_ptr), value.ctypes.data)
    assert status == 0
    return value[0]
