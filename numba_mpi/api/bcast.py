"""MPI_Bcast() implementation"""
import ctypes

import numba
import numpy as np

from numba_mpi.common import _MPI_Comm_World_ptr, libmpi
from numba_mpi.utils import _mpi_addr, _mpi_dtype, _MpiComm, _MpiDatatype

_MPI_Bcast = libmpi.MPI_Bcast
_MPI_Bcast.restype = ctypes.c_int
_MPI_Bcast.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    _MpiDatatype,
    ctypes.c_int,
    _MpiComm,
]


@numba.njit
def bcast(data, root):
    """wrapper for MPI_Bcast(). Returns integer status code (0 == MPI_SUCCESS)"""
    data = np.ascontiguousarray(data)

    status = _MPI_Bcast(
        data.ctypes.data,
        data.size,
        _mpi_dtype(data),
        root,
        _mpi_addr(_MPI_Comm_World_ptr),
    )

    return status
