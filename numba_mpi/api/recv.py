"""MPI_Recv() implementation"""
import ctypes

import numba
import numpy as np

from ..common import _MPI_Comm_World_ptr, _MpiComm, _MpiDatatype, _MpiStatusPtr, libmpi
from ..utils import _mpi_addr, _mpi_dtype

_MPI_Recv = libmpi.MPI_Recv
_MPI_Recv.restype = ctypes.c_int
_MPI_Recv.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    _MpiDatatype,
    ctypes.c_int,
    ctypes.c_int,
    _MpiComm,
    _MpiStatusPtr,
]


@numba.njit()
def recv(data, source, tag):
    """file containing wrapper for MPI_Recv (writes data directly if `data` is contiguous, otherwise
    allocates a buffer and later copies the data into non-contiguous `data` array).
    Returns integer status code (0 == MPI_SUCCESS)"""
    status = np.empty(5, dtype=np.intc)

    buffer = (
        data
        if data.flags.c_contiguous
        else np.empty(
            data.shape, data.dtype
        )  # np.empty_like(data, order='C') fails with Numba
    )

    status = _MPI_Recv(
        buffer.ctypes.data,
        buffer.size,
        _mpi_dtype(data),
        source,
        tag,
        _mpi_addr(_MPI_Comm_World_ptr),
        status.ctypes.data,
    )

    if not data.flags.c_contiguous:
        data[...] = buffer

    return status