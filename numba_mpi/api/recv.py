"""MPI_Recv() implementation"""
import ctypes

import numba
import numpy as np
from mpi4py.MPI import ANY_SOURCE, ANY_TAG

from numba_mpi.common import _MPI_Comm_World_ptr, _MpiStatusPtr, libmpi, send_recv_args
from numba_mpi.utils import _mpi_addr, _mpi_dtype

_MPI_Recv = libmpi.MPI_Recv
_MPI_Recv.restype = ctypes.c_int
_MPI_Recv.argtypes = send_recv_args + [
    _MpiStatusPtr,
]


@numba.njit()
def recv(data, source=ANY_SOURCE, tag=ANY_TAG):
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
