"""MPI_Send() implementation"""
import ctypes

import numba
import numpy as np

from numba_mpi.common import _MPI_Comm_World_ptr, libmpi, send_recv_args
from numba_mpi.utils import _mpi_addr, _mpi_dtype

_MPI_Send = libmpi.MPI_Send
_MPI_Send.restype = ctypes.c_int
_MPI_Send.argtypes = send_recv_args


@numba.njit
def send(data, dest, tag=0):
    """Wrapper for MPI_Send. Returns integer status code (0 == MPI_SUCCESS).
    Mimicking `mpi4py`, default value for `tag` is set to zero.
    """
    data = np.ascontiguousarray(data)
    status = _MPI_Send(
        data.ctypes.data,
        data.size,
        _mpi_dtype(data),
        dest,
        tag,
        _mpi_addr(_MPI_Comm_World_ptr),
    )

    # The following no-op prevents numba from too aggressive optimizations
    # This looks like a bug in numba (tested for version 0.55)
    data[0]  # pylint: disable=pointless-statement

    return status
