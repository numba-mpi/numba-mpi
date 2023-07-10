# pylint: disable=duplicate-code

"""MPI_Isend() implementation"""
import ctypes

import numba
import numpy as np

from numba_mpi.api.requests import _allocate_numpy_array_of_request_handles
from numba_mpi.common import _MPI_Comm_World_ptr, libmpi, send_recv_async_args
from numba_mpi.utils import _mpi_addr, _mpi_dtype

_MPI_Isend = libmpi.MPI_Isend
_MPI_Isend.restype = ctypes.c_int
_MPI_Isend.argtypes = send_recv_async_args


@numba.njit
def isend(data, dest, tag=0):
    """Wrapper for MPI_Send. If successful (i.e. result is MPI_SUCCESS),
    returns c-style pointer to valid MPI_Request handle that may be used
    with appropriate wait and test functions. Mimicking `mpi4py`, default
    value for `tag` is set to zero.
    """
    data = np.ascontiguousarray(data)

    request_buffer = _allocate_numpy_array_of_request_handles()

    status = _MPI_Isend(
        data.ctypes.data,
        data.size,
        _mpi_dtype(data),
        dest,
        tag,
        _mpi_addr(_MPI_Comm_World_ptr),
        request_buffer.ctypes.data,
    )

    return status, request_buffer
