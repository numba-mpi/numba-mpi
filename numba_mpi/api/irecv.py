"""MPI_Irecv() implementation"""
import ctypes

import numba
import numpy as np
from mpi4py.MPI import ANY_SOURCE, ANY_TAG

from numba_mpi.common import _MPI_Comm_World_ptr, libmpi, send_recv_async_args
from numba_mpi.utils import _mpi_addr, _mpi_dtype

_MPI_Irecv = libmpi.MPI_Irecv
_MPI_Irecv.restype = ctypes.c_int
_MPI_Irecv.argtypes = send_recv_async_args


@numba.njit()
def irecv(data, source=ANY_SOURCE, tag=ANY_TAG):
    """Wrapper for MPI_Irecv (only handles contiguous arrays, at least for now).
    If successful (i.e. result is MPI_SUCCESS), returns c-style pointer to valid
    MPI_Request handle that may be used with appropriate wait and test functions."""

    assert data.flags.c_contiguous

    request = np.empty(1, np.uintp)

    status = _MPI_Irecv(
        data.ctypes.data,
        data.size,
        _mpi_dtype(data),
        source,
        tag,
        _mpi_addr(_MPI_Comm_World_ptr),
        request.ctypes.data,
    )

    assert status == 0

    return request[0]
