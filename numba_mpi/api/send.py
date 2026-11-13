"""MPI_Send() implementation"""

import ctypes

import numba
import numpy as np
from mpi4py import MPI

from numba_mpi.api.custom_dtypes import _dtype_to_mpitype
from numba_mpi.common import (
    ZERO_SIZE_BYTE_ARRAY,
    _MPI_Comm_World_ptr,
    libmpi,
    send_recv_args,
)
from numba_mpi.utils import _mpi_addr, _mpi_dtype

_MPI_Send = libmpi.MPI_Send
_MPI_Send.restype = ctypes.c_int
_MPI_Send.argtypes = send_recv_args


def __send(data, dest, tag=0):

    data = np.ascontiguousarray(data)
    datasize = data.size

    dtype = _mpi_dtype(data)

    if dtype == _mpi_dtype(ZERO_SIZE_BYTE_ARRAY):
        datasize = data.nbytes
        # TODO: test for both aligned and non-aligned types

    status = _MPI_Send(
        data.ctypes.data,
        datasize,
        dtype,
        dest,
        tag,
        _mpi_addr(_MPI_Comm_World_ptr),
    )

    # The following no-op prevents numba from too aggressive optimizations
    # This looks like a bug in numba (tested for version 0.55)
    data[0]  # pylint: disable=pointless-statement

    return status


def send(data, dest, tag=0):
    """Wrapper for MPI_Send. Returns integer status code (`0` indicates `MPI_SUCCESS`).
    The default value for `tag` is set to zero as in mpi4py.
    Both built-in MPI datatypes, as well as custom datatypes defined in
    user code are supported; transfers of custom-type data are done
    byte-by-byte in JIT-compiled code, without datatype consistency checks,
    while in pure-Python calling scopes, the mpi4py's `Send()` function is
    called in order to perform type checks.
    """
    if data.dtype.isbuiltin:
        return __send(data, dest, tag)

    try:
        MPI.COMM_WORLD.Send(
            [
                data,
                data.size,
                _dtype_to_mpitype(data.dtype),
            ],
            dest,
            tag,
        )
        return MPI.SUCCESS
    except MPI.Exception as exception:
        return exception.error_code


@numba.extending.overload(send)
def send_numba(data, dest, tag=0):
    return __send
