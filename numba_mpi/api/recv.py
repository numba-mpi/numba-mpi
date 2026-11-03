"""MPI_Recv() implementation"""

import ctypes

import numba
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE, ANY_TAG, BYTE, COMM_WORLD, DATATYPE_NULL, _addressof

from numba_mpi import custom_dtypes
from numba_mpi.common import (
    _MPI_DTYPES,
    ZERO_SIZE_BYTE_ARRAY,
    _MPI_Comm_World_ptr,
    _MpiDatatype,
    _MpiStatusPtr,
    create_status_buffer,
    libmpi,
    send_recv_args,
)
from numba_mpi.utils import _mpi_addr, _mpi_dtype

_MPI_Recv = libmpi.MPI_Recv
_MPI_Recv.restype = ctypes.c_int
_MPI_Recv.argtypes = send_recv_args + [
    _MpiStatusPtr,
]


def __recv(data, source=ANY_SOURCE, tag=ANY_TAG):
    """file containing wrapper for MPI_Recv (writes data directly if `data` is contiguous, otherwise
    allocates a buffer and later copies the data into non-contiguous `data` array).
    Returns integer status code (0 == MPI_SUCCESS)"""
    status_buffer = create_status_buffer()

    buffer = (
        data
        if data.flags.c_contiguous
        else np.empty(
            data.shape, data.dtype
        )  # np.empty_like(data, order='C') fails with Numba
    )

    datasize = buffer.size
    dtype = _mpi_dtype(buffer)
    if dtype == _mpi_dtype(ZERO_SIZE_BYTE_ARRAY):
        datasize = buffer.nbytes

    status = _MPI_Recv(
        buffer.ctypes.data,
        datasize,
        dtype,
        source,
        tag,
        _mpi_addr(_MPI_Comm_World_ptr),
        status_buffer.ctypes.data,
    )

    if not data.flags.c_contiguous:
        data[...] = buffer

    return status


def recv(data, source=ANY_SOURCE, tag=ANY_TAG):
    if data.dtype.isbuiltin:
        return __recv(data, source, tag)
    else:
        try:
            COMM_WORLD.Recv(
                [
                    data,
                    data.size,
                    custom_dtypes._dtype_to_mpitype(data.dtype),
                ],
                source,
                tag,
            )
            return MPI.SUCCESS
        except MPI.Exception as exception:
            return exception.error_code


@numba.extending.overload(recv)
def recv_numba(data, source=ANY_SOURCE, tag=ANY_TAG):
    return __recv
