"""MPI_Sendrecv() implementation"""

import ctypes

import numba
import numpy as np
from mpi4py.MPI import ANY_SOURCE, ANY_TAG

from numba_mpi.common import (
    _MPI_Comm_World_ptr,
    _MpiComm,
    _MpiDatatype,
    _MpiStatusPtr,
    create_status_buffer,
    libmpi,
)
from numba_mpi.utils import _mpi_addr, _mpi_dtype

_MPI_Sendrecv = libmpi.MPI_Sendrecv
_MPI_Sendrecv.restype = ctypes.c_int

_MPI_Sendrecv.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    _MpiDatatype,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    _MpiDatatype,
    ctypes.c_int,
    ctypes.c_int,
    _MpiComm,
    _MpiStatusPtr,
]

# pylint: disable=too-many-arguments,too-many-positional-arguments
@numba.njit
def sendrecv(senddata, dest, recvdata, source=ANY_SOURCE, sendtag=0, recvtag=ANY_TAG):
    """
    Wrapper for MPI_Sendrecv.
    
    Performs a simultaneous send and receive operation: sends `senddata`
    to `dest` and receives data into `recvdata` from `source`.
    
    Mimicking MPI semantics, default value for `sendtag` is set to zero,
    while `source` defaults to ANY_SOURCE and `recvtag` defaults to ANY_TAG.
    
    Returns MPI error code (0 == MPI_SUCCESS).
    """

    senddata = np.ascontiguousarray(senddata)

    status_buffer = create_status_buffer()

    buffer = (
        recvdata
        if recvdata.flags.c_contiguous
        else np.empty(recvdata.shape, recvdata.dtype)
    )

    status = _MPI_Sendrecv(
        senddata.ctypes.data,
        senddata.size,
        _mpi_dtype(senddata),
        dest,
        sendtag,
        buffer.ctypes.data,
        buffer.size,
        _mpi_dtype(recvdata),
        source,
        recvtag,
        _mpi_addr(_MPI_Comm_World_ptr),
        status_buffer.ctypes.data,
    )

    if not recvdata.flags.c_contiguous:
        recvdata[...] = buffer

    return status
