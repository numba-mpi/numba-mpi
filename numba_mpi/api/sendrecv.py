"""MPI_Sendrecv() implementation"""

import ctypes

import numba
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE, ANY_TAG, COMM_WORLD

from numba_mpi.api.custom_dtypes import _dtype_to_mpitype
from numba_mpi.common import (
    ZERO_SIZE_BYTE_ARRAY,
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


def __sendrecv(senddata, dest, recvdata, source=ANY_SOURCE, sendtag=0, recvtag=ANY_TAG):
    senddata = np.ascontiguousarray(senddata)

    status_buffer = create_status_buffer()

    buffer = (
        recvdata
        if recvdata.flags.c_contiguous
        else np.empty(recvdata.shape, recvdata.dtype)
    )

    send_ds = senddata.size
    recv_ds = buffer.size

    send_dt = _mpi_dtype(senddata)
    recv_dt = _mpi_dtype(buffer)

    if send_dt == _mpi_dtype(ZERO_SIZE_BYTE_ARRAY):
        send_ds = senddata.nbytes

    if recv_dt == _mpi_dtype(ZERO_SIZE_BYTE_ARRAY):
        recv_ds = buffer.nbytes

    status = _MPI_Sendrecv(
        senddata.ctypes.data,
        send_ds,
        send_dt,
        dest,
        sendtag,
        buffer.ctypes.data,
        recv_ds,
        recv_dt,
        source,
        recvtag,
        _mpi_addr(_MPI_Comm_World_ptr),
        status_buffer.ctypes.data,
    )

    if not recvdata.flags.c_contiguous:
        recvdata[...] = buffer
    return status


def sendrecv(senddata, dest, recvdata, source=ANY_SOURCE, sendtag=0, recvtag=ANY_TAG):
    """
    Wrapper for MPI_Sendrecv. Returns integer status code (`0` indicates `MPI_SUCCESS`).

    Performs a simultaneous send and receive operation: sends `senddata`
    to `dest` and receives data into `recvdata` from `source`.

    Mimicking MPI semantics, default value for `sendtag` is set to zero,
    while `source` defaults to ANY_SOURCE and `recvtag` defaults to ANY_TAG.

    Both built-in MPI datatypes, as well as custom datatypes defined in
    user code are supported; transfers of custom-type data are done
    byte-by-byte in JIT-compiled code, without datatype consistency checks,
    while in pure-Python calling scopes, the mpi4py's `Sendrecv()` function is
    called in order to perform type checks.
    """

    if senddata.dtype.isbuiltin and recvdata.dtype.isbuiltin:
        return __sendrecv(senddata, dest, recvdata, source, sendtag, recvtag)

    try:
        COMM_WORLD.Sendrecv(
            [senddata, senddata.size, _dtype_to_mpitype(senddata.dtype)],
            dest,
            sendtag=sendtag,
            recvbuf=[recvdata, recvdata.size, _dtype_to_mpitype(recvdata.dtype)],
            source=source,
            recvtag=recvtag,
        )
        return MPI.SUCCESS
    except MPI.Exception as exception:
        return exception.error_code


@numba.extending.overload(sendrecv)
def sendrecv_numba(
    senddata, dest, recvdata, source=ANY_SOURCE, sendtag=0, recvtag=ANY_TAG
):
    return __sendrecv
