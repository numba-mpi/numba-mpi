"""file contains MPI_Reduce() implementations"""

import ctypes
from numbers import Number

import numpy as np
from numba.core import types
from numba.extending import overload

from numba_mpi.common import _MPI_Comm_World_ptr, _MpiComm, _MpiDatatype, _MpiOp, libmpi
from numba_mpi.utils import _mpi_addr, _mpi_dtype

_MPI_Reduce = libmpi.MPI_Reduce
_MPI_Reduce.restype = ctypes.c_int
_MPI_Reduce.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    _MpiDatatype,
    _MpiOp,
    ctypes.c_int,
    _MpiComm,
]


def reduce(sendobj, recvobj, operator, root):  # pylint: disable=unused-argument
    """wrapper for MPI_Reduce
    Note that complex datatypes and user-defined functions are not properly supported.
    Returns integer status code (0 == MPI_SUCCESS)
    """
    if isinstance(sendobj, Number):
        # reduce a single number
        sendobj = np.array([sendobj])
        status = _MPI_Reduce(
            sendobj.ctypes.data,
            recvobj.ctypes.data,
            sendobj.size,
            _mpi_dtype(sendobj),
            _mpi_addr(operator),
            root,
            _mpi_addr(_MPI_Comm_World_ptr),
        )

    elif isinstance(sendobj, np.ndarray):
        # reduce an array
        sendobj = np.ascontiguousarray(sendobj)
        status = _MPI_Reduce(
            sendobj.ctypes.data,
            recvobj.ctypes.data,
            sendobj.size,
            _mpi_dtype(sendobj),
            _mpi_addr(operator),
            root,
            _mpi_addr(_MPI_Comm_World_ptr),
        )

    else:
        raise TypeError(f"Unsupported type {sendobj.__class__.__name__}")

    return status


@overload(reduce)
def ol_reduce(sendobj, recvobj, operator, root):  # pylint: disable=unused-argument
    """wrapper for MPI_Reduce
    Note that complex datatypes and user-defined functions are not properly supported.
    Returns integer status code (0 == MPI_SUCCESS)
    """
    if isinstance(sendobj, types.Number):
        # reduce a single number

        def impl(sendobj, recvobj, operator, root):
            sendobj = np.array([sendobj])

            status = _MPI_Reduce(
                sendobj.ctypes.data,
                recvobj.ctypes.data,
                sendobj.size,
                _mpi_dtype(sendobj),
                _mpi_addr(operator),
                root,
                _mpi_addr(_MPI_Comm_World_ptr),
            )

            # The following no-op prevents numba from too aggressive optimizations
            # This looks like a bug in numba (tested for version 0.55)
            sendobj[0]  # pylint: disable=pointless-statement

            return status

    elif isinstance(sendobj, types.Array):
        # reduce an array

        def impl(sendobj, recvobj, operator, root):
            sendobj = np.ascontiguousarray(sendobj)

            status = _MPI_Reduce(
                sendobj.ctypes.data,
                recvobj.ctypes.data,
                sendobj.size,
                _mpi_dtype(sendobj),
                _mpi_addr(operator),
                root,
                _mpi_addr(_MPI_Comm_World_ptr),
            )

            return status

    else:
        raise TypeError(f"Unsupported type {sendobj.__class__.__name__}")

    return impl
