"""file contains MPI_Allreduce() implementations"""
import ctypes
from numbers import Number

import numba
import numpy as np
from numba.core import types

from numba_mpi.api.operator import Operator
from numba_mpi.common import _MPI_Comm_World_ptr, _MpiComm, _MpiDatatype, _MpiOp, libmpi
from numba_mpi.utils import _mpi_addr, _mpi_dtype

_MPI_Allreduce = libmpi.MPI_Allreduce
_MPI_Allreduce.restype = ctypes.c_int
_MPI_Allreduce.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    _MpiDatatype,
    _MpiOp,
    _MpiComm,
]


@numba.generated_jit(nopython=True)
def allreduce(
    sendobj, recvobj, operator=Operator.SUM
):  # pylint: disable=unused-argument
    """wrapper for MPI_Allreduce
    Note that complex datatypes and user-defined functions are not properly supported.
    Returns integer status code (0 == MPI_SUCCESS)
    """
    if isinstance(sendobj, (types.Number, Number)):

        def impl(sendobj, recvobj, operator=Operator.SUM):
            sendobj = np.array([sendobj])

            status = _MPI_Allreduce(
                sendobj.ctypes.data,
                recvobj.ctypes.data,
                sendobj.size,
                _mpi_dtype(sendobj),
                _mpi_addr(operator),
                _mpi_addr(_MPI_Comm_World_ptr),
            )

            # The following no-op prevents numba from too aggressive optimizations
            # This looks like a bug in numba (tested for version 0.55)
            sendobj[0]  # pylint: disable=pointless-statement

            return status

    elif isinstance(sendobj, (types.Array, np.ndarray)):

        def impl(sendobj, recvobj, operator=Operator.SUM):
            sendobj = np.ascontiguousarray(sendobj)

            status = _MPI_Allreduce(
                sendobj.ctypes.data,
                recvobj.ctypes.data,
                sendobj.size,
                _mpi_dtype(sendobj),
                _mpi_addr(operator),
                _mpi_addr(_MPI_Comm_World_ptr),
            )

            return status

    else:
        raise TypeError(f"Unsupported type {sendobj.__class__.__name__}")

    return impl
