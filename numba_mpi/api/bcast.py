"""MPI_Bcast() implementation"""
import ctypes

import numpy as np
from numba.core.extending import overload

from numba_mpi.common import _MPI_Comm_World_ptr, libmpi
from numba_mpi.utils import _mpi_addr, _mpi_dtype, _MpiComm, _MpiDatatype

_MPI_Bcast = libmpi.MPI_Bcast
_MPI_Bcast.restype = ctypes.c_int
_MPI_Bcast.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    _MpiDatatype,
    ctypes.c_int,
    _MpiComm,
]


def impl_ndarray(data, root):
    """MPI_Bcast implementation for ndarray datatype"""
    assert data.flags.c_contiguous

    status = _MPI_Bcast(
        data.ctypes.data,
        data.size,
        _mpi_dtype(data),
        root,
        _mpi_addr(_MPI_Comm_World_ptr),
    )
    return status


def impl_chararray(data, root):
    """MPI_Bcast implementation for chararray datatype"""
    assert data.flags.c_contiguous
    data = data.view(np.uint8)

    status = _MPI_Bcast(
        data.ctypes.data,
        data.size,
        _mpi_dtype(data),
        root,
        _mpi_addr(_MPI_Comm_World_ptr),
    )
    return status


def bcast(data, root):
    """wrapper for MPI_Bcast(). Returns integer status code (0 == MPI_SUCCESS)"""
    if data.dtype == np.dtype("S1"):
        return impl_chararray(data, root)
    if isinstance(data, np.ndarray):
        return impl_ndarray(data, root)

    raise TypeError(f"Unsupported type {data.__class__.__name__}")


@overload(bcast)
def __bcast_njit(data, root):
    """wrapper for MPI_Bcast(). Returns integer status code (0 == MPI_SUCCESS)"""

    if isinstance(data, str):
        data = np.ascontiguousarray(data)

        status = _MPI_Bcast(
            data.ctypes.data,
            data.size,
            _mpi_dtype(data),
            root,
            _mpi_addr(_MPI_Comm_World_ptr),
        )

        return status

    if isinstance(data, np.ndarray):
        return impl_ndarray(data, root)

    raise TypeError(f"Unsupported type {data.__class__.__name__}")
