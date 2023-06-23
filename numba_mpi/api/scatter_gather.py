"""MPI_Scatter() implementation"""
import ctypes

import numba

from numba_mpi.common import _MPI_Comm_World_ptr, libmpi
from numba_mpi.utils import _mpi_addr, _mpi_dtype, _MpiComm, _MpiDatatype

_MPI_Scatter = libmpi.MPI_Scatter
_MPI_Scatter.restype = ctypes.c_int
_MPI_Scatter.argtypes = [
    # pylint:disable=duplicate-code
    ctypes.c_void_p,  # send_data
    ctypes.c_int,  # send_count
    _MpiDatatype,  # send_data_type
    ctypes.c_void_p,  # recv_data
    ctypes.c_int,  # recv_count
    _MpiDatatype,  # recv_data_type
    ctypes.c_int,  # root
    _MpiComm,  # communicator
]

_MPI_Gather = libmpi.MPI_Gather
_MPI_Gather.restype = ctypes.c_int
_MPI_Gather.argtypes = _MPI_Scatter.argtypes


@numba.njit()
def scatter(send_data, recv_data, count, root):
    """wrapper for MPI_Scatter(). Returns integer status code (0 == MPI_SUCCESS)"""
    assert send_data.flags.c_contiguous  # TODO #60
    assert recv_data.flags.c_contiguous  # TODO #60

    status = _MPI_Scatter(
        send_data.ctypes.data,
        count,
        _mpi_dtype(send_data),
        recv_data.ctypes.data,
        recv_data.size,
        _mpi_dtype(recv_data),
        root,
        _mpi_addr(_MPI_Comm_World_ptr),
    )
    return status


@numba.njit()
def gather(send_data, recv_data, count, root):
    """wrapper for MPI_Gather(). Returns integer status code (0 == MPI_SUCCESS)"""
    assert send_data.flags.c_contiguous  # TODO #60
    assert recv_data.flags.c_contiguous  # TODO #60

    status = _MPI_Gather(
        send_data.ctypes.data,
        send_data.size,
        _mpi_dtype(send_data),
        recv_data.ctypes.data,
        count,
        _mpi_dtype(recv_data),
        root,
        _mpi_addr(_MPI_Comm_World_ptr),
    )
    return status
