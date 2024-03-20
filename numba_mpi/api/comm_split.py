# pylint: disable=missing-function-docstring
"""_MPI_Comm_split() implementation"""
import ctypes

from numba_mpi.common import libmpi
from numba_mpi.utils import _mpi_addr, _MpiComm

_MPI_Comm_split = libmpi.MPI_Comm_split
_MPI_Comm_split.restype = ctypes.c_int
_MPI_Comm_split.argtypes = [
    _MpiComm,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_MpiComm),
]


def comm_split(comm, color, key):
    newcomm = _MpiComm()
    status = _MPI_Comm_split(_mpi_addr(comm), color, key, ctypes.pointer(newcomm))

    return newcomm, status
