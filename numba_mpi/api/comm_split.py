# pylint: disable=missing-function-docstring
"""_MPI_Comm_split() implementation"""
import ctypes

from mpi4py import MPI

from numba_mpi.common import _MPI_Comm_World_ptr, libmpi
from numba_mpi.utils import _mpi_addr, _MpiComm

_MPI_Comm_split = libmpi.MPI_Comm_split
_MPI_Comm_split.restype = ctypes.c_int
_MPI_Comm_split.argtypes = [
    _MpiComm,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_MpiComm),
]


def comm_split(
    color: int, key: int, comm: int = _MPI_Comm_World_ptr
) -> tuple[int, int]:
    """
    :param int color: control of subset assignment
    :param int key: control of rank assignment
    :param int comm: communicator handle
    :return: Tuple of ints: (new communicator handle, status code)
    Note: Function not njittable due to MPI.Get_address() invocation - FIXME
    """
    newcomm = _MpiComm()
    status = _MPI_Comm_split(_mpi_addr(comm), color, key, ctypes.pointer(newcomm))

    return MPI.Get_address(newcomm), status
