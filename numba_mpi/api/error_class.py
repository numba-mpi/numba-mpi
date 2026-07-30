"""MPI_Error_class() wrapper"""

import ctypes

import numba
import numpy as np

from numba_mpi.common import libmpi

_MPI_Error_class = libmpi.MPI_Error_class
_MPI_Error_class.restype = ctypes.c_int
_MPI_Error_class.argtypes = [ctypes.c_int, ctypes.c_void_p]


@numba.njit()
def error_class(error_code):
    """
    Wrapper for MPI_Error_class()
    """
    value = np.empty(1, dtype=np.intc)
    status = _MPI_Error_class(error_code, value.ctypes.data)

    if status != 0:
        value[0] = 0

    return value[0]
