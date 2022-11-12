"""file contains MPI_Initialized() implementation"""
import ctypes

import numba
import numpy as np

from numba_mpi.common import libmpi

_MPI_Initialized = libmpi.MPI_Initialized
_MPI_Initialized.restype = ctypes.c_int
_MPI_Initialized.argtypes = [ctypes.c_void_p]


@numba.njit()
def initialized():
    """wrapper for MPI_Initialized()"""
    flag = np.empty((1,), dtype=np.intc)
    status = _MPI_Initialized(flag.ctypes.data)
    assert status == 0
    return bool(flag[0])
