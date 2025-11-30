"""file contains MPI_Query_thread() implementation"""

import ctypes

import numba
import numpy as np

from numba_mpi.common import libmpi

_MPI_Query_thread = libmpi.MPI_Query_thread
_MPI_Query_thread.restype = ctypes.c_int
_MPI_Query_thread.argtypes = [ctypes.c_void_p]


@numba.njit()
def query_thread():
    """wrapper for MPI_Query_thread()"""
    provided = np.empty(1, dtype=np.intc)
    _ = _MPI_Query_thread(provided.ctypes.data)
    return provided[0]
