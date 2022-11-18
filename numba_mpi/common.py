"""variables used across API implementation"""
import ctypes
from ctypes.util import find_library

import numpy as np
from mpi4py import MPI

# pylint: disable=protected-access
_MPI_Comm_World_ptr = MPI._addressof(MPI.COMM_WORLD)

_MPI_DTYPES = {
    np.dtype("uint8"): MPI._addressof(MPI.CHAR),
    np.dtype("int32"): MPI._addressof(MPI.INT32_T),
    np.dtype("int64"): MPI._addressof(MPI.INT64_T),
    np.dtype("float"): MPI._addressof(MPI.FLOAT),
    np.dtype("double"): MPI._addressof(MPI.DOUBLE),
    np.dtype("complex64"): MPI._addressof(MPI.C_FLOAT_COMPLEX),
    np.dtype("complex128"): MPI._addressof(MPI.C_DOUBLE_COMPLEX),
}

if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    _MpiComm = ctypes.c_int
else:
    _MpiComm = ctypes.c_void_p


if MPI._sizeof(MPI.Datatype) == ctypes.sizeof(ctypes.c_int):
    _MpiDatatype = ctypes.c_int
    _MpiOp = ctypes.c_int
else:
    _MpiDatatype = ctypes.c_void_p
    _MpiOp = ctypes.c_void_p

# pylint: enable=protected-access
_MpiStatusPtr = ctypes.c_void_p

for name in ("mpi", "msmpi", "impi"):
    LIB = find_library(name)
    if LIB is not None:
        break

if LIB is None:
    raise RuntimeError("no MPI library found")

libmpi = ctypes.CDLL(LIB)

send_recv_args = [
    ctypes.c_void_p,
    ctypes.c_int,
    _MpiDatatype,
    ctypes.c_int,
    ctypes.c_int,
    _MpiComm,
]
