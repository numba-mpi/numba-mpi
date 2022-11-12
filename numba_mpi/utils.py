"""helper functions used across API implementation"""
import numba
import numpy as np
from numba.core import cgutils, types

from .common import _MPI_DTYPES, _MpiComm, _MpiDatatype


def _mpi_dtype(arr):
    ptr = _get_dtype_numpy_to_mpi_ptr(arr)
    return _MpiDatatype.from_address(ptr)


@numba.extending.overload(_mpi_dtype)
def _mpi_dtype_njit(arr):
    mpi_dtype = _get_dtype_numba_to_mpi_ptr(arr)

    # pylint: disable-next=unused-argument
    def impl(arr):
        return numba.carray(
            # pylint: disable-next=no-value-for-parameter
            _address_as_void_pointer(mpi_dtype),
            shape=(1,),
            dtype=np.intp,
        )[0]

    return impl


def _get_dtype_numba_to_mpi_ptr(arr):
    for np_dtype, mpi_ptr in _MPI_DTYPES.items():
        if arr.dtype == numba.from_dtype(np_dtype):
            return mpi_ptr
    raise NotImplementedError(f"Type: {arr.dtype}")


def _get_dtype_numpy_to_mpi_ptr(arr):
    for np_dtype, mpi_ptr in _MPI_DTYPES.items():
        if np.can_cast(arr.dtype, np_dtype, casting="equiv"):
            return mpi_ptr
    raise NotImplementedError(f"Type: {arr.dtype}")


def _mpi_addr(ptr):
    return _MpiComm.from_address(ptr)


@numba.extending.overload(_mpi_addr)
def _mpi_addr_njit(ptr):
    def impl(ptr):
        return numba.carray(
            # pylint: disable-next=no-value-for-parameter
            _address_as_void_pointer(ptr),
            shape=(1,),
            dtype=np.intp,
        )[0]

    return impl


# https://stackoverflow.com/questions/61509903/how-to-pass-array-pointer-to-numba-function
@numba.extending.intrinsic
def _address_as_void_pointer(_, src):
    """returns a void pointer from a given memory address"""
    sig = types.voidptr(src)

    def codegen(__, builder, ___, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen
