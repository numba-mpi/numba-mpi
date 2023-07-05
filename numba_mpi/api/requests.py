""" Utilities for handling MPI_Requests with respective Wait and Test function
    wrappers.
"""
import ctypes

import numba
import numpy as np

from numba_mpi.common import _MpiRequestPtr, _MpiStatusPtr, create_status_buffer, libmpi

# helper function to allocate numpy array of request handles


@numba.njit
def create_requests_array(count=1):
    """Helper function for creating numpy array storing pointers to MPI_Request handles."""
    return np.empty(count, dtype=np.uintp)


# Wait* functions

_MPI_Wait = libmpi.MPI_Wait
_MPI_Wait.restype = ctypes.c_int
_MPI_Wait.argtypes = [_MpiRequestPtr, _MpiStatusPtr]


@numba.njit
def wait(request):
    """Wrapper for MPI_Wait. Returns integer status code (0 == MPI_SUCCESS).
    Status is currently not handled. Requires 'request' parameter to be a
    c-style pointer to MPI_Request (such as returned by 'isend'/'irecv').
    """

    status_buffer = create_status_buffer()
    status = _MPI_Wait(request.ctypes.data, status_buffer.ctypes.data)

    return status


_MPI_Waitall = libmpi.MPI_Waitall
_MPI_Waitall.restype = ctypes.c_int
_MPI_Waitall.argtypes = [ctypes.c_int, _MpiRequestPtr, _MpiStatusPtr]


@numba.njit
def waitall(requests):
    """Wrapper for MPI_Waitall. Returns integer status code (0 == MPI_SUCCESS).
    Status is currently not handled. Requires 'requests' parameter to be an
    array of c-style pointers to MPI_Requests (e.g. created by
    'create_requests_array' and popuated by 'isend'/'irecv').
    """

    status_buffer = create_status_buffer(requests.size)
    status = _MPI_Waitall(
        requests.size, requests.ctypes.data, status_buffer.ctypes.data
    )

    return status


_MPI_Waitany = libmpi.MPI_Waitany
_MPI_Waitany.restype = ctypes.c_int
_MPI_Waitany.argtypes = [ctypes.c_int, _MpiRequestPtr, ctypes.c_void_p, _MpiStatusPtr]


@numba.njit
def waitany(requests):
    """Wrapper for MPI_Waitany. Returns integer which is non-negative when call
    succeeded and is the index of request that was completed, otherwise -
    if result is negative - error occurred and its code may be obtained by
    negating returned integer. Status is currently not handled.
    Requires 'requests' parameter to be an array of c-style pointers to
    MPI_Requests (e.g. created by 'create_requests_array' and popuated by
    'isend'/'irecv').
    """

    status_buffer = create_status_buffer()
    index = np.empty(1, dtype=np.intc)
    status = _MPI_Waitany(
        requests.size,
        requests.ctypes.data,
        index.ctypes.data,
        status_buffer.ctypes.data,
    )

    if status > 0:
        return -status
    return index[0]


# - Test* functions

_MPI_Test = libmpi.MPI_Test
_MPI_Test.restype = ctypes.c_int
_MPI_Test.argtypes = [_MpiRequestPtr, ctypes.c_void_p, _MpiStatusPtr]


@numba.njit
def test(request):
    """Wrapper for MPI_Test. Returns boolean flag indicating whether given
    request is completed. Status is currently not handled. Requires
    'request' parameter to be a c-style pointer to MPI_Request
    (such as returned by 'isend'/'irecv').
    """

    status_buffer = create_status_buffer()
    flag = np.empty(1, dtype=np.intc)
    status = _MPI_Test(request, flag.ctypes.data, status_buffer.ctypes.data)

    assert status == 0

    return flag[0] != 0


_MPI_Testall = libmpi.MPI_Testall
_MPI_Testall.restype = ctypes.c_int
_MPI_Testall.argtypes = [ctypes.c_int, _MpiRequestPtr, ctypes.c_void_p, _MpiStatusPtr]


@numba.njit
def testall(requests):
    """Wrapper for MPI_Testall. Returns boolean flag indicating whether all
    requests in question are completed. Status is currently not handled.
    Requires 'requests' parameter to be an array of c-style pointers to
    MPI_Requests (e.g. created by 'create_requests_array' and popuated by
    'isend'/'irecv').
    """

    status_buffer = create_status_buffer(requests.size)
    flag = np.empty(1, dtype=np.intc)
    status = _MPI_Testall(
        requests.size, requests.ctypes.data, flag.ctypes.data, status_buffer.ctypes.data
    )

    assert status == 0

    return flag[0] != 0


_MPI_Testall = libmpi.MPI_Testany
_MPI_Testall.restype = ctypes.c_int
_MPI_Testall.argtypes = [
    ctypes.c_int,
    _MpiRequestPtr,
    ctypes.c_void_p,
    ctypes.c_void_p,
    _MpiStatusPtr,
]


class TestAnyResult:
    """Helper class for storing results of calls to MPI_Testany wrapper."""

    def __init__(self, flag, index):
        """Initializes instance from returned flag and indx parameters."""
        self._value = index if flag else -1

    def __bool__(self):
        """Returns true when flag parameter was true."""
        return self._value >= 0

    def index(self):
        """Returns index of request that is ensured to be completed. Valid if
        returned flag value was true.
        """
        return self._value


@numba.njit
def testany(requests):
    """Wrapper for MPI_Testany. Returns simple helper class that is truthy if
    any of requests in question was completed. Its 'index()' method may
    be used to obtain index of request that is guaranteed to be completed.
    Status is currently not handled. Requires 'requests' parameter to be an
    array of c-style pointers to MPI_Requests (e.g. created by
    'create_requests_array' and popuated by 'isend'/'irecv').
    """

    status_buffer = create_status_buffer()
    flag = np.empty(1, dtype=np.intc)
    index = np.empty(1, dtype=np.intc)
    status = _MPI_Waitany(
        requests.size,
        requests.ctypes.data,
        index.ctypes.data,
        flag.ctypes.data,
        status_buffer.ctypes.data,
    )

    assert status == 0

    return TestAnyResult(flag, index)
