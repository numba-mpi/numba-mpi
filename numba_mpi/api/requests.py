"""Utilities for handling MPI_Requests with respective Wait and Test function
wrappers.
"""

import ctypes

import numba
import numpy as np
from numba.core import types
from numba.core.extending import overload

from numba_mpi.common import (
    RequestType,
    _MpiRequestPtr,
    _MpiStatusPtr,
    create_status_buffer,
    libmpi,
)


@numba.njit
def _allocate_numpy_array_of_request_handles(count=1):
    """Helper function for creating numpy array storing pointers to MPI_Request handles."""
    return np.empty(count, dtype=RequestType)


_MPI_Wait = libmpi.MPI_Wait
_MPI_Wait.restype = ctypes.c_int
_MPI_Wait.argtypes = [_MpiRequestPtr, _MpiStatusPtr]


@numba.njit
def wait(request):
    """Wrapper for MPI_Wait. Returns integer status code (0 == MPI_SUCCESS).
    Status is currently not handled. Requires 'request' parameter to be a
    c-style pointer to MPI_Request (such as returned by 'isend'/'irecv').

    Uninitialized contents of 'request' (e.g., from numpy.empty()) may
    cause invalid pointer dereference and segmentation faults.
    """

    status_buffer = create_status_buffer()
    status = _MPI_Wait(request.ctypes.data, status_buffer.ctypes.data)

    return status


_MPI_Waitall = libmpi.MPI_Waitall
_MPI_Waitall.restype = ctypes.c_int
_MPI_Waitall.argtypes = [ctypes.c_int, _MpiRequestPtr, _MpiStatusPtr]


@numba.njit
def _waitall_array_impl(requests):
    """MPI_Waitall implementation for contiguous numpy arrays of MPI_Requests"""

    status_buffer = create_status_buffer(requests.size)

    status = _MPI_Waitall(
        requests.size, requests.ctypes.data, status_buffer.ctypes.data
    )

    return status


def waitall(requests):
    """Wrapper for MPI_Waitall. Returns integer status code (0 == MPI_SUCCESS).
    Status is currently not handled. Requires 'requests' parameter to be an
    array or tuple of MPI_Request objects.

    Uninitialized contents of 'requests' (e.g., from numpy.empty()) may
    cause invalid pointer dereference and segmentation faults.
    """
    if isinstance(requests, np.ndarray):
        return _waitall_array_impl(requests)

    if isinstance(requests, (list, tuple)):
        request_buffer = np.hstack(requests)
        return _waitall_array_impl(request_buffer)

    raise TypeError("Invalid type for array of MPI_Request objects")


@overload(waitall)
def _waitall_impl(requests):
    """List of overloads for MPI_Waitall implementation"""

    if isinstance(requests, types.Array):

        def impl(requests):
            return _waitall_array_impl(requests)

    elif isinstance(requests, types.UniTuple):

        def impl(requests):
            req_buffer = np.hstack(requests)
            return _waitall_array_impl(req_buffer)

    else:
        raise TypeError("Invalid type for array of MPI_Request objects")

    return impl


_MPI_Waitany = libmpi.MPI_Waitany
_MPI_Waitany.restype = ctypes.c_int
_MPI_Waitany.argtypes = [ctypes.c_int, _MpiRequestPtr, ctypes.c_void_p, _MpiStatusPtr]


@numba.njit
def _waitany_array_impl(requests):
    """MPI_Waitany implementation for contiguous numpy arrays of MPI_Requests"""

    status_buffer = create_status_buffer()
    index = np.empty(1, dtype=np.intc)

    status = _MPI_Waitany(
        requests.size,
        requests.ctypes.data,
        index.ctypes.data,
        status_buffer.ctypes.data,
    )

    return status, index[0]


def waitany(requests):
    """Wrapper for MPI_Waitany. Returns tuple of integers, first representing
    status; second - the index of request that was completed. Status is
    currently not handled. Requires 'requests' parameter to be an array
    or tuple of MPI_Request objects.

    Uninitialized contents of 'requests' (e.g., from numpy.empty()) may
    cause invalid pointer dereference and segmentation faults.
    """

    if isinstance(requests, np.ndarray):
        return _waitany_array_impl(requests)

    if isinstance(requests, (list, tuple)):
        request_buffer = np.hstack(requests)
        return _waitany_array_impl(request_buffer)

    raise TypeError("Invalid type for array of MPI_Request objects")


@overload(waitany)
def _waitany_impl(requests):
    """List of overloads for MPI_Waitany implementation"""

    if isinstance(requests, types.Array):

        def impl(requests):
            return _waitany_array_impl(requests)

    elif isinstance(requests, types.UniTuple):

        def impl(requests):
            req_buffer = np.hstack(requests)
            return _waitany_array_impl(req_buffer)

    else:
        raise TypeError("Invalid type for array of MPI_Request objects")

    return impl


_MPI_Test = libmpi.MPI_Test
_MPI_Test.restype = ctypes.c_int
_MPI_Test.argtypes = [_MpiRequestPtr, ctypes.c_void_p, _MpiStatusPtr]


@numba.njit
def test(request):
    """Wrapper for MPI_Test. Returns tuple containing both status and boolean
    flag that indicates whether given request is completed. Status is currently
    not handled. Requires 'request' parameter to be a c-style pointer to
    MPI_Request (such as returned by 'isend'/'irecv').

    Uninitialized contents of 'request' (e.g., from numpy.empty()) may
    cause invalid pointer dereference and segmentation faults.
    """

    status_buffer = create_status_buffer()
    flag = np.empty(1, dtype=np.intc)

    status = _MPI_Test(request.ctypes.data, flag.ctypes.data, status_buffer.ctypes.data)

    return status, flag[0] != 0


_MPI_Testall = libmpi.MPI_Testall
_MPI_Testall.restype = ctypes.c_int
_MPI_Testall.argtypes = [ctypes.c_int, _MpiRequestPtr, ctypes.c_void_p, _MpiStatusPtr]


@numba.njit
def _testall_array_impl(requests):
    """MPI_Testall implementation for contiguous numpy arrays of MPI_Requests"""

    status_buffer = create_status_buffer(requests.size)
    flag = np.empty(1, dtype=np.intc)
    status = _MPI_Testall(
        requests.size,
        requests.ctypes.data,
        flag.ctypes.data,
        status_buffer.ctypes.data,
    )

    return status, flag[0] != 0


def testall(requests):
    """Wrapper for MPI_Testall. Returns tuple containing both status and boolean
    flag that indicates whether given request is completed. Status is currently
    not handled. Requires 'requests' parameter to be an array or tuple of
    MPI_Request objects.

    Uninitialized contents of 'requests' (e.g., from numpy.empty()) may
    cause invalid pointer dereference and segmentation faults.
    """
    if isinstance(requests, np.ndarray):
        return _testall_array_impl(requests)

    if isinstance(requests, (list, tuple)):
        request_buffer = np.hstack(requests)
        return _testall_array_impl(request_buffer)

    raise TypeError("Invalid type for array of MPI_Request objects")


@overload(testall)
def _testall_impl(requests):
    """List of overloads for MPI_Testall implementation"""

    if isinstance(requests, types.Array):

        def impl(requests):
            return _testall_array_impl(requests)

    elif isinstance(requests, types.UniTuple):

        def impl(requests):
            req_buffer = np.hstack(requests)
            return _testall_array_impl(req_buffer)

    else:
        raise TypeError("Invalid type for array of MPI_Request objects")

    return impl


_MPI_Testany = libmpi.MPI_Testany
_MPI_Testany.restype = ctypes.c_int
_MPI_Testany.argtypes = [
    ctypes.c_int,
    _MpiRequestPtr,
    ctypes.c_void_p,
    ctypes.c_void_p,
    _MpiStatusPtr,
]


@numba.njit
def _testany_array_impl(requests):
    """MPI_Testany implementation for contiguous numpy arrays of MPI_Requests"""

    status_buffer = create_status_buffer()
    flag = np.empty(1, dtype=np.intc)
    index = np.empty(1, dtype=np.intc)
    status = _MPI_Testany(
        requests.size,
        requests.ctypes.data,
        index.ctypes.data,
        flag.ctypes.data,
        status_buffer.ctypes.data,
    )

    return status, flag[0] != 0, index[0]


def testany(requests):
    """Wrapper for MPI_Testany. Returns tuple containing status, boolean flag
    that indicates whether any of requests is completed, and index of request
    that is guaranteed to be completed. Requires 'requests' parameter to be an
    array or tuple of MPI_Request objects.

    Uninitialized contents of 'requests' (e.g., from numpy.empty()) may
    cause invalid pointer dereference and segmentation faults.
    """

    if isinstance(requests, np.ndarray):
        return _testany_array_impl(requests)

    if isinstance(requests, (list, tuple)):
        request_buffer = np.hstack(requests)
        return _testany_array_impl(request_buffer)

    raise TypeError("Invalid type for array of MPI_Request objects")


@overload(testany)
def _testany_impl(requests):
    """List of overloads for MPI_Testany implementation"""

    if isinstance(requests, types.Array):

        def impl(requests):
            return _testany_array_impl(requests)

    elif isinstance(requests, types.UniTuple):

        def impl(requests):
            req_buffer = np.hstack(requests)
            return _testany_array_impl(req_buffer)

    else:
        raise TypeError("Invalid type for array of MPI_Request objects")

    return impl
