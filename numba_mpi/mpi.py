""" Numba @njittable MPI wrappers tested on Linux, macOS and Windows """
import ctypes
import platform
import numba
from numba.core import types, cgutils
import numpy as np
from mpi4py import MPI


# pylint: disable-next=protected-access
if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    _MpiComm = ctypes.c_int
else:
    _MpiComm = ctypes.c_void_p

# pylint: disable-next=protected-access
if MPI._sizeof(MPI.Datatype) == ctypes.sizeof(ctypes.c_int):
    _MpiDatatype = ctypes.c_int
else:
    _MpiDatatype = ctypes.c_void_p

_MpiStatusPtr = ctypes.c_void_p

if platform.system() == 'Linux':
    LIB = 'libmpi.so'
elif platform.system() == 'Windows':
    LIB = 'msmpi.dll'
elif platform.system() == 'Darwin':
    LIB = 'libmpi.dylib'
else:
    raise NotImplementedError()
libmpi = ctypes.CDLL(LIB)

_MPI_Initialized = libmpi.MPI_Initialized
_MPI_Initialized.restype = ctypes.c_int
_MPI_Initialized.argtypes = [ctypes.c_void_p]

_MPI_Comm_size = libmpi.MPI_Comm_size
_MPI_Comm_size.restype = ctypes.c_int
_MPI_Comm_size.argtypes = [_MpiComm, ctypes.c_void_p]

_MPI_Comm_rank = libmpi.MPI_Comm_rank
_MPI_Comm_rank.restype = ctypes.c_int
_MPI_Comm_rank.argtypes = [_MpiComm, ctypes.c_void_p]

# pylint: disable-next=protected-access
_MPI_Comm_World_ptr = MPI._addressof(MPI.COMM_WORLD)

# pylint: disable-next=protected-access
_MPI_Double_ptr = MPI._addressof(MPI.DOUBLE)

_MPI_Send = libmpi.MPI_Send
_MPI_Send.restype = ctypes.c_int
_MPI_Send.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    _MpiDatatype,
    ctypes.c_int,
    ctypes.c_int,
    _MpiComm
]

_MPI_Recv = libmpi.MPI_Recv
_MPI_Recv.restype = ctypes.c_int
_MPI_Recv.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    _MpiDatatype,
    ctypes.c_int,
    ctypes.c_int,
    _MpiComm,
    _MpiStatusPtr
]


def _mpi_comm_world():
    return _MpiComm.from_address(_MPI_Comm_World_ptr)


@numba.extending.overload(_mpi_comm_world)
def _mpi_comm_world_njit():
    def impl():
        return numba.carray(
            # pylint: disable-next=no-value-for-parameter
            _address_as_void_pointer(_MPI_Comm_World_ptr),
            shape=(1,),
            dtype=np.intp
        )[0]
    return impl


def _mpi_double():
    return _MpiDatatype.from_address(_MPI_Double_ptr)

# WIN DOUBLE - 0x4c00080b

@numba.extending.overload(_mpi_double)
def _mpi_double_njit():
    def impl():
        return numba.carray(
            # pylint: disable-next=no-value-for-parameter
            _address_as_void_pointer(_MPI_Double_ptr),
            shape=(1,),
            dtype=np.intp
        )[0]
    return impl


# https://stackoverflow.com/questions/61509903/how-to-pass-array-pointer-to-numba-function
@numba.extending.intrinsic
def _address_as_void_pointer(_, src):
    """ returns a void pointer from a given memory address """
    sig = types.voidptr(src)

    def codegen(__, builder, ___, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen


@numba.njit()
def initialized():
    """ wrapper for MPI_Initialized() """
    flag = np.empty((1,), dtype=np.intc)
    status = _MPI_Initialized(flag.ctypes.data)
    assert status == 0
    return bool(flag[0])


@numba.njit()
def size():
    """ wrapper for MPI_Comm_size() """
    value = np.empty(1, dtype=np.intc)
    status = _MPI_Comm_size(_mpi_comm_world(), value.ctypes.data)
    assert status == 0
    return value[0]


@numba.njit()
def rank():
    """ wrapper for MPI_Comm_rank() """
    value = np.empty(1, dtype=np.intc)
    status = _MPI_Comm_rank(_mpi_comm_world(), value.ctypes.data)
    assert status == 0
    return value[0]


@numba.njit
def send(data, dest, tag):
    """ wrapper for MPI_Send """
    data = np.ascontiguousarray(data)
    result = _MPI_Send(
        data.ctypes.data,
        data.size,
        _mpi_double(),
        dest,
        tag,
        _mpi_comm_world()
    )
    assert result == 0

    # The following no-op prevents numba from too aggressive optimizations
    # This looks like a bug in numba (tested for version 0.55)
    data[0]  # pylint: disable=pointless-statement


@numba.njit()
def recv(data, source, tag):
    """ wrapper for MPI_Recv (writes data directly if `data` is contiguous, otherwise
        allocates a buffer and later copies the data into non-contiguous `data` array) """
    status = np.empty(5, dtype=np.intc)

    buffer = (
        data if data.flags.c_contiguous
        else np.empty_like(data)
    )

    result = _MPI_Recv(
        buffer.ctypes.data,
        buffer.size,
        _mpi_double(),
        source,
        tag,
        _mpi_comm_world(),
        status.ctypes.data
    )
    assert result == 0

    if not data.flags.c_contiguous:
        data[:] = buffer
