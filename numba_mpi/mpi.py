import numba
import ctypes
import numpy as np
import platform
from mpi4py import MPI


if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    _MPI_Comm_t = ctypes.c_int
else:
    _MPI_Comm_t = ctypes.c_void_p

if MPI._sizeof(MPI.Datatype) == ctypes.sizeof(ctypes.c_int):
    _MPI_Datatype_t = ctypes.c_int
else:
    _MPI_Datatype_t = ctypes.c_void_p

_MPI_Status_ptr_t = ctypes.c_void_p

if platform.system() == 'Linux':
    lib = 'libmpi.so'
elif platform.system() == 'Windows':
    lib = 'msmpi.dll'
elif platform.system() == 'Darwin':
    lib = 'libmpi.dylib'
else:
    raise NotImplementedError()
libmpi = ctypes.CDLL(lib)

_MPI_Initialized = libmpi.MPI_Initialized
_MPI_Initialized.restype = ctypes.c_int
_MPI_Initialized.argtypes = [ctypes.c_void_p]

_MPI_Comm_size = libmpi.MPI_Comm_size
_MPI_Comm_size.restype = ctypes.c_int
_MPI_Comm_size.argtypes = [_MPI_Comm_t, ctypes.c_void_p]

_MPI_Comm_rank = libmpi.MPI_Comm_rank
_MPI_Comm_rank.restype = ctypes.c_int
_MPI_Comm_rank.argtypes = [_MPI_Comm_t, ctypes.c_void_p]

_MPI_Comm_World_ptr = MPI._addressof(MPI.COMM_WORLD)

_MPI_Double_ptr = MPI._addressof(MPI.DOUBLE)

# int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,  MPI_Comm comm)
_MPI_Send = libmpi.MPI_Send
_MPI_Send.restype = ctypes.c_int
_MPI_Send.argtypes = [ctypes.c_void_p, ctypes.c_int, _MPI_Datatype_t, ctypes.c_int, ctypes.c_int, _MPI_Comm_t]

#int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status * status)
_MPI_Recv = libmpi.MPI_Recv
_MPI_Recv.restype = ctypes.c_int
_MPI_Recv.argtypes = [ctypes.c_void_p, ctypes.c_int, _MPI_Datatype_t, ctypes.c_int, ctypes.c_int, _MPI_Comm_t, _MPI_Status_ptr_t]

def _MPI_Comm_world():
    return _MPI_Comm_t.from_address(_MPI_Comm_World_ptr)

@numba.extending.overload(_MPI_Comm_world)
def _MPI_Comm_world_njit():
    def impl():
        return numba.carray(
            address_as_void_pointer(_MPI_Comm_World_ptr),
            shape=(1,),
            dtype=np.intp
        )[0]
    return impl


def _MPI_Double():
    return _MPI_Datatype_t.from_address(_MPI_Double_ptr)

# WIN DOUBLE - 0x4c00080b

@numba.extending.overload(_MPI_Double)
def _MPI_Double_njit():
    def impl():
        return numba.carray(
            address_as_void_pointer(_MPI_Double_ptr),
            shape=(1,),
            dtype=np.intp
        )[0]
    return impl


# https://stackoverflow.com/questions/61509903/how-to-pass-array-pointer-to-numba-function
@numba.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    """ returns a void pointer from a given memory address """
    from numba.core import types, cgutils
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen


@numba.njit()
def initialized():
    flag = np.empty((1,), dtype=np.intc)
    status = _MPI_Initialized(flag.ctypes.data)
    assert status == 0
    return bool(flag[0])


@numba.njit()
def size():
    value = np.empty(1, dtype=np.intc)
    status = _MPI_Comm_size(_MPI_Comm_world(), value.ctypes.data)
    assert status == 0
    return value[0]


@numba.njit()
def rank():
    value = np.empty(1, dtype=np.intc)
    status = _MPI_Comm_rank(_MPI_Comm_world(), value.ctypes.data)
    assert status == 0
    return value[0]

# int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,  MPI_Comm comm)
@numba.njit
def send(data, dest, tag):
    result = _MPI_Send(data.ctypes.data, data.size, _MPI_Double(), dest, tag, _MPI_Comm_world())
    assert result == 0

#int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status * status)
@numba.njit()
def recv(data, source, tag):
    status = np.empty(5, dtype=np.intc)
    result = _MPI_Recv(data.ctypes.data, data.size, _MPI_Double(), source, tag, _MPI_Comm_world(), status.ctypes.data)
    assert result == 0
