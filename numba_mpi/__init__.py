""" Numba @njittable MPI wrappers tested on Linux, macOS and Windows """
from .mpi import initialized, size, rank, send, recv, allreduce, Operator
