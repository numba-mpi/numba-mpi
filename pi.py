import timeit

import mpi4py
import numba
import numpy

import numba_mpi

INTERVALS = 5000
N_TIMES = 50000
RTOL = 1e-3


@numba.njit
def get_my_pi(out, rank, size):
    h = 1 / INTERVALS
    partial_sum = 0.0
    for i in range(rank + 1, INTERVALS, size):
        x = h * (i - 0.5)
        partial_sum += 4 / (1 + x**2)
    out[0] = h * partial_sum


@numba.njit
def pi_numba_mpi():
    pi = numpy.array(0.0)
    mypi = numpy.empty(1)
    for _ in range(N_TIMES):
        get_my_pi(mypi, numba_mpi.rank(), numba_mpi.size())
        numba_mpi.allreduce(mypi, pi, numba_mpi.Operator.SUM)
        assert abs(pi - numpy.pi) / numpy.pi < RTOL
    return pi


def pi_mpi4py():
    pi = numpy.array(0.0)
    mypi = numpy.empty(1)
    for _ in range(N_TIMES):
        get_my_pi(mypi, mpi4py.MPI.COMM_WORLD.rank, mpi4py.MPI.COMM_WORLD.size)
        mpi4py.MPI.COMM_WORLD.Reduce(
            mypi, [pi, mpi4py.MPI.DOUBLE], op=mpi4py.MPI.SUM, root=0
        )
        assert abs(pi - numpy.pi) / numpy.pi < RTOL
    return pi


for fun in ("pi_numba_mpi()", "pi_mpi4py()"):
    time = min(timeit.repeat(fun, globals=locals(), number=1, repeat=10))
    if numba_mpi.rank() == 0:
        print(f"{N_TIMES} x {fun}:\t{time:.2} s\t(MPI comm size = {numba_mpi.size()})")
