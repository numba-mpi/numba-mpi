import math
import time

import numba
import numpy
import pytest
from mpi4py import MPI

import numba_mpi as mpi

PI25DT = 3.141592653589793238462643
comm = MPI.COMM_WORLD


@numba.njit(fastmath=True)
def calculate_pi_numba_mpi(intervals):
    pi = numpy.array(0.0, "d")
    h = 1.0 / intervals
    partial_sum = 0.0

    for i in range(mpi.rank() + 1, intervals, mpi.size()):
        x = h * (float(i) - 0.5)
        partial_sum += 4.0 / (1.0 + x**2)
    mypi = numpy.array(h * partial_sum, "d")

    mpi.allreduce(mypi, pi, mpi.Operator.SUM)
    return pi


@numba.njit(fastmath=True)
def get_my_pi(intervals, rank, size):
    h = 1.0 / intervals
    partial_sum = 0.0

    for i in range(rank + 1, intervals, size):
        x = h * (float(i) - 0.5)
        partial_sum += 4.0 / (1.0 + x**2)

    return numpy.array(h * partial_sum, "d")


def calculate_pi_mpi4py(intervals):
    pi = numpy.array(0.0, "d")
    mypi = get_my_pi(intervals, comm.rank, comm.size)
    comm.Reduce(mypi, [pi, MPI.DOUBLE], op=MPI.SUM, root=0)
    return pi


@pytest.mark.parametrize("intervals", (10000,))
def test_calculate_pi(intervals):
    # first run to compile things
    calculate_pi_numba_mpi(intervals)
    calculate_pi_mpi4py(intervals)

    # proper time measure
    start = time.time()
    pi = calculate_pi_numba_mpi(intervals)
    end_numba_mpi = time.time()

    pi_mpi4py = calculate_pi_mpi4py(intervals)
    end_mpi4py = time.time()

    if comm.rank == 0:
        print(
            f"Calculated PI(numba_mpi): {pi} with error of {math.fabs(pi - PI25DT)}. Time elapsed: {end_numba_mpi - start}"
        )
        print(
            f"Calculated PI(mpi4py): {pi_mpi4py} with error of {math.fabs(pi - PI25DT)}. Time elapsed: {end_mpi4py - end_numba_mpi}"
        )
