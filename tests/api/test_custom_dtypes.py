import numba
import numpy as np
import pytest
from mpi4py import MPI

import numba_mpi


@numba.njit()
def jit_send(data, dest, tag):
    return numba_mpi.send(data, dest, tag)


@numba.njit()
def jit_recv(data, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
    return numba_mpi.recv(data, source, tag)


@pytest.mark.parametrize(
    "snd, rcv, kind",
    (
        (numba_mpi.send, numba_mpi.recv, "numba disabled"),
        (jit_send, jit_recv, "numba enabled"),
    ),
)
@pytest.mark.parametrize("align", (True, False))
def test_custom_dtypes(snd, rcv, align, kind):
    particle_dtype = np.dtype(
        [
            ("id", np.intc),
            ("x", np.float64),
            ("y", np.float64, (3)),
            ("z", np.float64),
            ("mass", np.float64),
        ],
        align=align,
    )

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    world_rank = comm.Get_rank()

    particle_type = numba_mpi.type_create_struct(particle_dtype)

    particle_type = numba_mpi.type_create_resized(
        oldtype=particle_type, lb=0, extent=particle_dtype.itemsize
    )  # prevent code from failing for align=False

    numba_mpi.type_commit(particle_type)

    particle = (95, 1.5, [3.4, 1.0, 2.5], -9.2, 10.3)
    p_send = np.array([particle, particle], dtype=particle_dtype)

    if world_rank == 0:
        for i in range(1, world_size):
            snd(p_send, i, 0)
    else:
        p_recv = np.zeros(2, dtype=particle_dtype)
        rcv(p_recv, 0, 0)
        np.testing.assert_equal(p_recv, p_send)

    numba_mpi.type_free(particle_type)


@pytest.mark.parametrize(
    "snd, rcv, kind",
    (
        (numba_mpi.send, numba_mpi.recv, "numba disabled"),
        (jit_send, jit_recv, "numba enabled"),
    ),
)
@pytest.mark.parametrize("align", (True, False))
def test_every_type_in_dtypes(snd, rcv, align, kind):
    every_dtype = np.dtype(
        [
            ("uint8", np.uint8),
            ("int32", np.int32),
            ("int64", np.int64),
            ("float64", np.float64),
            ("double", np.double),
            ("complex64", np.complex64),
            ("complex128", np.complex128),
        ],
        align=align,
    )

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    world_rank = comm.Get_rank()

    every_type = numba_mpi.type_create_struct(every_dtype)

    every_type = numba_mpi.type_create_resized(
        oldtype=every_type, lb=0, extent=every_dtype.itemsize
    )

    numba_mpi.type_commit(every_type)

    e1 = (255, -123456789, 122, 1.45e-5, 1.122e-12, 1 + 5j, 1.873 + 3e-9j)
    e2 = (0, 23456789, -987, -1.47e-5, 1.122e-8, 1 + 57j, 1.873 + 3e6j)
    e_send = np.array([e1, e2], dtype=every_dtype)

    if world_rank == 0:
        for i in range(1, world_size):
            snd(e_send, i, 0)
    else:
        e_recv = np.zeros(2, dtype=every_dtype)
        rcv(e_recv, 0, 0)
        np.testing.assert_equal(e_send, e_recv)

    numba_mpi.type_free(every_type)


@pytest.mark.parametrize(
    "snd, rcv, kind",
    (
        (numba_mpi.send, numba_mpi.recv, "numba disabled"),
        (jit_send, jit_recv, "numba enabled"),
    ),
)
@pytest.mark.parametrize("align", (True, False))
def test_multiple_custom_dtypes(snd, rcv, align, kind):
    a_dtype = np.dtype(
        [("a1", np.intc), ("a2", np.float64), ("a3", np.complex64)], align=align
    )

    b_dtype = np.dtype(
        [
            ("b1", np.float64),
            ("b2", np.double),
            ("b3", np.int64),
        ],
        align=align,
    )

    c_dtype = np.dtype([("c1", np.intc), ("c2", np.intc), ("c3", np.intc)], align=align)

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    world_rank = comm.Get_rank()

    a_type = numba_mpi.type_create_struct(a_dtype)
    b_type = numba_mpi.type_create_struct(b_dtype)
    c_type = numba_mpi.type_create_struct(c_dtype)

    c_type = numba_mpi.type_create_resized(
        oldtype=c_type, lb=0, extent=c_dtype.itemsize
    )
    a_type = numba_mpi.type_create_resized(
        oldtype=a_type, lb=0, extent=a_dtype.itemsize
    )
    b_type = numba_mpi.type_create_resized(
        oldtype=b_type, lb=0, extent=b_dtype.itemsize
    )

    numba_mpi.type_commit(a_type)
    numba_mpi.type_commit(b_type)
    numba_mpi.type_commit(c_type)

    a = (100, 53.4e-4, 567 + 890j)
    b = (999.999, -123.321e-4, 2026)
    c = (123, 456, 789)

    a_send = np.array([a], dtype=a_dtype)
    b_send = np.array([b, b], dtype=b_dtype)
    c_send = np.array([c, c, c], dtype=c_dtype)

    if world_rank == 0:
        for i in range(1, world_size):
            snd(b_send, i, 12)
            snd(c_send, i, 13)
            snd(a_send, i, 11)
    else:
        a_recv = np.zeros(1, dtype=a_dtype)
        b_recv = np.zeros(2, dtype=b_dtype)
        c_recv = np.zeros(3, dtype=c_dtype)

        rcv(a_recv, 0, 11)
        rcv(c_recv, 0, 13)
        rcv(b_recv, 0, 12)

        np.testing.assert_equal(a_send, a_recv)
        np.testing.assert_equal(b_send, b_recv)
        np.testing.assert_equal(c_send, c_recv)

    numba_mpi.type_free(a_type)
    numba_mpi.type_free(b_type)
    numba_mpi.type_free(c_type)
