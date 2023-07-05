# pylint: skip-file

# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD

import numba_mpi as mpi
from tests.common import data_types
from tests.utils import get_random_array


@pytest.mark.parametrize(
    "isnd, ircv, wait",
    (
        (mpi.isend, mpi.irecv, mpi.wait),
        (mpi.isend.py_func, mpi.irecv.py_func, mpi.wait.py_func),
    ),
)
@pytest.mark.parametrize("data_type", data_types)
def test_isend_irecv(isnd, ircv, wait, data_type):
    src = get_random_array((3, 3), data_type)
    dst_exp = np.empty_like(src)
    dst_tst = np.empty_like(src)

    if mpi.rank() == 0:
        req = isnd(src, dest=1, tag=11)
        req_exp = COMM_WORLD.Isend(src, dest=1, tag=22)
        wait(req)
        req_exp.wait()
    elif mpi.rank() == 1:
        req = ircv(dst_tst, source=0, tag=11)
        req_exp = COMM_WORLD.Irecv(dst_exp, source=0, tag=22)
        wait(req)
        req_exp.wait()

        np.testing.assert_equal(dst_tst, src)
        np.testing.assert_equal(dst_exp, src)


@pytest.mark.parametrize(
    "isnd, ircv, wait",
    (
        (mpi.isend, mpi.irecv, mpi.wait),
        (mpi.isend.py_func, mpi.irecv.py_func, mpi.wait.py_func),
    ),
)
def test_send_default_tag(isnd, ircv, wait):
    src = get_random_array(())
    dst_tst = np.empty_like(src)

    if mpi.rank() == 0:
        req = isnd(src, dest=1)
        wait(req)
    elif mpi.rank() == 1:
        req = ircv(dst_tst, source=0, tag=0)
        wait(req)

        np.testing.assert_equal(dst_tst, src)


@pytest.mark.parametrize(
    "isnd, ircv, wait",
    (
        (mpi.isend, mpi.irecv, mpi.wait),
        (mpi.isend.py_func, mpi.irecv.py_func, mpi.wait.py_func),
    ),
)
def test_recv_default_tag(isnd, ircv, wait):
    src = get_random_array(())
    dst_tst = np.empty_like(src)

    if mpi.rank() == 0:
        req = isnd(src, dest=1, tag=44)
        wait(req)
    elif mpi.rank() == 1:
        req = ircv(dst_tst, source=0)
        wait(req)

        np.testing.assert_equal(dst_tst, src)


@pytest.mark.parametrize(
    "isnd, ircv, wait",
    (
        (mpi.isend, mpi.irecv, mpi.wait),
        (mpi.isend.py_func, mpi.irecv.py_func, mpi.wait.py_func),
    ),
)
def test_recv_default_source(isnd, ircv, wait):
    src = get_random_array(())
    dst_tst = np.empty_like(src)

    if mpi.rank() == 0:
        req = isnd(src, dest=1, tag=44)
        wait(req)
    elif mpi.rank() == 1:
        req = ircv(dst_tst, tag=44)
        wait(req)

        np.testing.assert_equal(dst_tst, src)


@pytest.mark.parametrize(
    "isnd, ircv, wall, create_reqs",
    [
        (mpi.isend, mpi.irecv, mpi.waitall, mpi.create_requests_array),
        (
            mpi.isend.py_func,
            mpi.irecv.py_func,
            mpi.waitall.py_func,
            mpi.create_requests_array.py_func,
        ),
    ],
)
@pytest.mark.parametrize("data_type", data_types)
def test_isend_irecv_waitall_oneway(isnd, ircv, wall, create_reqs, data_type, size):
    src1 = get_random_array((5,), data_type)
    src2 = get_random_array((5,), data_type)
    dst1 = np.empty_like(src1)
    dst2 = np.empty_like(src2)

    reqs = create_reqs(2)
    if mpi.rank() == 0:
        reqs[: mpi.common.MPI_REQUEST_SIZE] = isnd(src1, dest=1, tag=11)
        reqs[mpi.common.MPI_REQUEST_SIZE :] = isnd(src2, dest=1, tag=22)
        wall(reqs)
    elif mpi.rank() == 1:
        reqs[: mpi.common.MPI_REQUEST_SIZE] = r1 = ircv(dst1, source=0, tag=11)
        reqs[mpi.common.MPI_REQUEST_SIZE :] = r2 = ircv(dst2, source=0, tag=22)
        wall(reqs)

        np.testing.assert_equal(dst1, src1)
        np.testing.assert_equal(dst2, src2)


"""
@pytest.mark.parametrize(
    "isnd, ircv, wall, create_reqs",
    [
        (mpi.isend, mpi.irecv, mpi.waitall, mpi.create_requests_array),
        (
            mpi.isend.py_func,
            mpi.irecv.py_func,
            mpi.waitall.py_func,
            mpi.create_requests_array.py_func,
        ),
    ],
)
@pytest.mark.parametrize("data_type", data_types)
def test_isend_irecv_waitall_exchange(isnd, ircv, wall, create_reqs, data_type):
    src = get_random_array((5,), data_type)
    dst = np.empty_like(src)

    reqs = create_reqs(2)
    if mpi.rank() == 0:
        reqs[0] = isnd(src, dest=1, tag=11)
        reqs[1] = ircv(dst, source=1, tag=22)
    elif mpi.rank() == 1:
        reqs[0] = isnd(src, dest=0, tag=22)
        reqs[1] = ircv(dst, source=0, tag=11)
    wall(reqs)

    np.testing.assert_equal(dst, src)
"""
