# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,too-many-arguments
import time

import numba
import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD

import numba_mpi as mpi
from tests.common import data_types
from tests.utils import get_random_array

TEST_WAIT_FULL_IN_SECONDS = 0.3
TEST_WAIT_INCREMENT_IN_SECONDS = 0.1


@numba.njit
def jit_waitall(requests):
    return mpi.waitall(requests)


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
    "isnd, ircv, tst, wait",
    [
        (mpi.isend, mpi.irecv, mpi.test, mpi.wait),
        (mpi.isend.py_func, mpi.irecv.py_func, mpi.test.py_func, mpi.wait.py_func),
    ],
)
def test_isend_irecv_test(isnd, ircv, tst, wait):
    src = get_random_array((5,))
    dst = np.empty_like(src)

    if mpi.rank() == 0:
        time.sleep(TEST_WAIT_FULL_IN_SECONDS)
        req = isnd(src, dest=1, tag=11)
        wait(req)
    elif mpi.rank() == 1:
        req = ircv(dst, source=0, tag=11)

        while not tst(req):
            time.sleep(TEST_WAIT_INCREMENT_IN_SECONDS)

        np.testing.assert_equal(dst, src)
        wait(req)
