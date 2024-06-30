# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,too-many-arguments
import time

import numba
import numpy as np
import pytest
from mpi4py.MPI import ANY_SOURCE, ANY_TAG, COMM_WORLD

import numba_mpi as mpi
from tests.common import data_types
from tests.utils import get_random_array

TEST_WAIT_FULL_IN_SECONDS = 0.3
TEST_WAIT_INCREMENT_IN_SECONDS = 0.1


@numba.njit
def jit_isend(data, dest, tag=0):
    return mpi.isend(data, dest, tag)


@numba.njit
def jit_irecv(data, source=ANY_SOURCE, tag=ANY_TAG):
    return mpi.irecv(data, source, tag)


@numba.njit
def jit_wait(request):
    return mpi.wait(request)


@numba.njit
def jit_waitall(requests):
    return mpi.waitall(requests)


@numba.njit
def jit_waitany(requests):
    return mpi.waitany(requests)


@numba.njit
def jit_test(request):
    return mpi.test(request)


@numba.njit
def jit_testall(requests):
    return mpi.testall(requests)


@numba.njit
def jit_testany(requests):
    return mpi.testany(requests)


@pytest.mark.parametrize(
    "isnd, ircv, wait",
    (
        (jit_isend.py_func, jit_irecv.py_func, jit_wait.py_func),
        (jit_isend, jit_irecv, jit_wait),
    ),
)
@pytest.mark.parametrize("data_type", data_types)
def test_isend_irecv(isnd, ircv, wait, data_type):
    src = get_random_array((3, 3), data_type)
    dst_exp = np.empty_like(src)
    dst_tst = np.empty_like(src)

    if mpi.rank() == 0:
        status, req = isnd(src, dest=1, tag=11)
        assert status == mpi.SUCCESS

        req_exp = COMM_WORLD.Isend(src, dest=1, tag=22)

        status = wait(req)
        assert status == mpi.SUCCESS

        req_exp.wait()

    elif mpi.rank() == 1:
        status, req = ircv(dst_tst, source=0, tag=11)
        assert status == mpi.SUCCESS

        req_exp = COMM_WORLD.Irecv(dst_exp, source=0, tag=22)

        status = wait(req)
        assert status == mpi.SUCCESS

        req_exp.wait()

        np.testing.assert_equal(dst_tst, src)
        np.testing.assert_equal(dst_exp, src)


@pytest.mark.parametrize(
    "isnd, ircv, wait",
    (
        (jit_isend.py_func, jit_irecv.py_func, jit_wait.py_func),
        (jit_isend, jit_irecv, jit_wait),
    ),
)
def test_send_default_tag(isnd, ircv, wait):
    src = get_random_array(())
    dst_tst = np.empty_like(src)

    if mpi.rank() == 0:
        status, req = isnd(src, dest=1)
        assert status == mpi.SUCCESS
        wait(req)
    elif mpi.rank() == 1:
        status, req = ircv(dst_tst, source=0, tag=0)
        assert status == mpi.SUCCESS
        wait(req)

        np.testing.assert_equal(dst_tst, src)


@pytest.mark.parametrize(
    "isnd, ircv, wait",
    (
        (jit_isend.py_func, jit_irecv.py_func, jit_wait.py_func),
        (jit_isend, jit_irecv, jit_wait),
    ),
)
def test_recv_default_tag(isnd, ircv, wait):
    src = get_random_array(())
    dst_tst = np.empty_like(src)

    if mpi.rank() == 0:
        status, req = isnd(src, dest=1, tag=44)
        assert status == mpi.SUCCESS
        wait(req)
    elif mpi.rank() == 1:
        status, req = ircv(dst_tst, source=0)
        assert status == mpi.SUCCESS
        wait(req)

        np.testing.assert_equal(dst_tst, src)


@pytest.mark.parametrize(
    "isnd, ircv, wait",
    (
        (jit_isend.py_func, jit_irecv.py_func, jit_wait.py_func),
        (jit_isend, jit_irecv, jit_wait),
    ),
)
def test_recv_default_source(isnd, ircv, wait):
    src = get_random_array(())
    dst_tst = np.empty_like(src)

    if mpi.rank() == 0:
        status, req = isnd(src, dest=1, tag=44)
        assert status == mpi.SUCCESS
        wait(req)
    elif mpi.rank() == 1:
        status, req = ircv(dst_tst, tag=44)
        assert status == mpi.SUCCESS
        wait(req)

        np.testing.assert_equal(dst_tst, src)


@pytest.mark.parametrize(
    "isnd, ircv, wall",
    [
        (jit_isend, jit_irecv, jit_waitall),
        (jit_isend.py_func, jit_irecv.py_func, jit_waitall.py_func),
    ],
)
@pytest.mark.parametrize("data_type", data_types)
def test_isend_irecv_waitall(isnd, ircv, wall, data_type):
    src1 = get_random_array((5,), data_type)
    src2 = get_random_array((5,), data_type)
    dst1 = np.empty_like(src1)
    dst2 = np.empty_like(src2)

    reqs = np.zeros((2,), dtype=mpi.RequestType)
    if mpi.rank() == 0:
        status, reqs[0:1] = isnd(src1, dest=1, tag=11)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = isnd(src2, dest=1, tag=22)
        assert status == mpi.SUCCESS

        status = wall(reqs)
        assert status == mpi.SUCCESS

    elif mpi.rank() == 1:
        status, reqs[0:1] = ircv(dst1, source=0, tag=11)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = ircv(dst2, source=0, tag=22)
        assert status == mpi.SUCCESS

        status = wall(reqs)
        assert status == mpi.SUCCESS

        np.testing.assert_equal(dst1, src1)
        np.testing.assert_equal(dst2, src2)


@pytest.mark.parametrize(
    "isnd, ircv, wall",
    [
        (jit_isend.py_func, jit_irecv.py_func, jit_waitall.py_func),
        (jit_isend, jit_irecv, jit_waitall),
    ],
)
def test_isend_irecv_waitall_tuple(isnd, ircv, wall):
    src1 = get_random_array((5,))
    src2 = get_random_array((5,))
    dst1 = np.empty_like(src1)
    dst2 = np.empty_like(src2)

    if mpi.rank() == 0:
        status, req_1 = isnd(src1, dest=1, tag=11)
        assert status == mpi.SUCCESS
        status, req_2 = isnd(src2, dest=1, tag=22)
        assert status == mpi.SUCCESS

        status = wall((req_1, req_2))
        assert status == mpi.SUCCESS

    elif mpi.rank() == 1:
        status, req_1 = ircv(dst1, source=0, tag=11)
        assert status == mpi.SUCCESS
        status, req_2 = ircv(dst2, source=0, tag=22)
        assert status == mpi.SUCCESS

        status = wall((req_1, req_2))
        assert status == mpi.SUCCESS

        np.testing.assert_equal(dst1, src1)
        np.testing.assert_equal(dst2, src2)


@pytest.mark.parametrize(
    "isnd, ircv, wall",
    [
        (jit_isend.py_func, jit_irecv.py_func, jit_waitall.py_func),
        (jit_isend, jit_irecv, jit_waitall),
    ],
)
def test_isend_irecv_waitall_exchange(isnd, ircv, wall):
    src = get_random_array((5,))
    dst = np.empty_like(src)

    reqs = np.zeros((2,), dtype=mpi.RequestType)
    if mpi.rank() == 0:
        status, reqs[0:1] = isnd(src, dest=1, tag=11)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = ircv(dst, source=1, tag=22)
        assert status == mpi.SUCCESS

    elif mpi.rank() == 1:
        status, reqs[0:1] = isnd(src, dest=0, tag=22)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = ircv(dst, source=0, tag=11)
        assert status == mpi.SUCCESS

    wall(reqs)

    np.testing.assert_equal(dst, src)


@pytest.mark.parametrize(
    "fun",
    (
        jit_waitany.py_func,
        jit_waitall.py_func,
        jit_testany.py_func,
        jit_testall.py_func,
        jit_waitany,
        jit_waitall,
        jit_testany,
        jit_testall,
    ),
)
def test_wall_segfault(fun):
    reqs = np.zeros((2,), dtype=mpi.RequestType)
    fun(reqs)


@pytest.mark.parametrize(
    "isnd, ircv, wany, wall",
    [
        (
            jit_isend.py_func,
            jit_irecv.py_func,
            jit_waitany.py_func,
            jit_waitall.py_func,
        ),
        (jit_isend, jit_irecv, jit_waitany, jit_waitall),
    ],
)
@pytest.mark.parametrize("data_type", data_types)
def test_isend_irecv_waitany(isnd, ircv, wany, wall, data_type):
    src1 = get_random_array((5,), data_type)
    src2 = get_random_array((5,), data_type)
    dst1 = np.empty_like(src1)
    dst2 = np.empty_like(src2)

    reqs = np.zeros((2,), dtype=mpi.RequestType)
    if mpi.rank() == 0:
        status, reqs[0:1] = isnd(src1, dest=1, tag=11)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = isnd(src2, dest=1, tag=22)
        assert status == mpi.SUCCESS
        wall(reqs)

    elif mpi.rank() == 1:
        status, reqs[0:1] = ircv(dst1, source=0, tag=11)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = ircv(dst2, source=0, tag=22)
        assert status == mpi.SUCCESS

        status, index = wany(reqs)
        assert status == mpi.SUCCESS

        if index == 0:
            np.testing.assert_equal(dst1, src1)
        elif index == 1:
            np.testing.assert_equal(dst2, src2)
        else:
            assert False

        wall(reqs)


@pytest.mark.parametrize(
    "isnd, ircv, tst, wait",
    [
        (jit_isend, jit_irecv, jit_test, jit_wait),
        (jit_isend.py_func, jit_irecv.py_func, jit_test.py_func, jit_wait.py_func),
    ],
)
def test_isend_irecv_test(isnd, ircv, tst, wait):
    src = get_random_array((5,))
    dst = np.empty_like(src)

    if mpi.rank() == 0:
        time.sleep(TEST_WAIT_FULL_IN_SECONDS)
        status, req = isnd(src, dest=1, tag=11)
        assert status == mpi.SUCCESS
        wait(req)
    elif mpi.rank() == 1:
        status, req = ircv(dst, source=0, tag=11)
        assert status == mpi.SUCCESS

        status, flag = tst(req)
        while status == mpi.SUCCESS and not flag:
            time.sleep(TEST_WAIT_INCREMENT_IN_SECONDS)
            status, flag = tst(req)

        np.testing.assert_equal(dst, src)
        wait(req)


@pytest.mark.parametrize(
    "isnd, ircv, tall, wall",
    [
        (
            jit_isend.py_func,
            jit_irecv.py_func,
            jit_testall.py_func,
            jit_waitall.py_func,
        ),
        (jit_isend, jit_irecv, jit_testall, jit_waitall),
    ],
)
def test_isend_irecv_testall(isnd, ircv, tall, wall):
    src1 = get_random_array((5,))
    src2 = get_random_array((5,))
    dst1 = np.empty_like(src1)
    dst2 = np.empty_like(src2)

    reqs = np.zeros((2,), dtype=mpi.RequestType)
    if mpi.rank() == 0:
        time.sleep(TEST_WAIT_FULL_IN_SECONDS)

        status, reqs[0:1] = isnd(src1, dest=1, tag=11)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = isnd(src2, dest=1, tag=22)
        assert status == mpi.SUCCESS

        wall(reqs)

    elif mpi.rank() == 1:
        status, reqs[0:1] = ircv(dst1, source=0, tag=11)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = ircv(dst2, source=0, tag=22)
        assert status == mpi.SUCCESS

        status, flag = tall(reqs)
        while status == mpi.SUCCESS and not flag:
            time.sleep(TEST_WAIT_INCREMENT_IN_SECONDS)
            status, flag = tall(reqs)

        np.testing.assert_equal(dst1, src1)
        np.testing.assert_equal(dst2, src2)

        wall(reqs)


@pytest.mark.parametrize(
    "isnd, ircv, tany, wall",
    [
        (
            jit_isend.py_func,
            jit_irecv.py_func,
            jit_testany.py_func,
            jit_waitall.py_func,
        ),
        (jit_isend, jit_irecv, jit_testany, jit_waitall),
    ],
)
def test_isend_irecv_testany(isnd, ircv, tany, wall):
    src1 = get_random_array((5,))
    src2 = get_random_array((5,))
    dst1 = np.empty_like(src1)
    dst2 = np.empty_like(src2)

    reqs = np.zeros((2,), dtype=mpi.RequestType)
    if mpi.rank() == 0:
        time.sleep(TEST_WAIT_FULL_IN_SECONDS)

        status, reqs[0:1] = isnd(src1, dest=1, tag=11)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = isnd(src2, dest=1, tag=22)
        assert status == mpi.SUCCESS

        wall(reqs)

    elif mpi.rank() == 1:
        status, reqs[0:1] = ircv(dst1, source=0, tag=11)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = ircv(dst2, source=0, tag=22)
        assert status == mpi.SUCCESS

        status, flag, index = tany(reqs)
        while status == mpi.SUCCESS and not flag:
            time.sleep(TEST_WAIT_INCREMENT_IN_SECONDS)
            status, flag, index = tany(reqs)

        if index == 0:
            np.testing.assert_equal(dst1, src1)
        elif index == 1:
            np.testing.assert_equal(dst2, src2)
        else:
            assert False

        wall(reqs)
