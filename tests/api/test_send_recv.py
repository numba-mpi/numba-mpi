# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD

import numba_mpi as mpi
from tests.common import MPI_SUCCESS, data_types
from tests.utils import get_random_array


@pytest.mark.parametrize(
    "snd, rcv", [(mpi.send, mpi.recv), (mpi.send.py_func, mpi.recv.py_func)]
)
@pytest.mark.parametrize("fortran_order", [True, False])
@pytest.mark.parametrize("data_type", data_types)
def test_send_recv(snd, rcv, fortran_order, data_type):
    src = get_random_array((3, 3), data_type)

    if fortran_order:
        src = np.asfortranarray(src, dtype=data_type)
    else:
        src = src.astype(dtype=data_type)

    dst_exp = np.empty_like(src)
    dst_tst = np.empty_like(src)

    if mpi.rank() == 0:
        status = snd(src, dest=1, tag=11)
        assert status == MPI_SUCCESS
        COMM_WORLD.Send(src, dest=1, tag=22)
    elif mpi.rank() == 1:
        status = rcv(dst_tst, source=0, tag=11)
        assert status == MPI_SUCCESS
        COMM_WORLD.Recv(dst_exp, source=0, tag=22)

        np.testing.assert_equal(dst_tst, src)
        np.testing.assert_equal(dst_exp, src)


@pytest.mark.parametrize(
    "snd, rcv", [(mpi.send, mpi.recv), (mpi.send.py_func, mpi.recv.py_func)]
)
@pytest.mark.parametrize("data_type", data_types)
def test_send_recv_noncontiguous(snd, rcv, data_type):
    src = get_random_array((5,), data_type)
    dst_tst = np.zeros_like(src)

    if mpi.rank() == 0:
        status = snd(src[::2], dest=1, tag=11)
        assert status == MPI_SUCCESS
    elif mpi.rank() == 1:
        status = rcv(dst_tst[::2], source=0, tag=11)
        assert status == MPI_SUCCESS

        np.testing.assert_equal(dst_tst[1::2], 0)
        np.testing.assert_equal(dst_tst[::2], src[::2])


@pytest.mark.parametrize(
    "snd, rcv", [(mpi.send, mpi.recv), (mpi.send.py_func, mpi.recv.py_func)]
)
@pytest.mark.parametrize("data_type", data_types)
def test_send_0d_arrays(snd, rcv, data_type):
    src = get_random_array((), data_type)
    dst_tst = np.empty_like(src)

    if mpi.rank() == 0:
        status = snd(src, dest=1, tag=11)
        assert status == MPI_SUCCESS
    elif mpi.rank() == 1:
        status = rcv(dst_tst, source=0, tag=11)
        assert status == MPI_SUCCESS

        np.testing.assert_equal(dst_tst, src)
