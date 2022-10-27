# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD

import numba_mpi as mpi

MPI_SUCCESS = 0


def get_random_array(shape, data_type):
    """helper function creating the same random array in each process"""
    rng = np.random.default_rng(0)
    if np.issubdtype(data_type, np.complexfloating):
        return rng.random(shape) + rng.random(shape) * 1j
    if np.issubdtype(data_type, np.integer):
        return rng.integers(0, 10, size=shape)
    return rng.random(shape)


def allreduce_pyfunc(sendobj, recvobj, operator):
    """helper function to call pyfunc of allreduce without jitting"""
    return mpi.allreduce.py_func(sendobj, recvobj, operator)(sendobj, recvobj, operator)


class TestMPI:

    data_types_real = [
        int,
        np.int32,
        np.int64,
        float,
        np.float64,
        np.double,
    ]
    data_types_complex = [complex, np.complex64, np.complex128]
    data_types = data_types_real + data_types_complex

    @staticmethod
    @pytest.mark.parametrize("sut", [mpi.initialized, mpi.initialized.py_func])
    def test_init(sut):
        assert sut()

    @staticmethod
    @pytest.mark.parametrize("sut", [mpi.size, mpi.size.py_func])
    def test_size(sut):
        size = sut()
        assert size == COMM_WORLD.Get_size()

    @staticmethod
    @pytest.mark.parametrize("sut", [mpi.rank, mpi.rank.py_func])
    def test_rank(sut):
        rank = sut()
        assert rank == COMM_WORLD.Get_rank()

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    @pytest.mark.parametrize("allreduce", [mpi.allreduce, allreduce_pyfunc])
    @pytest.mark.parametrize(
        "op_mpi, op_np",
        [
            (mpi.Operator.SUM, np.sum),
            (mpi.Operator.MIN, np.min),
            (mpi.Operator.MAX, np.max),
        ],
    )
    @pytest.mark.parametrize("data_type", data_types_real)
    def test_allreduce(allreduce, op_mpi, op_np, data_type):
        # test arrays
        src = get_random_array((3,), data_type)
        rcv = np.empty_like(src)
        status = allreduce(src, rcv, operator=op_mpi)
        assert status == MPI_SUCCESS
        expect = op_np(np.tile(src, [mpi.size(), 1]), axis=0)
        np.testing.assert_equal(rcv, expect)

        # test scalars
        src = src[0]
        rcv = np.empty(1, dtype=src.dtype)
        status = allreduce(src, rcv, operator=op_mpi)
        assert status == MPI_SUCCESS
        expect = op_np(np.tile(src, [mpi.size(), 1]), axis=0)
        np.testing.assert_equal(rcv, expect)

        # test 0d arrays
        src = get_random_array((), data_type)
        rcv = np.empty_like(src)
        status = allreduce(src, rcv, operator=op_mpi)
        assert status == MPI_SUCCESS
        expect = op_np(np.tile(src, [mpi.size(), 1]), axis=0)
        np.testing.assert_equal(rcv, expect)
