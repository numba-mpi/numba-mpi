# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
from mpi4py.MPI import COMM_WORLD
import numpy as np
import pytest
import numba_mpi as mpi


class TestMPI:
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
    @pytest.mark.parametrize("snd, rcv", [
        (mpi.send, mpi.recv),
        (mpi.send.py_func, mpi.recv.py_func)
    ])
    @pytest.mark.parametrize("data_type", [np.float64])
    def test_send_recv(snd, rcv, data_type):
        src = np.array([1, 2, 3, 4, 5], dtype=data_type)
        dst_tst = np.empty(5, dtype=data_type)
        dst_exp = np.empty(5, dtype=data_type)

        if mpi.rank() == 0:
            snd(src, dest=1, tag=11)
            COMM_WORLD.Send(src, dest=1, tag=22)
        elif mpi.rank() == 1:
            rcv(dst_tst, source=0, tag=11)
            COMM_WORLD.Recv(dst_exp, source=0, tag=22)

            assert np.all(dst_tst == src)
            assert np.all(dst_tst == dst_exp)

    @staticmethod
    @pytest.mark.parametrize("snd, rcv", [
        (mpi.send, mpi.recv),
        (mpi.send.py_func, mpi.recv.py_func)
    ])
    @pytest.mark.parametrize("data_type", [np.float64])
    def test_send_recv_noncontiguous(snd, rcv, data_type):
        src = np.array([1, 2, 3, 4, 5], dtype=data_type)
        dst_tst = np.zeros(5, dtype=data_type)

        if mpi.rank() == 0:
            snd(src[::2], dest=1, tag=11)
        elif mpi.rank() == 1:
            rcv(dst_tst[::2], source=0, tag=11)

            assert np.all(dst_tst[1::2] == 0)
            assert np.all(dst_tst[::2] == src[::2])
