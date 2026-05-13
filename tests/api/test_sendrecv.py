# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring

import numpy as np
import pytest

import numba_mpi as mpi


class TestSendrecv:

    @pytest.mark.parametrize("sr", (mpi.sendrecv, mpi.sendrecv.py_func))
    @staticmethod
    def test_pairwise_smoke(sr):
        """Minimal 0↔1 exchange — rank 0’s send tag matches rank 1’s recv tag, and vice versa."""
        # arrange
        snd = [
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([7.0, 8.0], dtype=np.float64),
        ]
        rcv = np.empty_like(snd[0])
        tags = [11, 22]

        if (r := mpi.rank()) > 1:
            return

        # act
        status = sr(snd[r], (r + 1) % 2, rcv, sendtag=tags[r], recvtag=tags[r - 1])

        # assert
        assert status == mpi.SUCCESS
        np.testing.assert_array_equal(rcv, snd[r - 1])

    @pytest.mark.parametrize("sr", (mpi.sendrecv, mpi.sendrecv.py_func))
    @staticmethod
    def test_circular(sr):
        """Minimal neighbour exchange — rank 0’s send tag matches rank 1’s recv tag, etc"""
        # arrange
        send_data = [rank + np.array([1.0, 2.0]) for rank in range(mpi.size())]
        recv_data = np.empty_like(send_data[0])

        # act
        status = sr(
            senddata=send_data[mpi.rank()],
            dest=(mpi.rank() + 1) % mpi.size(),
            recvdata=recv_data,
            sendtag=mpi.rank(),
            recvtag=(mpi.rank() - 1) % mpi.size(),
        )

        # assert
        assert status == mpi.SUCCESS
        np.testing.assert_array_equal(recv_data, send_data[mpi.rank() - 1])

    @pytest.mark.parametrize("sr", (mpi.sendrecv, mpi.sendrecv.py_func))
    @staticmethod
    def test_selfsend(sr):
        # arrange
        data = np.empty(10)

        # act
        status = sr(data, mpi.rank(), recvdata=data)

        # assert
        assert status == mpi.SUCCESS

    @pytest.mark.parametrize("sr", (mpi.sendrecv, mpi.sendrecv.py_func))
    @staticmethod
    def test_noncontiguous_input(sr):
        # arrange
        inpt = np.arange(100).reshape(10, 10)[1:8, 1:8]
        inpt += 1
        assert not inpt.flags.c_contiguous
        oupt = np.zeros_like(inpt)
        assert oupt.flags.c_contiguous

        # act
        status = sr(inpt, mpi.rank(), oupt)

        # assert
        assert status == mpi.SUCCESS
        assert all(inpt.flatten() == oupt.flatten())

    @pytest.mark.parametrize("sr", (mpi.sendrecv, mpi.sendrecv.py_func))
    @staticmethod
    def test_noncontiguous_output(sr):
        # arrange
        oupt = np.arange(100).reshape(10, 10)[1:8, 1:8]
        oupt += 1
        inpt = np.zeros_like(oupt)
        assert inpt.flags.c_contiguous
        assert not oupt.flags.c_contiguous

        # act
        status = sr(inpt, mpi.rank(), oupt)

        # assert
        assert status == mpi.SUCCESS
        assert all(inpt.flatten() == oupt.flatten())

    @pytest.mark.parametrize("sr", (mpi.sendrecv, mpi.sendrecv.py_func))
    @staticmethod
    def test_invalid_dest(sr):
        # arrange
        data = np.zeros(10)

        # act
        status = sr(data, dest=mpi.size(), recvdata=data)

        # assert
        assert status != mpi.SUCCESS

    @pytest.mark.parametrize("sr", (mpi.sendrecv, mpi.sendrecv.py_func))
    @staticmethod
    def test_invalid_source(sr):
        # arrange
        data = np.zeros(10)

        # act
        status = sr(data, 0, source=mpi.size(), recvdata=data)

        # assert
        assert status != mpi.SUCCESS
