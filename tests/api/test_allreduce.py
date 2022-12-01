# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numpy as np
import pytest
from numba import njit

import numba_mpi as mpi
from tests.common import MPI_SUCCESS, data_types_real
from tests.utils import get_random_array


@njit
def jit_allreduce(sendobj, recvobj, operator):
    """helper function to produce a jitted version of `allreduce`"""
    return mpi.allreduce(sendobj, recvobj, operator)


@pytest.mark.parametrize("allreduce", (mpi.allreduce, jit_allreduce))
@pytest.mark.parametrize(
    "op_mpi, op_np",
    (
        (mpi.Operator.SUM, np.sum),
        (mpi.Operator.MIN, np.min),
        (mpi.Operator.MAX, np.max),
    ),
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
