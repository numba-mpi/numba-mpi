# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numpy as np
import pytest
from numba import njit

import numba_mpi as mpi
from tests.common import data_types_real
from tests.utils import get_random_array


@njit
def jit_reduce(sendobj, recvobj, operator, root):
    """helper function to produce a jitted version of `reduce`"""
    return mpi.reduce(sendobj, recvobj, operator, root)


@pytest.mark.parametrize("reduce", (mpi.reduce, jit_reduce))
@pytest.mark.parametrize(
    "op_mpi, op_np",
    (
        (mpi.Operator.SUM, np.sum),
        (mpi.Operator.MIN, np.min),
        (mpi.Operator.MAX, np.max),
    ),
)
@pytest.mark.parametrize("data_type", data_types_real)
@pytest.mark.parametrize("root", range(mpi.size()))
def test_reduce(reduce, op_mpi, op_np, data_type, root):
    # test arrays
    src = get_random_array((3,), data_type)
    rcv = np.empty_like(src)
    status = reduce(src, rcv, operator=op_mpi, root=root)
    assert status == mpi.SUCCESS
    if mpi.rank() == root:
        expect = op_np(np.tile(src, [mpi.size(), 1]), axis=0)
        np.testing.assert_equal(rcv, expect)

    # test scalars
    src = src[0]
    rcv = np.empty(1, dtype=src.dtype)
    status = reduce(src, rcv, operator=op_mpi, root=root)
    assert status == mpi.SUCCESS
    if mpi.rank() == root:
        expect = op_np(np.tile(src, [mpi.size(), 1]), axis=0)
        np.testing.assert_equal(rcv, expect)

    # test 0d arrays
    src = get_random_array((), data_type)
    rcv = np.empty_like(src)
    status = reduce(src, rcv, operator=op_mpi, root=root)
    assert status == mpi.SUCCESS
    if mpi.rank() == root:
        expect = op_np(np.tile(src, [mpi.size(), 1]), axis=0)
        np.testing.assert_equal(rcv, expect)
