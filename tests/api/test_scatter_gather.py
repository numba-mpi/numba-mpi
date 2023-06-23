# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numba
import numpy as np
import pytest

import numba_mpi as mpi
from tests.common import MPI_SUCCESS, data_types
from tests.utils import get_random_array


@numba.njit()
def jit_scatter(send_data, send_count, recv_data, root):
    return mpi.scatter(send_data, send_count, recv_data, root)


@numba.njit()
def jit_gather(send_data, send_count, recv_data, root):
    return mpi.gather(send_data, send_count, recv_data, root)


@pytest.mark.parametrize("send_count, data_size", ((1, 5), (2, 10), (3, 10)))
@pytest.mark.parametrize("scatter", (mpi.scatter, jit_scatter))
@pytest.mark.parametrize("data_type", data_types)
def test_scatter(data_type, scatter, send_count, data_size):
    if send_count * mpi.size() > data_size:
        pytest.skip()

    root = 0
    rank = mpi.rank()
    data = get_random_array(data_size, data_type).astype(dtype=data_type)
    if rank == root:
        send_data = data
    else:
        send_data = np.empty(shape=(0,), dtype=data_type)

    recv_data = np.empty(send_count, data_type).astype(dtype=data_type)
    status = scatter(send_data, recv_data, send_count, root)

    assert status == MPI_SUCCESS
    np.testing.assert_equal(
        recv_data, data[rank * send_count : (rank + 1) * send_count]
    )


@pytest.mark.parametrize("recv_count, data_size", ((1, 5), (2, 10), (3, 10)))
@pytest.mark.parametrize(
    "gather",
    (
        mpi.gather,
        jit_gather,
    ),
)
@pytest.mark.parametrize("data_type", data_types)
def test_gather(data_type, gather, recv_count, data_size):
    if recv_count * mpi.size() > data_size:
        pytest.skip()

    root = 0
    rank = mpi.rank()
    data = get_random_array(data_size, data_type).astype(dtype=data_type)
    send_data = data[rank * recv_count : (rank + 1) * recv_count]

    recv_data = np.empty(
        shape=(data_size,) if rank == root else (0,), dtype=data_type
    ).astype(dtype=data_type)

    status = gather(send_data, recv_data, recv_count, root)

    assert status == MPI_SUCCESS
    valid_range = slice(0, mpi.size() * recv_count)
    if rank == root:
        np.testing.assert_equal(data[valid_range], recv_data[valid_range])
