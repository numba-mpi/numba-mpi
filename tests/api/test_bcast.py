# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numpy as np
import pytest

import numba_mpi as mpi
from tests.common import MPI_SUCCESS, data_types
from tests.utils import get_random_array


@pytest.mark.parametrize("data_type", data_types)
def test_bcast(data_type):
    root = 0
    data = np.empty(5, data_type).astype(dtype=data_type)
    datatobcast = get_random_array(5, data_type).astype(dtype=data_type)

    if mpi.rank() == root:
        data = datatobcast

    status = mpi.bcast(data, root)

    assert status == MPI_SUCCESS

    np.testing.assert_equal(data, datatobcast)
