# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import pytest
from mpi4py import MPI

import numba_mpi as mpi


@pytest.mark.parametrize(
    argnames="sut", argvalues=(mpi.error_class, mpi.error_class.py_func)
)
@pytest.mark.parametrize(
    "err_const",
    [
        0,
        MPI.ERR_ARG,
        MPI.ERR_COMM,
        MPI.ERR_RANK,
        MPI.ERR_TYPE,
    ],
)
def test_error_class(sut, err_const):
    result = sut(err_const)

    assert result == MPI.Get_error_class(err_const)
