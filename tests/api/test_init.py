# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import pytest

import numba_mpi as mpi


@pytest.mark.parametrize("sut", [mpi.initialized, mpi.initialized.py_func])
def test_init(sut):
    assert sut()
