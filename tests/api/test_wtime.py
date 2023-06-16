# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import time

import pytest

import numba_mpi as mpi

SLEEP_TIME_IN_SECONDS = 0.1


@pytest.mark.parametrize("sut", (mpi.wtime, mpi.wtime.py_func))
def test_wtime(sut):
    assert sut() >= 0.0
    assert isinstance(sut(), float)

    start_time = sut()
    time.sleep(SLEEP_TIME_IN_SECONDS)
    assert sut() - start_time > (SLEEP_TIME_IN_SECONDS * 0.9)
