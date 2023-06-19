# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import time

import pytest

import numba_mpi as mpi

SLEEP_TIME_IN_SECONDS = 0.1


@pytest.mark.parametrize(
    "sut",
    (
        pytest.param(mpi.wtime, id="JIT if enabled"),
        pytest.param(mpi.wtime.py_func, id="py_func"),
    ),
)
class TestWtime:
    @staticmethod
    def test_returns_value_ge_zero(sut):
        value = sut()
        assert value >= 0.0

    @staticmethod
    def test_returns_a_float(sut):
        value = sut()
        assert isinstance(value, float)

    @staticmethod
    def test_returned_time_matches_sleep_time(sut):
        start_time = sut()
        time.sleep(SLEEP_TIME_IN_SECONDS)
        value = sut() - start_time
        assert value > (SLEEP_TIME_IN_SECONDS * 0.9)
