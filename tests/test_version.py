# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numba_mpi


def test_version():
    print(numba_mpi.__version__)
