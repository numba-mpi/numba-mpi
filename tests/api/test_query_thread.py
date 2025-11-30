# pylint: disable=missing-function-docstring,missing-module-docstring
from mpi4py import MPI

import numba_mpi as mpi


def test_query_thread():
    assert MPI.Query_thread() == mpi.query_thread()
