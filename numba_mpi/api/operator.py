"""operators supported by MPI"""
from enum import IntEnum

from mpi4py import MPI


class Operator(IntEnum):
    """collection of operators that MPI supports"""

    # pylint: disable=protected-access
    MAX = MPI._addressof(MPI.MAX)
    MIN = MPI._addressof(MPI.MIN)
    SUM = MPI._addressof(MPI.SUM)
    # pylint: enable=protected-access
