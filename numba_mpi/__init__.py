""" Numba @njittable MPI wrappers tested on Linux, macOS and Windows """
from pkg_resources import DistributionNotFound, VersionConflict, get_distribution

from .mpi import Operator, allreduce, initialized, rank, recv, send, size

try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass
