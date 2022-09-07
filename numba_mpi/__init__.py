""" Numba @njittable MPI wrappers tested on Linux, macOS and Windows """
from pkg_resources import DistributionNotFound, VersionConflict, get_distribution
from .mpi import initialized, size, rank, send, recv, allreduce, Operator


try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass
