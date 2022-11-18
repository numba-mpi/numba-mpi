""" Numba @njittable MPI wrappers tested on Linux, macOS and Windows """
from pkg_resources import DistributionNotFound, VersionConflict, get_distribution

from .api.allreduce import allreduce
from .api.barrier import barrier
from .api.bcast import bcast
from .api.initialized import initialized
from .api.operator import Operator
from .api.rank import rank
from .api.recv import recv
from .api.send import send
from .api.size import size

try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass
