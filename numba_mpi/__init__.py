""" Numba @njittable MPI wrappers tested on Linux, macOS and Windows """
from pkg_resources import (DistributionNotFound, VersionConflict,
                           get_distribution)

from .api.allreduce import allreduce
from .api.barrier import barrier
from .api.bcast import bcast
from .api.initialized import initialized
from .api.irecv import irecv
from .api.isend import isend
from .api.operator import Operator
from .api.rank import rank
from .api.recv import recv
from .api.requests import (create_requests_array, test, testall, testany, wait,
                           waitall, waitany)
from .api.scatter_gather import allgather, gather, scatter
from .api.send import send
from .api.size import size
from .api.wtime import wtime

try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass
