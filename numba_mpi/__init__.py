"""
.. include::../README.md
"""

from importlib.metadata import PackageNotFoundError, version

from .api.allreduce import allreduce
from .api.barrier import barrier
from .api.bcast import bcast
from .api.initialized import initialized
from .api.irecv import irecv
from .api.isend import isend
from .api.operator import Operator
from .api.rank import rank
from .api.recv import recv
from .api.reduce import reduce
from .api.requests import test, testall, testany, wait, waitall, waitany
from .api.scatter_gather import allgather, gather, scatter
from .api.send import send
from .api.size import size
from .api.wtime import wtime
from .common import RequestType

SUCCESS = 0

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
