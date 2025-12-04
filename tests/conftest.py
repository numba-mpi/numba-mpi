"""sanity check ensuring that pytest session was run as an MPI task with more than one worker"""

import numba_mpi

assert (
    numba_mpi.size() > 1
), "the tests require multiple MPI workers - please run with `mpirun pytest ...`"
