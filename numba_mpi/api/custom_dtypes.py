"""wrappers for MPI_Type_*() functions. Note: these are
not using JIT compilation, and constitute thin wrappers
arround mpi4py API with additional logic for housekeeping
of custom type catalogue for use in numba-mpi"""

import numpy as np
from mpi4py import MPI

from numba_mpi.common import _MPI4PY_DTYPES

__TYPES = {}


def type_commit(mpi_dtype):
    mpi_dtype.Commit()


def type_create_struct(dtype):
    blocklengths = np.zeros(len(dtype), dtype=np.intc)
    displacements = np.zeros(len(dtype), dtype=np.intc)
    datatypes = [None] * len(dtype)

    for i, v in enumerate(dtype.fields.values()):
        size = 1
        if type(v[0]) == np.dtypes.VoidDType:
            field_dtype, shape = v[0].subdtype
            size = np.prod(shape)
        else:
            field_dtype = v[0]

        blocklengths[i] = size
        displacements[i] = v[1]
        datatypes[i] = _MPI4PY_DTYPES[field_dtype]

    mpi_type = MPI.Datatype.Create_struct(
        blocklengths=blocklengths, displacements=displacements, datatypes=datatypes
    )

    __TYPES[dtype] = [mpi_type, mpi_type.py2f()]

    return mpi_type


def type_create_resized(oldtype, lb, extent):
    mpi_type = oldtype.Create_resized(lb, extent)

    old_number = oldtype.py2f()

    key = next(k for k, (v1, v2) in __TYPES.items() if v2 == old_number)

    __TYPES[key] = [mpi_type, old_number]

    return mpi_type


def _dtype_to_mpitype(dtype):
    return __TYPES[dtype][0]


# TODO: Free
def type_free(dtype):
    dtype.Free()
