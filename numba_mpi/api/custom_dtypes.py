"""wrappers for MPI_Type_*() functions. Note: these are
not using JIT compilation, and constitute thin wrappers
arround mpi4py API with additional logic for housekeeping
of custom type catalogue for use in numba-mpi"""

import numpy as np
from mpi4py import MPI

from numba_mpi.common import _MPI4PY_DTYPES

__DTYPE_TO_MPITYPE = {}
__MPIPY2F_TO_DTYPE = {}


def type_commit(mpi_type):
    """Wrapper for MPI_Type_commit. Commits a data type.
    Additionally, it checks if extent of a `mpi_type` is set correctly;
    otherwise raises `ValueError`.

    Args:
        mpi_type (MPI_Datatype): input datatype to commit.

    Returns:
        None
    """
    a = __MPIPY2F_TO_DTYPE[mpi_type.py2f()].itemsize
    b = mpi_type.extent

    if a != b:
        raise ValueError(
            f"Size of the passed dtype ({a}) "
            f"does not match its extent ({b}).\n"
            "Use align=True when using np.dtype(..., align=True) "
            "or change the extent with numba_mpi.type_create_resized(..., extent=dtype.itemsize)"
        )

    mpi_type.Commit()


def type_create_struct(dtype):
    """Wrapper for MPI_Type_create_struct. Create an MPI datatype
    from given numpy.dtype datatype. Doesn't require passing set of `blocklengths`,
    `displacements` and `datatypes`.

    Args:
        dtype (numpy.dtype): input datatype.

    Returns:
        MPI_Datatype: created datatype.
    """
    blocklengths = np.zeros(len(dtype), dtype=np.intc)
    displacements = np.zeros(len(dtype), dtype=np.intc)
    datatypes = [None] * len(dtype)

    for i, v in enumerate(dtype.fields.values()):
        size = 1
        if isinstance(v[0], np.dtypes.VoidDType):  # pylint: disable=no-member
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

    __DTYPE_TO_MPITYPE[dtype] = mpi_type
    __MPIPY2F_TO_DTYPE[mpi_type.py2f()] = dtype

    return mpi_type


def type_create_resized(oldtype, lb, extent):
    """Wrapper for MPI_Type_create_resized. Create a datatype
    with a new lower bound and extent from an existing datatype.
    When creating a new, user-defined datatype using `numpy.dtype`
    arg `align` must be set to True. Otherwise user should use this function
    with `lb` set to 0 and `extent` set to dtype.itemsize.

    Args:
        oldtype (MPI_Datatype): input datatype.
        lb (int): new lower bound of datatype.
        extent (int): new extent of datatype.

    Returns:
        MPI_Datatype: new resized datatype.
    """
    mpi_type = oldtype.Create_resized(lb, extent)
    dtype = __MPIPY2F_TO_DTYPE[oldtype.py2f()]
    __DTYPE_TO_MPITYPE[dtype] = mpi_type
    __MPIPY2F_TO_DTYPE[mpi_type.py2f()] = dtype
    return mpi_type


def _dtype_to_mpitype(dtype):
    return __DTYPE_TO_MPITYPE[dtype]


def type_free(dtype):
    """Wrapper for MPI_Type_free. Frees the datatype.

    Args:
        dtype (MPI_Datatype): datatype that is freed

    Returns:
        None
    """
    dtype.Free()
