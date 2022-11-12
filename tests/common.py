# pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring
import numpy as np

MPI_SUCCESS = 0

data_types_real = [
    int,
    np.int32,
    np.int64,
    float,
    np.float64,
    np.double,
]
data_types_complex = [complex, np.complex64, np.complex128]
data_types = data_types_real + data_types_complex
