# <img src="https://raw.githubusercontent.com/numba-mpi/numba-mpi/main/.github/numba_mpi_logo.svg" style="height:50pt" alt="numba-mpi logo"> numba-mpi

[![Python 3](https://img.shields.io/static/v1?label=Python&logo=Python&color=3776AB&message=3)](https://www.python.org/)
[![LLVM](https://img.shields.io/static/v1?label=LLVM&logo=LLVM&color=gold&message=Numba)](https://numba.pydata.org)
[![Linux OK](https://img.shields.io/static/v1?label=Linux&logo=Linux&color=yellow&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Linux)
[![macOS OK](https://img.shields.io/static/v1?label=macOS&logo=Apple&color=silver&message=%E2%9C%93)](https://en.wikipedia.org/wiki/macOS)
[![Windows OK](https://img.shields.io/static/v1?label=Windows&logo=Windows&color=white&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Windows)
[![Github Actions Status](https://github.com/numba-mpi/numba-mpi/workflows/tests+pypi/badge.svg?branch=main)](https://github.com/numba-mpi/numba-mpi/actions/workflows/tests+pypi.yml)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/numba-mpi/numba-mpi/graphs/commit-activity)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI version](https://badge.fury.io/py/numba-mpi.svg)](https://pypi.org/project/numba-mpi)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/numba-mpi/badges/version.svg)](https://anaconda.org/conda-forge/numba-mpi)
[![DOI](https://zenodo.org/badge/316911228.svg)](https://zenodo.org/badge/latestdoi/316911228)

#### Numba @njittable MPI wrappers
- covering: `size`/`rank`, `send`/`recv`, `allreduce`, `bcast`, `scatter`/`gather`, `barrier` and `wtime`
- not yet implemented: support for non-default communicators, ...
- API based on NumPy and supporting numeric and character datatypes 
- auto-generated docstring-based API docs on the web: https://numba-mpi.github.io/numba-mpi
- pure-Python implementation with packages available on [PyPI](https://pypi.org/project/numba-mpi) and [Conda Forge](https://anaconda.org/conda-forge/numba-mpi)
- CI-tested on: Linux ([MPICH](https://www.mpich.org/), [OpenMPI](https://www.open-mpi.org/doc/) & [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html)), macOS ([MPICH](https://www.mpich.org/) & [OpenMPI](https://www.open-mpi.org/doc/)) and Windows ([MS MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi))

Hello world example:
```python
import numba, numba_mpi, numpy

@numba.njit()
def hello():
  print(numba_mpi.rank())
  print(numba_mpi.size())

  src = numpy.array([1., 2., 3., 4., 5.])
  dst_tst = numpy.empty_like(src)

  if numba_mpi.rank() == 0:
    numba_mpi.send(src, dest=1, tag=11)
  elif numba_mpi.rank() == 1:
    numba_mpi.recv(dst_tst, source=0, tag=11)

hello()
```

For information on MPI, see:
- https://www.mpi-forum.org/docs/
- https://mpi4py.readthedocs.io/en/stable/
