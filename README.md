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
[![AUR package](https://repology.org/badge/version-for-repo/aur/python:numba-mpi.svg)](https://aur.archlinux.org/packages/python-numba-mpi)
[![DOI](https://zenodo.org/badge/316911228.svg)](https://zenodo.org/badge/latestdoi/316911228)

## Overview
numba-mpi provides Numba @njittable MPI wrappers:
- covering: `size`/`rank`, `send`/`recv`, `allreduce`, `bcast`, `scatter`/`gather` & `allgather`, `barrier` and `wtime`
- basic asynchronous communication with `isend`/`irecv` (only for contiguous arrays); for request handling including `wait`/`waitall`/`waitany` and `test`/`testall`/`testany`
- not yet implemented: support for non-default communicators, ...
- API based on NumPy and supporting numeric and character datatypes 
- auto-generated docstring-based API docs on the web: https://numba-mpi.github.io/numba-mpi
- pure-Python implementation with packages available at [PyPI](https://pypi.org/project/numba-mpi), [Conda Forge](https://anaconda.org/conda-forge/numba-mpi) and for [Arch Linux](https://aur.archlinux.org/packages/python-numba-mpi)
- CI-tested on: Linux ([MPICH](https://www.mpich.org/), [OpenMPI](https://www.open-mpi.org/doc/) & [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html)), macOS ([MPICH](https://www.mpich.org/) & [OpenMPI](https://www.open-mpi.org/doc/)) and Windows ([MS MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi))

## Hello world send/recv example:
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

## numba-mpi vs. mpi4py:

The example below compares Numba + mpi4py vs. Numba + numba-mpi performance by
computing $\pi$ by integration of $4/(1+x^2)$ between 0 and 1 divided into
`N_INTERVALS` handled by separate MPI processes and then obtaining a sum
using `allreduce`. The computation is repeated `N_TIMES` within a JIT-compiled
loop. Timing is repeated `N_REPEAT` times and the minimum time is reported.
```python
import timeit, mpi4py, numba, numpy, numba_mpi

N_INTERVALS = 5000
N_TIMES = 50000
N_REPEAT = 10
RTOL = 1e-3

@numba.njit
def compute_pi_part(out, rank, size):
    h = 1 / N_INTERVALS
    partial_sum = 0.0
    for i in range(rank + 1, N_INTERVALS, size):
        x = h * (i - 0.5)
        partial_sum += 4 / (1 + x**2)
    out[0] = h * partial_sum

@numba.njit
def pi_numba_mpi():
    pi = numpy.array(0.0)
    pi_part = numpy.empty(1)
    for _ in range(N_TIMES):
        compute_pi_part(pi_part, numba_mpi.rank(), numba_mpi.size())
        numba_mpi.allreduce(pi_part, pi, numba_mpi.Operator.SUM)
        assert abs(pi - numpy.pi) / numpy.pi < RTOL
    return pi

def pi_mpi4py():
    pi = numpy.array(0.0)
    pi_part = numpy.empty(1)
    for _ in range(N_TIMES):
        compute_pi_part(pi_part, mpi4py.MPI.COMM_WORLD.rank, mpi4py.MPI.COMM_WORLD.size)
        mpi4py.MPI.COMM_WORLD.Reduce(
            pi_part, [pi, mpi4py.MPI.DOUBLE], op=mpi4py.MPI.SUM, root=0
        )
        assert abs(pi - numpy.pi) / numpy.pi < RTOL
    return pi

for fun in ("pi_numba_mpi()", "pi_mpi4py()"):
    time = min(timeit.repeat(fun, globals=locals(), number=1, repeat=N_REPEAT))
    if numba_mpi.rank() == 0:
        print(f"{N_TIMES} x {fun}:\t{time:.2} s\t(MPI comm size = {numba_mpi.size()})")
```

## Information on MPI

- MPI standard and general information:
    - https://www.mpi-forum.org/docs
    - https://en.wikipedia.org/wiki/Message_Passing_Interface
- MPI implementations:
    - https://www.open-mpi.org
    - https://www.mpich.org
    - https://learn.microsoft.com/en-us/message-passing-interface
    - https://intel.com/content/www/us/en/developer/tools/oneapi/mpi-library-documentation.html
- MPI bindings:
    - Python: https://mpi4py.readthedocs.io
    - Julia: https://juliaparallel.org/MPI.jl
    - Rust: https://docs.rs/mpi
    - C++: https://boost.org/doc/html/mpi.html
    - R: https://cran.r-project.org/web/packages/Rmpi

#### Acknowledgement:

Development of numba-mpi has been supported by the [Polish National Science Centre](https://ncn.gov.pl/en) (grant no. 2020/39/D/ST10/01220).
