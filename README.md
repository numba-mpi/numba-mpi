# <img src="https://raw.githubusercontent.com/numba-mpi/numba-mpi/main/.github/numba_mpi_logo.svg" width=128 height=142 alt="numba-mpi logo"> numba-mpi

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

### Overview
numba-mpi provides Python wrappers to the C MPI API callable from within [Numba JIT-compiled code](https://numba.readthedocs.io/en/stable/user/jit.html) (@jit mode). For an outline of the project, rationale, architecture, and features, refer to: [numba-mpi paper in SoftwareX (open access)](https://www.sciencedirect.com/science/article/pii/S235271102400267X) (please cite if numba-mpi is used in your research).

Support is provided for a subset of MPI routines covering: `size`/`rank`, `send`/`recv`, `allreduce`, `reduce`, `bcast`, `scatter`/`gather` & `allgather`, `barrier`, `wtime`
and basic asynchronous communication with `isend`/`irecv` (only for contiguous arrays); for request handling including `wait`/`waitall`/`waitany` and `test`/`testall`/`testany`.

The API uses NumPy and supports both numeric and character datatypes (e.g., `broadcast`).
Auto-generated docstring-based API docs are published on the web: https://numba-mpi.github.io/numba-mpi

Packages can be obtained from
  [PyPI](https://pypi.org/project/numba-mpi),
  [Conda Forge](https://anaconda.org/conda-forge/numba-mpi),
  [Arch Linux](https://aur.archlinux.org/packages/python-numba-mpi)
  or by invoking `pip install git+https://github.com/numba-mpi/numba-mpi.git`.

numba-mpi is a pure-Python package.
The codebase includes a test suite used through the GitHub Actions workflows ([thanks to mpi4py's setup-mpi](https://github.com/mpi4py/setup-mpi)!)
for automated testing on: Linux ([MPICH](https://www.mpich.org/), [OpenMPI](https://www.open-mpi.org/doc/)
& [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html)),
macOS ([MPICH](https://www.mpich.org/) & [OpenMPI](https://www.open-mpi.org/doc/)) and
Windows ([MS MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)). Note, that some of those
combinations may not be fully supported yet - see [Known Issues](#known-issues) for more information.

Features that are not implemented yet include (help welcome!):
- support for non-default communicators
- support for `MPI_IN_PLACE` in `[all]gather`/`scatter` and `allreduce`
- support for `MPI_Type_create_struct` (Numpy structured arrays)
- ...

### Hello world send/recv example:
```python
import numba, numba_mpi, numpy

@numba.jit()
def hello():
    src = numpy.array([1., 2., 3., 4., 5.])
    dst_tst = numpy.empty_like(src)

    if numba_mpi.rank() == 0:
        numba_mpi.send(src, dest=1, tag=11)
    elif numba_mpi.rank() == 1:
        numba_mpi.recv(dst_tst, source=0, tag=11)

hello()
```

### Example comparing numba-mpi vs. mpi4py performance:

The example below compares `Numba`+`mpi4py` vs. `Numba`+`numba-mpi` performance.
The sample code estimates $\pi$ by numerical integration of $\int_0^1 (4/(1+x^2))dx=\pi$
dividing the workload into `n_intervals` handled by separate MPI processes
and then obtaining a sum using `allreduce` (see, e.g., analogous [Matlab docs example](https://www.mathworks.com/help/parallel-computing/numerical-estimation-of-pi-using-message-passing.html)).
The computation is carried out in a JIT-compiled function `get_pi_part()` and is repeated
`N_TIMES`. The repetitions and the MPI-handled reduction are done outside or
inside of the JIT-compiled block for `mpi4py` and `numba-mpi`, respectively.
Timing is repeated `N_REPEAT` times and the minimum time is reported.
The generated plot shown below depicts the speedup obtained by replacing `mpi4py`
with `numba_mpi`, plotted as a function of `N_TIMES / n_intervals` - the number of MPI calls per
interval. The speedup, which stems from avoiding roundtrips between JIT-compiled
and Python code is significant (150%-300%) in all cases. The more often communication
is needed (smaller `n_intervals`), the larger the measured speedup. Note that nothing
in the actual number crunching (within the `get_pi_part()` function) or in the employed communication logic
(handled by the same MPI library) differs between the `mpi4py` or `numba-mpi` solutions.
These are the overhead of `mpi4py` higher-level abstractions and the overhead of
repeatedly entering and leaving the JIT-compiled block if using `mpi4py`, which can be
eliminated by using `numba-mpi`, and which the measured differences in execution time
stem from.
```python
import timeit, mpi4py, numba, numpy as np, numba_mpi

N_TIMES = 10000
RTOL = 1e-3

@numba.jit
def get_pi_part(n_intervals=1000000, rank=0, size=1):
    h = 1 / n_intervals
    partial_sum = 0.0
    for i in range(rank + 1, n_intervals, size):
        x = h * (i - 0.5)
        partial_sum += 4 / (1 + x**2)
    return h * partial_sum

@numba.jit
def pi_numba_mpi(n_intervals):
    pi = np.array([0.])
    part = np.empty_like(pi)
    for _ in range(N_TIMES):
        part[0] = get_pi_part(n_intervals, numba_mpi.rank(), numba_mpi.size())
        numba_mpi.allreduce(part, pi, numba_mpi.Operator.SUM)
        assert abs(pi[0] - np.pi) / np.pi < RTOL

def pi_mpi4py(n_intervals):
    pi = np.array([0.])
    part = np.empty_like(pi)
    for _ in range(N_TIMES):
        part[0] = get_pi_part(n_intervals, mpi4py.MPI.COMM_WORLD.rank, mpi4py.MPI.COMM_WORLD.size)
        mpi4py.MPI.COMM_WORLD.Allreduce(part, (pi, mpi4py.MPI.DOUBLE), op=mpi4py.MPI.SUM)
        assert abs(pi[0] - np.pi) / np.pi < RTOL

plot_x = [x for x in range(1, 11)]
plot_y = {'numba_mpi': [], 'mpi4py': []}
for x in plot_x:
    for impl in plot_y:
        plot_y[impl].append(min(timeit.repeat(
            f"pi_{impl}(n_intervals={N_TIMES // x})",
            globals=locals(),
            number=1,
            repeat=10
        )))

if numba_mpi.rank() == 0:
    from matplotlib import pyplot
    pyplot.figure(figsize=(8.3, 3.5), tight_layout=True)
    pyplot.plot(plot_x, np.array(plot_y['mpi4py'])/np.array(plot_y['numba_mpi']), marker='o')
    pyplot.xlabel('number of MPI calls per interval')
    pyplot.ylabel('mpi4py/numba-mpi wall-time ratio')
    pyplot.title(f'mpiexec -np {numba_mpi.size()}')
    pyplot.grid()
    pyplot.savefig('readme_plot.svg')
```

![plot](https://github.com/numba-mpi/numba-mpi/releases/download/tip/readme_plot.png)

### Known Issues

**NOTE**: Issues listed below only relate to combinations of platforms and MPI distributions that we target to support, but due to various reason are currently not working and are temporarily excluded from automated testing:

- tests on Ubuntu 2024.4 that use MPICH are not run due to failures caused by newer version of MPICH (`4.2.0`); note, that previous tests ran
using version `4.0.2` of MPICH (that is installed by default on Ubuntu 2022.4 using `apt`) were passing (see [related issue](https://github.com/numba-mpi/numba-mpi/issues/162) - TODO #162),
- tests on Intel MacOS (v13) that use OpenMPI are currently not run due to failures being under investigation (see [related issue](https://github.com/numba-mpi/numba-mpi/issues/163)  - TODO #163),
- `numba-mpi` currently does not support ARM-based MacOS, due to required code improvement (see [related issue](https://github.com/numba-mpi/numba-mpi/issues/164) - TODO #164).

### MPI resources on the web:

- MPI standard and general information:
    - https://www.mpi-forum.org/docs
    - https://en.wikipedia.org/wiki/Message_Passing_Interface
- MPI implementations:
    - OpenMPI: https://www.open-mpi.org
    - MPICH: https://www.mpich.org
    - MS MPI: https://learn.microsoft.com/en-us/message-passing-interface
    - Intel MPI: https://intel.com/content/www/us/en/developer/tools/oneapi/mpi-library-documentation.html
- MPI bindings:
    - Python: https://mpi4py.readthedocs.io
    - Python/JAX: https://mpi4jax.readthedocs.io
    - Julia: https://juliaparallel.org/MPI.jl
    - Rust: https://docs.rs/mpi
    - C++: https://boost.org/doc/html/mpi.html
    - R: https://cran.r-project.org/web/packages/Rmpi

### Acknowledgements:

We thank [all contributors](https://github.com/numba-mpi/numba-mpi/graphs/contributors) and users who reported feedback to the project
  through [GitHub issues](https://github.com/numba-mpi/numba-mpi/issues).

Development of numba-mpi has been supported by the [Polish National Science Centre](https://ncn.gov.pl/en) (grant no. 2020/39/D/ST10/01220),
  the [Max Planck Society](https://www.mpg.de/en) and the [European Union](https://erc.europa.eu/) (ERC, EmulSim, 101044662).
We further acknowledge Poland’s high-performance computing infrastructure [PLGrid](https://plgrid.pl) (HPC Centers: [ACK Cyfronet AGH](https://www.cyfronet.pl/en))
  for providing computer facilities and support within computational grant no. PLG/2023/016369.
