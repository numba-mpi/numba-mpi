# numba-mpi
Numba @njittable MPI wrappers tested on Linux, macOS and Windows

Hello world example:
```python
import numba, numba_mpi, numpy

@numba.njit()
def hello():
  print(numba_mpi.rank())
  print(numba_mpi.size())

  src = numpy.array([1., 2., 3., 4., 5.])
  dst_tst = numpy.empty_like(src)

  if mpi.rank() == 0:
    numba_mpi.send(src, dest=1, tag=11)
  elif mpi.rank() == 1:
    numba_mpi.recv(dst_tst, source=0, tag=11)
```
