import math
import time

import numba
import numpy as np
from mpi4py import MPI

x0, x1, w = -2.0, +2.0, 640 * 2 * 2
y0, y1, h = -1.5, +1.5, 480 * 2 * 2
dx = (x1 - x0) / w
dy = (y1 - y0) / h

c = complex(0, 0.65)


@numba.njit(fastmath=True)
def subdomain(span, rank, size):
    if rank >= size:
        raise ValueError("rank >= size")

    n_max = math.ceil(span / size)
    start = n_max * rank
    stop = start + (n_max if start + n_max <= span else span - start)
    return start, stop


@numba.njit(fastmath=True)
def julia(x, y):
    z = complex(x, y)
    n = 255
    while abs(z) < 3 and n > 1:
        z = z**2 + c
        n -= 1
    return n


@numba.njit(fastmath=True)
def julia_line(k, line):
    y = y1 - k * dy
    for j in range(w):
        x = x0 + j * dx
        line[j] = julia(x, y)
    return line


comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

domain_chunk = subdomain(span=h, rank=rank, size=size)
image = np.empty((w, domain_chunk[1] - domain_chunk[0]), dtype=np.int32)

#
start = time.time()
for n in range(domain_chunk[1] - domain_chunk[0]):
    julia_line(n + domain_chunk[0], image[:, n])
result = comm.allgather(image)
print(":: ", time.time() - start)
#

if rank == 0:
    image = np.empty((w, h), dtype=np.int32)
    offset = 0
    for item in result:
        image[:, offset : offset + item.shape[1]] = item
        offset += item.shape[1]
    np.save(f"output_mpi4py_{rank=}_{size=}.npy", image)
