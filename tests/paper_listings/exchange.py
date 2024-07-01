import numpy as np
import numba_mpi as mpi
import numba

@numba.jit(nopython=True)
def exchange_data(src_data):
    dst_data = np.empty_like(src_data)
    reqs = np.zeros((2,), dtype=mpi.RequestType)

    if mpi.rank() == 0:
        status, reqs[0:1] = mpi.isend(src_data, dest=1, tag=11)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = mpi.irecv(dst_data, source=1, tag=22)
        assert status == mpi.SUCCESS
    elif mpi.rank() == 1:
        status, reqs[0:1] = mpi.isend(src_data, dest=0, tag=22)
        assert status == mpi.SUCCESS
        status, reqs[1:2] = mpi.irecv(dst_data, source=0, tag=11)
        assert status == mpi.SUCCESS

    mpi.waitall(reqs)
    return dst_data

if mpi.rank() < 2:
    src_data = np.random.rand(10000)
    dst_data = exchange_data(src_data)
