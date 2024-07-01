import mpi4py, numba, numpy as np

N_TIMES = 10000
RTOL = 1e-3

def pi_mpi4py(n_intervals):
    pi = np.array([0.])
    part = np.empty_like(pi)
    for _ in range(N_TIMES):
        part[0] = get_pi_part(n_intervals,
            mpi4py.MPI.COMM_WORLD.rank, mpi4py.MPI.COMM_WORLD.size)
        mpi4py.MPI.COMM_WORLD.Allreduce(part,
            (pi, mpi4py.MPI.DOUBLE), op=mpi4py.MPI.SUM)
        assert abs(pi[0] - np.pi) / np.pi < RTOL

try:
    numba.jit(pi_mpi4py, nopython=True)(1)
except Exception as ex:
    print(ex)
