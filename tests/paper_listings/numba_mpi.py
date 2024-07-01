import numba_mpi

@numba.jit(nopython=True)
def pi_numba_mpi(n_intervals):
    pi = np.array([0.])
    part = np.empty_like(pi)
    for _ in range(N_TIMES):
        part[0] = get_pi_part(n_intervals, numba_mpi.rank(), numba_mpi.size())
        numba_mpi.allreduce(part, pi, numba_mpi.Operator.SUM)
        assert abs(pi[0] - np.pi) / np.pi < RTOL
