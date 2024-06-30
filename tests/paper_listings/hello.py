import numba, timeit

@numba.jit(nopython=True)
def get_pi_part(n_intervals=1000000, rank=0, size=1):
    h = 1 / n_intervals
    partial_sum = 0.0
    for i in range(rank + 1, n_intervals, size):
        x = h * (i - 0.5)
        partial_sum += 4 / (1 + x**2)
    return h * partial_sum

time = lambda fun: min(timeit.repeat(fun, number=1, repeat=5))
print(f"speedup: {time(get_pi_part.py_func) / time(get_pi_part):.3g}")
