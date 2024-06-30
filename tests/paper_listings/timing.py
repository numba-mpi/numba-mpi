from matplotlib import pyplot

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
    pyplot.plot(
        plot_x,
        np.array(plot_y['mpi4py'])/np.array(plot_y['numba_mpi']),
        marker='o'
    )
