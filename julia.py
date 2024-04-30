import subprocess
import sys
import time

import numpy as np
from matplotlib import pyplot

if __name__ == "__main__":
    x = [1, 2, 4]
    y = []
    for n in x:
        print(f"spawning mpiexec -n {n}...", end="", file=sys.stderr)
        start = time.time()
        subprocess.run(["mpiexec", "-n", str(n), sys.executable, "julia_mpi4py.py"])
        y.append(time.time() - start)
        print(f" done in {y[-1]}s", file=sys.stderr)

        if n != 1:
            np.testing.assert_equal(
                actual=np.load(f"output_mpi4py_rank={0}_size={n}.npy"),
                desired=np.load(f"output_mpi4py_rank={0}_size={1}.npy"),
            )
        else:
            pyplot.imshow(np.load(f"output_mpi4py_rank={0}_size={1}.npy"))
            pyplot.savefig("image.pdf")

    pyplot.clf()
    pyplot.plot(x, y, marker="o")
    pyplot.ylabel("wall time [s]")
    pyplot.xlabel("number of MPI workers")
    pyplot.savefig("plot.pdf")
