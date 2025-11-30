"""checks if listings from the paper run OK"""

import pathlib

import pytest

import numba_mpi


@pytest.mark.parametrize(
    "files",
    (
        ("hello.py", "mpi4py_with_error.py", "numba_mpi.py", "timing.py"),
        ("exchange.py",),
        ("test.py",),
        pytest.param(
            ("py-pde.py",),
            marks=pytest.mark.skipif(numba_mpi.size() != 2, reason="hardcoded"),
        ),
    ),
)
@pytest.mark.skipif(numba_mpi.size() == 1, reason="listings assume more than 1 worker")
def test_paper_listings(files):
    """concatenates code from all files and executes it in global scope"""
    code = ""
    for file in files:
        with open(
            pathlib.Path(__file__).parent / "paper_listings" / file, encoding="utf-8"
        ) as stream:
            code += "".join(stream.readlines())
        code += "\n"
    try:
        exec(code, globals())  # pylint: disable=exec-used
    except Exception as ex:
        for line_num, line in enumerate(code.split("\n"), start=1):
            print(line_num, ":", line)
        raise ex
