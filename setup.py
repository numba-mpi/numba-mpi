""" the magick behind ``pip install ...`` """
from setuptools import find_packages, setup


def get_long_description():
    """returns contents of README.md file"""
    with open("README.md", "r", encoding="utf8") as file:
        long_description = file.read()
    return long_description


setup(
    name="numba-mpi",
    url="https://github.com/numba-mpi/numba-mpi",
    author="https://github.com/numba-mpi/numba-mpi/graphs/contributors",
    use_scm_version={
        "local_scheme": lambda _: "",
        "version_scheme": "post-release",
    },
    setup_requires=["setuptools_scm"],
    license="GPL v3",
    description="Numba @njittable MPI wrappers tested on Linux, macOS and Windows",
    install_requires=("numba", "numpy", "mpi4py"),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["numba_mpi", "numba_mpi.*"]),
    project_urls={
        "Tracker": "https://github.com/numba-mpi/numba-mpi/issues",
        "Documentation": "https://numba-mpi.github.io/numba-mpi",
        "Source": "https://github.com/numba-mpi/numba-mpi",
    },
)
