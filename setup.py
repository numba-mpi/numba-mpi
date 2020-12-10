from distutils.core import setup

def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

setup(
    name='numba-mpi',
    version='0.1.0',
    packages=[
        'numba_mpi'],
    license='GPL v3',
    long_description='Numba @njittable MPI wrappers tested on Linux, macOS and Windows',
    install_requires=parse_requirements('requirements.txt')
)
