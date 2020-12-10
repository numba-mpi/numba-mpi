from distutils.core import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt')

setup(
    name='numba-mpi',
    version='0.1.0',
    packages=[
        'numba_mpi'],
    license='GPL v3',
    long_description='Numba @njittable MPI wrappers tested on Linux, macOS and Windows',
    install_requires=[str(ir.req) for ir in install_reqs]
)
