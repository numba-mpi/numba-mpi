from distutils.core import setup
from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt')

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
#reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='numba-mpi',
    version='0.1.0',
    packages=[
        'numba_mpi'],
    license='GPL v3',
    long_description='Numba @njittable MPI wrappers tested on Linux, macOS and Windows',
    install_requires=install_reqs
)
