#!/bin/bash

# ###################################################################
# Slurm script to run tests on Bowie Supercomputer Cluster          #
# Reference: https://open-atmos-krk.github.io/projects/hpc-diy.html #
# ###################################################################

#SBATCH --output=/mnt/cluster-workspace/gha-runner/logs/stdout.txt
#SBATCH --error=/mnt/cluster-workspace/gha-runner/logs/joberr.txt

source $HOME/.setup-pyenv
pyenv shell $1
python --version
python -m venv ./.venv
source ./.venv/bin/activate
export PATH=/mnt/cluster-workspace/bin/openmpi-5.0.10/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/cluster-workspace/bin/openmpi-5.0.10/lib

pip install --only-binary=:all: -e .[tests,CI_version_pins]
mpiexec --mca btl_tcp_if_include eth0 -n 2 python -m pytest -s -v --durations=10 -p no:unraisableexception -We tests/api
exit_code=$?
echo "Tests completed with exit code $exit_code"
echo $exit_code > /mnt/cluster-workspace/gha-runner/test_exit_code.txt
