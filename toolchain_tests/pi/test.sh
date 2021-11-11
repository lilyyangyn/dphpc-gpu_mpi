#!/bin/bash

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

export GPU_MPI_PROJECT=$(readlink -f .)
export MPICC=../../scripts/gpumpicc.py

make

./pi ---gpumpi -g 4 -b 1 -s 8192

