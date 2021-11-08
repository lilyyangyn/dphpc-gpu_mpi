export GPU_MPI_PROJECT=/home/raliao/gpu_mpi
export PATH=$PATH:/usr/local/cuda-11.3/bin/
../../../miniconda3/bin/python3.7 ../../build/scripts/gpumpicc.py cpi.c 
./a.out ---gpumpi -g 4 -b 1 -s 8192