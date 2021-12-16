# export GPU_MPI_PROJECT=/home/xindzuo/dphpc-gpu_mpi
# export PATH=$PATH:/usr/local/cuda-11.3/bin/
# /home/xindzuo/miniconda3/bin/python3.7 /home/xindzuo/dphpc-gpu_mpi/build/scripts/gpumpicc.py wr_time_multi_sameblock.c
# ./a.out ---gpumpi -g 150 -b 1


for x in 1 2 4 8 16 32 64 128
do
    export GPU_MPI_PROJECT=/home/xindzuo/dphpc-gpu_mpi
    export PATH=$PATH:/usr/local/cuda-11.3/bin/
    /home/xindzuo/miniconda3/bin/python3.7 /home/xindzuo/dphpc-gpu_mpi/build/scripts/gpumpicc.py wr_time_multi_diffblock.c
    ./a.out ---gpumpi -g $x -b 1
done