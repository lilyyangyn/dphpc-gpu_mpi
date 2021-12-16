for x in 1 2 3 4 5 6 7 8 9 10
do
    python3 /home/xindzuo/dphpc-gpu_mpi/scripts/replace_batch_size.py
    cd /home/xindzuo/dphpc-gpu_mpi/build
    bash /home/xindzuo/dphpc-gpu_mpi/scripts/recompile.sh
    cd /home/xindzuo/dphpc-gpu_mpi/toolchain_tests/basicIO
    export GPU_MPI_PROJECT=/home/xindzuo/dphpc-gpu_mpi
    export PATH=$PATH:/usr/local/cuda-11.3/bin/
    /home/xindzuo/miniconda3/bin/python3.7 /home/xindzuo/dphpc-gpu_mpi/build/scripts/gpumpicc.py best_block_size.c
    ./a.out ---gpumpi -g 4 -b 1
done