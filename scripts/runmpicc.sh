
export GPU_MPI_PROJECT=/home/$USER/dphpc-gpu_mpi
export PATH=$PATH:/usr/local/cuda-11.3/bin/
scriptdir=$(readlink -f $(dirname "$0"))
if [ -z "$1" ]; then
    echo "---------- no input source file, default toolchain_tests/pi ----------"
    SOURCE_DIR="$GPU_MPI_PROJECT/toolchain_tests/pi"
else
    echo "---------------------- using sourcefile $1 ---------------------------"
    SOURCE_DIR=$(readlink -f $(dirname "$1"))
fi

echo $SOURCE_DIR
pushd .
cd $scriptdir

#TODO: compile sourcefile from input
/home/$USER/miniconda3/bin/python3.7 $GPU_MPI_PROJECT/build/scripts/gpumpicc.py $SOURCE_DIR/cpi.c -o $SOURCE_DIR/pi

$SOURCE_DIR/pi ---gpumpi -g 4 -b 1 -s 8192
popd