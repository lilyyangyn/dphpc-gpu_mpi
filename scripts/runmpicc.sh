export GPU_MPI_PROJECT=/home/$USER/dphpc-gpu_mpi
export PATH=$PATH:/usr/local/cuda-11.3/bin/

Help()
{
   # Display Help
    echo "-g -b -s to specify gpumpi params, default 4,1,8192"
    echo "-f to specify .c file"        
}
# while getopts :h option; do
#     case $option in
#         h) # display Help
#             Help
#             exit;;
#     esac
# done

scriptdir=$(readlink -f $(dirname "$0"))


vg=4
vb=1
vs=8192
# while test $# -gt 0; do
while getopts ":h:g:b:s:f:" flag; do
    case "${flag}" in
        h)
            Help
            exit;;
        g) vg=${OPTARG};;
        b) vb=${OPTARG};;
        s) vs=${OPTARG};;
        f) input=${OPTARG}
            SOURCE_DIR=$(readlink -f $(dirname "${OPTARG}"));;
        \?) # incorrect option
         echo "Error: Invalid option"
         Help
         exit;;
    esac
done
echo "$vg, $vb, $vs, $input"
if [ -z "$input" ]; then
    echo "
    ---------- no input source file, default toolchain_tests/pi ----------
    "
    SOURCE_DIR="$GPU_MPI_PROJECT/toolchain_tests/pi"
    input="$GPU_MPI_PROJECT/toolchain_tests/pi"
else
    echo "
    ---------------------- using sourcefile $input ---------------------------
    "
fi

OBJECT_DIR="$GPU_MPI_PROJECT/build/toolchain_tests/"
mkdir -p $OBJECT_DIR
OBJECT="$OBJECT_DIR$(basename "$input" | sed 's/\(.*\)\..*/\1/')"

echo "source dir: $SOURCE_DIR"
echo "object dir: $OBJECT"
pushd .
cd $OBJECT_DIR

echo "
------compiling with gpumpicc.py-----
"

if ! /home/$USER/miniconda3/bin/python3.7 $GPU_MPI_PROJECT/build/scripts/gpumpicc.py $SOURCE_DIR/$(basename $input) -o $OBJECT; then
    echo "
    COMPILE ERROR!
    " >&2
    exit
fi

echo "
------------running---------------
"
$OBJECT ---gpumpi -g $vg -b $vb -s $vs
popd