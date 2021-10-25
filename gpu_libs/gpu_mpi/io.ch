#include "io.cuh"

namespace gpu_mpi {

    __device__ int MPI_FILE_OPEN(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh){
        // todo
        return 0;
    }

}