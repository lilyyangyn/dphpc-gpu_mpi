#include "io.cuh"

namespace gpu_mpi {

    struct MPI_Info{
        // todo
    };    
    
    struct MPI_File{
        // todo
    };

    __device__ int MPI_FILE_OPEN(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh){
        // todo
        return 0;
    }

}