#ifndef IO_CUH
#define IO_CUH

#include "mpi.cuh"

// See documentation: https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf

namespace gpu_mpi {
    
    struct MPI_Info{
        // todo
    };    
    
    struct MPI_File{
        // todo
    };

    __device__ int MPI_FILE_OPEN(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);

}

#endif