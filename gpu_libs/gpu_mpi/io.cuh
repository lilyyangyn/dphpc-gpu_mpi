#ifndef IO_CUH
#define IO_CUH

#include "mpi.cuh"

// See documentation: https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf

namespace gpu_mpi {

    /**
     * amode constants
     */
    #define MPI_MODE_RDONLY             1
    #define MPI_MODE_RDWR               2
    #define MPI_MODE_WRONLY             4
    #define MPI_MODE_CREATE             8
    #define MPI_MODE_EXCL               16
    #define MPI_MODE_DELETE_ON_CLOSE    32
    #define MPI_MODE_UNIQUE_OPEN        64
    #define MPI_MODE_SEQUENTIAL         128
    #define MPI_MODE_APPEND             256

    struct MPI_Info{
        // todo
    };    
    
    struct MPI_File{
        // todo
    };

    __device__ int MPI_FILE_OPEN(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);

}

#endif