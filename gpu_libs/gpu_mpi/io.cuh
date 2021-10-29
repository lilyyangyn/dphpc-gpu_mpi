#ifndef IO_CUH
#define IO_CUH

#include "mpi.cuh"

// See documentation: https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf

/**
 * amode constants 
 * see documentation p494
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


/**
 * error code
 * see documentation p556 table 13.3
 */
#define MPI_ERR_FILE                    1
#define MPI_ERR_NOT_SAME                2
#define MPI_ERR_AMODE                   3
#define MPI_ERR_UNSUPPORTED_DATAREP     4
#define MPI_ERR_UNSUPPORTED_OPERATION   5
#define MPI_ERR_NO_SUCH_FILE            6
#define MPI_ERR_FILE_EXISTS             7
#define MPI_ERR_BAD_FILE                8
#define MPI_ERR_ACCESS                  9
#define MPI_ERR_NO_SPACE                10
#define MPI_ERR_QUOTA                   11
#define MPI_ERR_READ_ONLY               12
#define MPI_ERR_FILE_IN_USE             13
#define MPI_ERR_DUP_DATAREP             14
#define MPI_ERR_CONVERSION              15
#define MPI_ERR_IO                      16


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