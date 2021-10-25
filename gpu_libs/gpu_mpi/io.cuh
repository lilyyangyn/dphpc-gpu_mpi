#ifndef IO_CUH
#define IO_CUH

// See documentation: https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf

namespace gpu_mpi {

    __device__ int MPI_FILE_OPEN(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);

}

#endif