#ifndef IO_CUH
#define IO_CUH

namespace gpu_mpi {

    __device__ int MPI_FILE_OPEN(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);

}

#endif