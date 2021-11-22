#ifndef IO_CUH
#define IO_CUH

#include "mpi.cuh"
#include <stdio.h>

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
#define MPI_ERR_FILE                    1       // Invalid file handle
#define MPI_ERR_NOT_SAME                2       // Collective argument not identical on all processes, 
                                                //     or collective routines called in a different order by different processes
#define MPI_ERR_AMODE                   3       // Error related to the amode passed to MPI_FILE_OPEN
#define MPI_ERR_UNSUPPORTED_DATAREP     4       // Unsupported datarep passed to MPI_FILE_SET_VIEW
#define MPI_ERR_UNSUPPORTED_OPERATION   5       // Unsupported operation, such as seeking on a file which supports sequential access only
#define MPI_ERR_NO_SUCH_FILE            6       // File does not exist
#define MPI_ERR_FILE_EXISTS             7       // File exists
#define MPI_ERR_BAD_FILE                8       // Invalid file name (e.g., path name too long)
#define MPI_ERR_ACCESS                  9       // Permission denied
#define MPI_ERR_NO_SPACE                10      // Not enough space
#define MPI_ERR_QUOTA                   11      // Quota exceeded
#define MPI_ERR_READ_ONLY               12      // Read-only file or file system
#define MPI_ERR_FILE_IN_USE             13      // File operation could not be completed, as the file is currently open by some process
#define MPI_ERR_DUP_DATAREP             14      // Conversion functions could not be registered because a data representation identifier 
                                                //      that was already defined was passed to MPI_REGISTER_DATAREP
#define MPI_ERR_CONVERSION              15      // An error occurred in a user supplied data conversion function.
#define MPI_ERR_IO                      16      // Other I/O error


/**
 * seek position
 * see documentation p521
 */
#define MPI_SEEK_SET    1
#define MPI_SEEK_CUR    2
#define MPI_SEEK_END    3

#define MPI_Offset      int
#define MPI_FILE_NULL nullptr

#define I_READY     0
#define I_FSEEK     1
#define I_FFLUSH    2
#define I_FCLOSE    3
#define I_FREAD     4
#define I_FWRITE    5
#define I_FOPEN     6
#define I_FDELETE   7

#define I_FOPEN_MODE_RD        0
#define I_FOPEN_MODE_RW        1
#define I_FOPEN_MODE_WD        2
#define I_FOPEN_MODE_RW_APPEND 3
#define I_FOPEN_MODE_WD_APPEND 4

#define BLOCK_NOT_IN       0
#define BLOCK_IN_CLEAN     1
#define BLOCK_IN_DIRTY     2

#define INIT_BUFFER_BLOCK_NUM   10
#define INIT_BUFFER_BLOCK_SIZE  64


struct MPI_Status;
namespace gpu_mpi { 
}
    struct MPI_Info{
        // todo
    };    
    
    struct MPI_File{
        MPI_Comm    comm;
        int*        seek_pos;
        int         amode;
        FILE*       file;
        void**      buffer; // points to the buffer of the file blocks
        int*        status; // status of each block: NOT_IN/ CLEAN/ DIRTY
        int*        num_blocks;   // number of blocks
        const char* filename;
    };

    // see documentation p493
    __device__ int MPI_File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);
    // see documentation p496
    __device__ int MPI_File_close(MPI_File *fh);
    // see documentation p520
    __device__ int MPI_File_seek(MPI_File fh, MPI_Offset offset, int whence);
    // mpi3.1 p516
    __device__ int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __device__ int MPI_File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);

    __device__ int MPI_File_read_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
    __device__ int MPI_File_write_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);

    // see documentation p521
    __device__ int MPI_File_get_position(MPI_File fh, MPI_Offset *offset);
    // see documentation p521
    __device__ int MPI_File_get_size(MPI_File fh, MPI_Offset *size);
    // see documentation p496
    __device__ int MPI_File_delete(const char *filename, MPI_Info info);


#endif
