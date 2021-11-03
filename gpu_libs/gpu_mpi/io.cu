#include "io.cuh"

namespace gpu_mpi {

    __host__ __device__ int __open_file(const char *filename, int amode, MPI_File *fh){
        FILE* file = NULL;

        // check file existence
        // TODO: file=fopen(filename,"r");
        if(file == NULL){
            if(!(amode & MPI_MODE_CREATE)) {
                return MPI_ERR_NO_SUCH_FILE;
            }
            // No need to create new file as file will be created below
        }else{
            // TODO: fclose(file);
            if(amode & MPI_MODE_EXCL) {
                // File must not exist
                return MPI_ERR_FILE_EXISTS;
            }
        }

        const char *mode;
        if(amode & MPI_MODE_RDONLY){
            mode = "r";
        }else if(amode & MPI_MODE_RDWR){
            mode = "r+";
        }else if(amode & MPI_MODE_WRONLY){
            // TODO: append or overwrite???
            mode = "a";
        }
        
        // TODO: fh->file = fopen(filename, mode);

        return 0;
    }

    __device__ int __get_file_size(FILE* file){
        // fseek(file, 0L, SEEK_END);
        // return ftell(file);
        
        // todo: how to get the file size? how to talk to cpu?
        return 0;
    }

    __device__ int MPI_File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh){
        // check amode
        if (((amode & MPI_MODE_RDONLY) ? 1 : 0) + ((amode & MPI_MODE_RDWR) ? 1 : 0) +
            ((amode & MPI_MODE_WRONLY) ? 1 : 0) != 1) {
           // see documentation p495 line 7
           return MPI_ERR_AMODE;
        }
        if(((amode & MPI_MODE_CREATE) || (amode & MPI_MODE_EXCL)) && (amode & MPI_MODE_RDONLY)){
            // see documentation p495 line 8
            return MPI_ERR_AMODE;
        }
        if((amode & MPI_MODE_RDWR) && (amode & MPI_MODE_SEQUENTIAL)){
            // see documentation p495 line 9
            return MPI_ERR_AMODE;
        }

        // create MPI_FILE
        int rank;
        MPI_Comm_rank(comm, &rank);
        if(rank == 0){
            fh->amode = amode;
            fh->comm = comm;

            // initialize fh->file
            int status = __open_file(filename, amode, fh);
            if(status != 0) {
                // TODO: how to handel error code between different threads?
            }
            
            // initialize fh->seek_pos
            // TODO: MPI_MODE_UNIQUE_OPEN -> Only one seek_pos???
            int size;
            MPI_Comm_rank(comm, &size);
            fh->seek_pos = new int[size];
            int init_pos = 0;
            if(amode & MPI_MODE_APPEND){
                // In append mode: set pointer to end of file 
                // see documentation p494 line 42
                init_pos = __get_file_size(fh->file);
            }
            memset(fh->seek_pos, init_pos, sizeof(int) * size);
        }
        
        // TODO: check whether synchronization is in need
        __syncthreads();
        
        return 0;
    }

    __device__ int MPI_File_seek(MPI_File fh, MPI_Offset offset, int whence){
        if(fh.amode & MPI_MODE_SEQUENTIAL){
            return MPI_ERR_UNSUPPORTED_OPERATION;
        }

        int rank;
        MPI_Comm_rank(fh.comm, &rank);
        if(whence == MPI_SEEK_SET){
            if(offset < 0){
                // see documentation p521 line 11
                return MPI_ERR_UNSUPPORTED_OPERATION;
            }
            fh.seek_pos[rank] = offset;
        }else if(whence == MPI_SEEK_CUR){
            int new_offset = fh.seek_pos[rank] + offset;
            if(new_offset < 0){
                // see documentation p521 line 11
                return MPI_ERR_UNSUPPORTED_OPERATION;
            }
            fh.seek_pos[rank] = new_offset;
        }else if(whence == MPI_SEEK_END){
            int sz = __get_file_size(fh.file);
            int new_offset = sz + offset;
            if(new_offset < 0){
                // see documentation p521 line 11
                return MPI_ERR_UNSUPPORTED_OPERATION;
            }
            fh.seek_pos[rank] = new_offset;
        }

        return 0;
    }

    __device__ int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
        //todo
        return 0;
    }
    
    __device__ int MPI_File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
        //todo
        return 0;
    }

    __device__ int MPI_File_get_position(MPI_File fh, MPI_Offset *offset){
        int rank;
        MPI_Comm_rank(fh.comm, &rank);
        *offset = fh.seek_pos[rank];
        return 0;
    }

    __device__ int MPI_File_close(MPI_File *fh){
        // synchronize file state
        //fflush(fh->file);

        int rank;
        MPI_Comm_rank(fh->comm, &rank);

        // only free the file handle object once
        if(rank == 0){
            // close the file associated with file handle
            //fclose(fh->file);
            // release the fh object
            free(fh);
            fh = MPI_FILE_NULL;
        }
        __syncthreads();
        //MPI_Barrier(MPI_COMM_WORLD);
        return 0;
    }
}
