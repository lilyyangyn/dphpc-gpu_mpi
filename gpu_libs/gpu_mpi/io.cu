#include "io.cuh"
#include "../gpu_main/device_host_comm.cuh"

// #include "mpi.cuh"

//why we need this namespace? MPI_File struct is in this namespace 
namespace gpu_mpi {


    __device__ FILE* __open_file(const char* filename, const char* mode){
        if(filename == NULL){
            return nullptr;
        }
        int buffer_size = 128;
        char* data = (char*) allocate_host_mem(buffer_size);
        ((int*)data)[0] = I_FOPEN;
        ((const char**)data)[1] = filename;
        ((const char**)data)[2] = mode;
        delegate_to_host((void*)data, buffer_size);
        // wait
        while(((int*)data)[0] != I_READY){};
        
        FILE* file = ((FILE**)data)[1];
        free_host_mem(data);
        return file;
    }

    __device__ void __close_file(FILE* file){
        if(file == NULL){
            return;
        }

        int buffer_size = 128;
        char* data = (char*) allocate_host_mem(buffer_size);
        // close the file associated with file handle
        ((int*)data)[0] = I_FCLOSE;
        ((FILE**)data)[1] = file;
        delegate_to_host((void*)data, buffer_size);
        // wait
        while(((int*)data)[0] != I_READY){};
        //fclose done
        free_host_mem(data);
    }

    __device__ long int __get_file_size(FILE* file){
        int buffer_size = 128;
        char* data = (char*) allocate_host_mem(buffer_size);
        ((int*)data)[0] = I_FSEEK;
        ((FILE**)data)[1] = file;
        delegate_to_host((void*)data, buffer_size);
        // wait
        while(((int*)data)[0] != I_READY){};
        long int file_length = ((long int*)data)[1];
        free_host_mem(data);
        return file_length;
    }

    __device__ int MPI_File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh){
        __device__ __shared__ int err_code;
        // check amode
        if (((amode & MPI_MODE_RDONLY) ? 1 : 0) + ((amode & MPI_MODE_RDWR) ? 1 : 0) +
            ((amode & MPI_MODE_WRONLY) ? 1 : 0) != 1) {
            // see documentation p495 line 7
            err_code = MPI_ERR_AMODE;
        }
        if(((amode & MPI_MODE_CREATE) || (amode & MPI_MODE_EXCL)) && (amode & MPI_MODE_RDONLY)){
            // see documentation p495 line 8
            err_code = MPI_ERR_AMODE;
        }
        if((amode & MPI_MODE_RDWR) && (amode & MPI_MODE_SEQUENTIAL)){
            // see documentation p495 line 9
            err_code = MPI_ERR_AMODE;
        }

        // create MPI_FILE
        int rank;
        MPI_Comm_rank(comm, &rank);
        if(err_code == 0 && rank == 0){
            fh->amode = amode;
            fh->comm = comm;

            // initialize fh->file
            fh->file = __open_file(filename, "r");

            // check file existence
            if(fh->file == NULL){
                if(!(amode & MPI_MODE_RDONLY)){
                    err_code = MPI_ERR_NO_SUCH_FILE;
                }
                if(!(amode & MPI_MODE_CREATE)){
                    err_code = MPI_ERR_NO_SUCH_FILE;
                }
            }
            if(amode & MPI_MODE_EXCL){
                // File must not exist
                err_code = MPI_ERR_FILE_EXISTS;
            }

            if(err_code != 0) {
                __close_file(fh->file);
                __syncthreads();
                return err_code;
            }

            if(!(amode & MPI_MODE_RDONLY)){
                __close_file(fh->file);
                const char *mode;
                if(amode & MPI_MODE_RDWR){
                    mode = "r+";
                }else if(amode & MPI_MODE_WRONLY){
                    // TODO: append or overwrite???
                    mode = "a";
                }
                fh->file = __open_file(filename, mode);
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
        
        __syncthreads();
        
        return err_code;
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

    // __device__ int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    //     //todo
    //     return 0;
    // }
    
    // __device__ int MPI_File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    //     //todo
    //     return 0;
    // }

    __device__ int MPI_File_get_position(MPI_File fh, MPI_Offset *offset){
        int rank;
        MPI_Comm_rank(fh.comm, &rank);
        *offset = fh.seek_pos[rank];
        return 0;
    }

    __device__ int MPI_File_close(MPI_File *fh){
        // synchronize file state
        //fflush(fh->file);
        int buffer_size = 128;
        char* data = (char*) allocate_host_mem(buffer_size);
        ((int*)data)[0] = I_FFLUSH;
        ((FILE**)data)[1] = fh->file;
        delegate_to_host((void*)data, buffer_size);
        // wait
        while(((int*)data)[0] != I_READY){};
        //fflush done

        int rank;
        MPI_Comm_rank(fh->comm, &rank);

        // only free the file handle object once
        if(rank == 0){
            // close the file associated with file handle
            //fclose(fh->file);
            ((int*)data)[0] = I_FCLOSE;
            ((FILE**)data)[1] = fh->file;
            delegate_to_host((void*)data, buffer_size);
            // wait
            while(((int*)data)[0] != I_READY){};
            //fclose done
            free_host_mem(data);
            
            // release the fh object
            free(fh);
            fh = MPI_FILE_NULL;
        }
        __syncthreads();
        //MPI_Barrier(MPI_COMM_WORLD);
        return 0;
    }
}
