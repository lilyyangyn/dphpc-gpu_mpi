#include "io.cuh"

namespace gpu_mpi {

    __device__ int MPI_File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh){
        // check amode
        int cnt = 0;
        if(amode & MPI_MODE_RDONLY){
            cnt++;
        }
        if(amode & MPI_MODE_RDWR){
            cnt++;
        }
        if(amode & MPI_MODE_WRONLY){
            cnt++;
        }
        if(cnt != 1){
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

        // todo: initialize fh->amode
        // todo: initialize fh->comm
        // todo: initialize fh->seek_pos
        // todo: fh->seek_pos[] should be initialized to all zeros
        // todo: initialize fh->file
        
        // todo
        return 0;
    }

    __device__ int __get_file_size(FILE* file){
        // fseek(file, 0L, SEEK_END);
        // return ftell(file);
        
        // todo: how to get the file size? how to talk to cpu?
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
}