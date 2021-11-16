#include "io.cuh"
#include "../gpu_main/device_host_comm.cuh"
#include <cassert>

// #include "mpi.cuh"
#define N 100
namespace gpu_mpi {
}
    __device__ FILE* __open_file(const char* filename, int mode){
        if(filename == NULL){
            return nullptr;
        }
        int buffer_size = 128;
        char* data = (char*) allocate_host_mem(buffer_size);
        ((int*)data)[0] = I_FOPEN;
        ((int*)data)[1] = mode;

        int filename_size = 0;
        while (filename[filename_size] != '\0') filename_size++;
        memcpy((const char**)data + 2 , filename, filename_size+1);
        
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
        __device__ __shared__ MPI_File shared_fh;

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
            shared_fh.amode = amode;
            shared_fh.comm = comm;

            // initialize fh->file
            shared_fh.file = __open_file(filename, I_FOPEN_MODE_RD);

            // check file existence
            if(shared_fh.file == NULL){
                if(amode & MPI_MODE_RDONLY){
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
                __close_file(shared_fh.file);
                __syncthreads();
                return err_code;
            }

            if(!(amode & MPI_MODE_RDONLY)){
                __close_file(shared_fh.file);
                int mode;
                if(amode & MPI_MODE_RDWR){
                    if(amode & MPI_MODE_APPEND) {
                        mode = I_FOPEN_MODE_RW_APPEND;
                    }else{
                        mode = I_FOPEN_MODE_RW;
                    }
                }else if(amode & MPI_MODE_WRONLY){
                    if(amode & MPI_MODE_APPEND) {
                        mode = I_FOPEN_MODE_WD_APPEND;
                    }else{
                        mode = I_FOPEN_MODE_WD;
                    }
                }
                shared_fh.file = __open_file(filename, mode);
            }
            
            // initialize fh->seek_pos
            // TODO: MPI_MODE_UNIQUE_OPEN -> Only one seek_pos???
            int size;
            MPI_Comm_rank(comm, &size);
            fh->seek_pos = (int*)malloc(size*sizeof(int));
            int init_pos = 0;
            if(amode & MPI_MODE_APPEND){
                // In append mode: set pointer to end of file 
                // see documentation p494 line 42
                init_pos = __get_file_size(shared_fh.file);
            }
            memset(shared_fh.seek_pos, init_pos, sizeof(int) * size);
        }
        
        __syncthreads();
        *fh = shared_fh;

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

    __device__ int MPI_File_get_position(MPI_File fh, MPI_Offset *offset){
        int rank;
        MPI_Comm_rank(fh.comm, &rank);
        *offset = fh.seek_pos[rank];
        return 0;
    }

    __device__ int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
        if (!(fh.amode & MPI_MODE_RDONLY) && !(fh.amode & MPI_MODE_RDWR)) return MPI_ERR_AMODE;
        if (fh.amode & MPI_MODE_SEQUENTIAL) return MPI_ERR_UNSUPPORTED_OPERATION;  // p514 l43
        // TODO: Only one thread with RDWR can gain access; unlimited threads with RDONLY can gain access (?)
        // TODO: write into MPI_Status

        int buffer_size = sizeof(int) + sizeof(FILE*) + sizeof(MPI_Datatype) + sizeof(void*) + sizeof(int) + 2048;  // (TODO: dynamic size) sizeof(datatype) * count;
        void* data = (void*)allocate_host_mem(buffer_size);
        ((int*)data)[0] = I_FREAD;
        ((FILE**)data)[1] = fh.file;
        ((MPI_Datatype*)data)[2] = datatype;
        ((void**)data)[3] = (void**)data + 5;  // buf;
        ((int*)data)[4] = count;
        
        delegate_to_host((void*)data, buffer_size);
        while (((int*)data)[0] != I_READY)
        {
            // blocking wait (p506 l44)
        }

        memcpy(buf, (void**)data + 5, sizeof(datatype) * count);
        free_host_mem(data);
        return ((size_t*)data)[1];
    }
    
    __device__ int __howManyBits(int x) {
        assert(sizeof(int)==4);
        return __double2int_ru(__log2f(x));
        // int a = x >> 31;
        // int newx = (x ^ a); //2
        // int temp1 = !!(newx >> 16);//5
        // int ntemp1 = ~temp1 + 1; //7
        // int n = 16 & ntemp1;     //8
        // int temp2 = !!(newx >> (8 + n)); //12
        // int ntemp2 = ~temp2 + 1; //14
        // int temp3 = 0;
        // int ntemp3 = 0;
        // int temp4 = 0;
        // int ntemp4 = 0;
        // int temp5 = 0;

        // n = n + (ntemp2 & 8);
        // temp3 = !!(newx >> (4 + n));
        // ntemp3 = ~temp3 + 1;
        // n = n + (ntemp3 & 4);
        // temp4 = !!(newx >> (2 + n));
        // ntemp4 = ~temp4 + 1;
        // n = n + (ntemp4 & 2);
        // temp5 = !!(newx >> (1 + n));
        // n = n + temp5 + 1 + !!newx; 
        // return n;
    }

    //for debug
    __device__ __host__ void __show_memory(char * mem, size_t size){
        char *tmem = (char *)mem;
        for(int i=0;i+7<size;i+=8){
            printf("%02X  %02X  %02X  %02X  %02X  %02X  %02X  %02X\n",tmem[i],tmem[i+1],tmem[i+2],tmem[i+3],tmem[i+4],tmem[i+5],tmem[i+6],tmem[i+7]);
        }
    }

    //not thread safe
    __device__ int MPI_File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
        //TODO: dynamically assign buffer size
        int buffer_size = 2048;
        // int MPI_Type_size(MPI_Datatype datatype, int *size)
        //TODO: MPI_Type_size not implemented
        assert(datatype==MPI_CHAR);
        assert(buffer_size > sizeof(int*)*2+sizeof(FILE**)+sizeof(char)*count);
        //init
        char* data = (char*) allocate_host_mem(buffer_size);
        //assemble
        *((int*)data) = I_FWRITE;
        *((int*)(data+4)) = count;
        // printf("OS file descripter address in MPI_File_write:%p\n", fh.file);
        *((FILE **)(data+8)) = fh.file;
        // __show_memory(data, 64);
        

        *((MPI_Datatype*)(data+16)) = datatype;
        memcpy( ((const char**)data+24) , buf, sizeof(char)*count);

        //execute on CPU
        delegate_to_host((void*)data, buffer_size);
        // wait
        while(((int*)data)[0] != I_READY){};
        int return_value = (int) *((size_t*)(data+8));
        
        int rank;
        MPI_Comm_rank(fh.comm, &rank);
        //assuming individual file pointer, but how does shared pointer differ from this?
        // fh.seek_pos[rank]+=return_value;
        free_host_mem(data);
        //TODO: step 4 error catching
        //#memory cosistency: assuming that write is not reordered with write
        return return_value;
    }


    __device__ int MPI_File_close(MPI_File *fh){
        // synchronize file state
        __syncthreads();

        int rank;
        MPI_Comm_rank(fh->comm, &rank);

        // only free the file handle object once
        if(rank == 0){
            // close the file associated with file handle
            // fclose(fh->file);
            __close_file(fh->file);
            
            // release the fh object
            free(fh->seek_pos);
        }
        __syncthreads();
        //MPI_Barrier(MPI_COMM_WORLD);
        return 0;
    }

