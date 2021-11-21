#include "io.cuh"
#include "../gpu_main/device_host_comm.cuh"
#include <cassert>
#include <cooperative_groups.h>

// #include "mpi.cuh"
#define N 100
namespace gpu_mpi {
}

    __device__ void mutex_lock(unsigned int *mutex) {
        unsigned int ns = 8;
        while (atomicCAS(mutex, 0, 1) == 1) {
            //__nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
        }
    }

    __device__ void mutex_unlock(unsigned int *mutex) {
        atomicExch(mutex, 0);
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


    __device__ MPI_File shared_fh;
    __device__ int MPI_File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh){
        // __shared__ int err_code;
        // __shared__ MPI_File shared_fh;   
        int err_code;

        // create MPI_FILE
        int rank;
        MPI_Comm_rank(comm, &rank);
        // auto block = cooperative_groups::this_thread_block();
        // rank = block.thread_rank();
        // printf("rank %d, blockIdx: %d, %d, %d\n", rank, blockIdx.x, blockIdx.y, blockIdx.z);
        if(rank == 0){
            err_code = 0;
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

            if(err_code == 0){
                shared_fh.amode = amode;
                shared_fh.comm = comm;

                // initialize fh->file
                shared_fh.file = __open_file(filename, I_FOPEN_MODE_RD);
                shared_fh.filename = filename;
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
                }else{
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
                    MPI_Comm_size(comm, &size);
                    shared_fh.seek_pos = (int*)malloc(size*sizeof(int));
                    int init_pos = 0;
                    if(amode & MPI_MODE_APPEND){
                        // In append mode: set pointer to end of file 
                        // see documentation p494 line 42
                        init_pos = __get_file_size(shared_fh.file);
                    }
                    // init_pos = 1;
                    for (int i = 0; i < size; i++){
                        shared_fh.seek_pos[i] = init_pos;
                    }

                    // TODO: allocate and initialize buffer array, status array, size
                    shared_fh.num_blocks = (int*)malloc(sizeof(int));
                    *(shared_fh.num_blocks) = INIT_BUFFER_BLOCK_NUM;
                    shared_fh.buffer = (void**)malloc(INIT_BUFFER_BLOCK_NUM * sizeof(void*));
                    shared_fh.status = (int*)malloc(INIT_BUFFER_BLOCK_NUM * sizeof(int));
                    for(int i = 0; i < INIT_BUFFER_BLOCK_NUM; i++){
                        shared_fh.buffer[i] = nullptr;
                        shared_fh.status[i] = BLOCK_NOT_IN;
                    }
                }   
            }
        }
        
        // printf("rank %d, First\n", rank);
        MPI_Barrier(MPI_COMM_WORLD); 
        // __syncthreads(); 
        // printf("rank %d, Second\n", rank);
        MPI_Bcast(&err_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
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

    __device__ void __read_file(MPI_File* fh, int block_index, int seekpos){
        // read specific block from file
        int buffer_size = 128;
        char* data = (char*) allocate_host_mem(buffer_size);
        ((int*)data)[0] = I_FREAD;
        ((FILE**)data)[1] = fh->file;
        ((void**)data)[2] = fh->buffer[block_index];
        ((int*)data)[6] = seekpos;
        delegate_to_host((void*)data, buffer_size);
        // wait
        while(((int*)data)[0] != I_READY){};
        free_host_mem(data);
        // printf("%s", "__read_file succeed\n"); 
    }
    
    __device__ void __read_block(MPI_File* fh, int block_index, int start, int count, void* buf, int seekpos){
        // check whether the whole block is in buffer
        if(fh->status[block_index] == BLOCK_NOT_IN){
            // data not in buffer, read to from cpu
            fh->buffer[block_index] = allocate_host_mem(INIT_BUFFER_BLOCK_SIZE);
            __read_file(fh, block_index, seekpos);
            fh->status[block_index] = BLOCK_IN_CLEAN;
        }

        // data in buffer
        void* buffer_start = fh->buffer[block_index];
        buffer_start = (char*)buffer_start + start;
        memcpy(buf, buffer_start, count);   
        // printf("%s", "__read_block succeed\n");     
    }

    __device__ int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
        if (!(fh.amode & MPI_MODE_RDONLY) && !(fh.amode & MPI_MODE_RDWR)) return MPI_ERR_AMODE;
        if (fh.amode & MPI_MODE_SEQUENTIAL) return MPI_ERR_UNSUPPORTED_OPERATION;  // p514 l43
        // TODO: Only one thread with RDWR can gain access; unlimited threads with RDONLY can gain access (?)
        // TODO: write into MPI_Status

        int rank;
        MPI_Comm_rank(fh.comm, &rank);
        int cur_pos = fh.seek_pos[rank];
        int cur_block = cur_pos / INIT_BUFFER_BLOCK_SIZE;
        int block_offset = cur_pos % INIT_BUFFER_BLOCK_SIZE;

        // for now only support read the whole block
        assert(block_offset == 0);
        assert(count % INIT_BUFFER_BLOCK_SIZE == 0);
        int num_block = count / INIT_BUFFER_BLOCK_SIZE;
        for(int i=0;i<num_block;i++){
            void* buf_start = ((char*)buf) + i * INIT_BUFFER_BLOCK_SIZE;
            int seekpos = cur_pos + i * INIT_BUFFER_BLOCK_SIZE;
            __read_block(&fh, cur_block + i, 0, INIT_BUFFER_BLOCK_SIZE, buf_start, seekpos);
        }

        // assume we always can read count data
        fh.seek_pos[rank] += count;
        return count;
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

    // TODO: ask CPU to write the buffer back to disk
    __device__ void __write_file(MPI_File* fh, MPI_Datatype datatype, int block_index, int seekpos){
        // printf("file %p, buf %p, seekpos %d\n", fh->file, fh->buffer[block_index], seekpos);
        // write specific block back to file
        int buffer_size = 128;
        char* data = (char*) allocate_host_mem(buffer_size);
        ((int*)data)[0] = I_FWRITE;
        ((FILE**)data)[1] = fh->file;
        ((void**)data)[2] = fh->buffer[block_index];
        ((int*)data)[6] = seekpos;
        delegate_to_host((void*)data, buffer_size);
        // wait
        while(((int*)data)[0] != I_READY){};
        free_host_mem(data);
        // int cnt = (int)(((size_t*)data)[1]);
        // return the number of how much should seek_pos change
        // return cnt;
    }

    __device__ void __write_block(MPI_File* fh, int block_index, int start, int count, const void* buf, int seekpos){
        // TODO: check if block_index > num_blocks and whether need to allocate more buffer

        // for now only support read the whole block
        // check whether the whole block is in buffer
        if(fh->status[block_index] == BLOCK_NOT_IN){
            // data not in buffer, read to from cpu
            fh->buffer[block_index] = allocate_host_mem(INIT_BUFFER_BLOCK_SIZE);
            __read_file(fh, block_index, seekpos);
            fh->status[block_index] = BLOCK_IN_CLEAN;
        }

        // write buffer
        void* buffer_start = fh->buffer[block_index];
        buffer_start = (char*)buffer_start + start;
        memcpy(buffer_start, buf, count);
        fh->status[block_index] = BLOCK_IN_DIRTY;
        // printf("%s", "__write_block succeeds\n");
    }

    __device__ unsigned int lock;
    __device__ int MPI_File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
        // TODO: check amode

        // TODO: Only one thread can get access 
        mutex_lock(&lock);

        assert(datatype == MPI_CHAR);
        // write into buffer
        int rank;
        MPI_Comm_rank(fh.comm, &rank);
        int cur_pos = fh.seek_pos[rank];
        int cur_block = cur_pos / INIT_BUFFER_BLOCK_SIZE;
        int block_offset = cur_pos % INIT_BUFFER_BLOCK_SIZE;
        int num_block = 1 + (count - (INIT_BUFFER_BLOCK_SIZE - block_offset)) / INIT_BUFFER_BLOCK_SIZE;

        // pointer to the buf we are reading from
        const void* buf_start = buf;
        int remain_count = count;
        int seekpos = cur_pos;

        // need to write partial of the first block
        if(block_offset != 0){
            __write_block(&fh, cur_block, block_offset, INIT_BUFFER_BLOCK_SIZE - block_offset, buf_start, seekpos);
            // check if this += is correct!!!!!!!!!!!!!!!!!!!!!!!!!
            buf_start = (const char*)buf_start + INIT_BUFFER_BLOCK_SIZE - block_offset;
            seekpos += INIT_BUFFER_BLOCK_SIZE - block_offset;
            cur_block += 1;
            num_block -= 1;
            remain_count -= INIT_BUFFER_BLOCK_SIZE - block_offset;
        }   
        // Is it alright to mix size and count here? what if the MPI_Datatype is double, whose size is not 1 as char?
        for(int i = 0; i < num_block; i++){
            size_t write_size = INIT_BUFFER_BLOCK_SIZE;
            size_t write_buffer_offset = 0;
            // need to write partial of the last block
            if(i == num_block - 1){
                write_size = remain_count - i * INIT_BUFFER_BLOCK_SIZE;
            }
            __write_block(&fh, cur_block + i, write_buffer_offset, write_size, buf_start, seekpos);
            buf_start = (const char*)buf_start + INIT_BUFFER_BLOCK_SIZE;
            seekpos += INIT_BUFFER_BLOCK_SIZE;
        }

        // assume we can always write count data
        fh.seek_pos[rank] += count;

        mutex_unlock(&lock);
        // TODO: Make sure what return value should be
        return count;
    }
    
    __device__ void __delete_file(const char* filename){
        if(filename == NULL){
            return;
        }
        // TODO: ask cpu to remove the file
        int buffer_size = 128;
        char* data = (char*) allocate_host_mem(buffer_size);

        ((int*)data)[0] = I_FDELETE;
        // remove the file associated with filename
        int filename_size = 0;
        while (filename[filename_size] != '\0') filename_size++;
        memcpy((const char**)data + 1, filename, filename_size + 1);
        delegate_to_host((void*)data, buffer_size);
        // wait
        while(((int*)data)[0] != I_READY){};
        //file remove done
        free_host_mem(data);
    }

    __device__ int MPI_File_close(MPI_File *fh){
        // synchronize file state
        MPI_Barrier(MPI_COMM_WORLD);
        // __syncthreads();

        int rank;
        MPI_Comm_rank(fh->comm, &rank);

        // only free the file handle object once
        if(rank == 0){
            // TODO: if fh->amode is MPI_MODE_DELETE_ON_CLOSE, need to delete that file
            if(fh->amode == MPI_MODE_DELETE_ON_CLOSE){
                ;
            }
            // release the fh object
            free(fh->seek_pos);
            // TODO: check status of buffer blocks, write dirty blocks back
            int num_blocks = *(fh->num_blocks);
            for(int i = 0; i < num_blocks; i++){
                if(fh->status[i] == BLOCK_IN_DIRTY){
                    // TODO: write back
                    __write_file(fh, MPI_CHAR, i, i * INIT_BUFFER_BLOCK_SIZE);
                    fh->status[i] = BLOCK_IN_CLEAN;
                }
            }
            // TODO: release buffer array
            for(int i = 0; i < num_blocks; i++){
                if(fh->status[i] != BLOCK_NOT_IN)
                    free(fh->buffer[i]);
            }
            free(fh->buffer);
            // TODO: release status array
            free(fh->status);
            free(fh->num_blocks);
            fh->filename = nullptr;
            // close the file associated with file handle
            __close_file(fh->file);
        }

        // __syncthreads();
        MPI_Barrier(MPI_COMM_WORLD);
        return 0;
    }

    __device__ int MPI_File_get_size(MPI_File fh, MPI_Offset *size){
        int sz; 

        int rank;
        MPI_Comm_rank(fh.comm, &rank);
        if(rank == 0){
            sz = __get_file_size(fh.file);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        *size = sz;

        return 0;
    } 

