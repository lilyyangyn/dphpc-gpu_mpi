#include "io.cuh"
#include "mpi_common.cuh"
#include "datatypes.cuh"
#include "../gpu_main/device_host_comm.cuh"
#include <cassert>
#include <cooperative_groups.h>

/**
 * version switch
 */
#define USE_BUFFER true
#define USE_VIEW_LAYOUT true

#define N 100
namespace gpu_mpi {
}

/* ------HELPER FUNCTIONS------ */

__device__ void mutex_lock(unsigned int *mutex) {
    while (atomicCAS(mutex, 0, 1) == 1) {}
}

__device__ void mutex_unlock(unsigned int *mutex) {
    atomicExch(mutex, 0);
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

__device__ int __view_pos_to_file_pos(MPI_File_View view, int position){
    
    int typemap_offset_idx = position % view.filetype.typemap_len;
    int typemap_idx = position / view.filetype.typemap_len;
    return view.disp + typemap_idx * view.filetype.size() + view.filetype.typemap[typemap_offset_idx].disp;
}

__device__ int __file_pos_to_view_pos(MPI_File_View view, int position){
    int adjusted_pos = position - view.disp;
    if(adjusted_pos < 0) return -1;
    int typemap_offset = adjusted_pos % view.filetype.size();
    int typemap_idx = adjusted_pos / view.filetype.size();
    int typemap_offset_idx = 0;
    while(typemap_offset_idx < view.filetype.typemap_len){
        if(typemap_offset == view.filetype.typemap[typemap_offset_idx].disp){
            break;
        }
        typemap_offset_idx++;
    }
    if(typemap_offset_idx == view.filetype.typemap_len) return -1;
    return typemap_idx * view.filetype.typemap_len + typemap_offset_idx;
}

__device__ FILE* __open_file(const char* filename, int amode, int* file_exist){
    if(filename == NULL){
        return nullptr;
    }
    int buffer_size = 128;
    char* data = (char*) allocate_host_mem(buffer_size);
    ((int*)data)[0] = I_FOPEN;
    ((int*)data)[1] = amode;

    int filename_size = 0;
    while (filename[filename_size] != '\0') filename_size++;
    memcpy((const char**)data + 2 , filename, filename_size+1);
    
    delegate_to_host((void*)data, buffer_size);
    // wait
    while(((int*)data)[0] != I_READY){};
    
    *file_exist = ((int*)data)[1];
    FILE* file = ((FILE**)data)[2];
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

__device__ int __delete_file(const char* filename){
    if(filename == NULL){
        return 0;
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
    int res = ((int*)data)[1];
    free_host_mem(data);

    return res;
}

#if USE_BUFFER

__device__ void __read_file(MPI_File* fh, int block_index, int seekpos){
    // read specific block from file
    int buffer_size = 128;
    char* data = (char*) allocate_host_mem(buffer_size);
    ((int*)data)[0] = I_FREAD_BUFFER;
    ((FILE**)data)[1] = fh->file;
    ((void**)data)[2] = fh->buffer[block_index].block;
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
        fh->buffer[block_index].block = allocate_host_mem(INIT_BUFFER_BLOCK_SIZE);
        __read_file(fh, block_index, seekpos);
        fh->status[block_index] = BLOCK_IN_CLEAN;
    }
    // data in buffer
    void* buffer_start = fh->buffer[block_index].block;
    buffer_start = (char*)buffer_start + start;
    memcpy(buf, buffer_start, count); 
    // printf("%s", "__read_block succeed\n");     
}

__device__ int __read_buffer(MPI_File fh, void *buf, size_t size, size_t count, int pos){
    int cur_block = pos / INIT_BUFFER_BLOCK_SIZE;
    int block_offset = pos % INIT_BUFFER_BLOCK_SIZE;
    int num_block;
    int bytes_count = count * size;
    if(block_offset == 0){
        if(bytes_count % INIT_BUFFER_BLOCK_SIZE == 0){
            num_block = bytes_count / INIT_BUFFER_BLOCK_SIZE;
        }
        else{
            num_block = bytes_count / INIT_BUFFER_BLOCK_SIZE + 1;
        }
    }
    else{
        int remain = bytes_count - (INIT_BUFFER_BLOCK_SIZE - block_offset);
        if(remain < 0){
            num_block = 1;
        }
        else if(remain % INIT_BUFFER_BLOCK_SIZE == 0){
            num_block = 1 + remain / INIT_BUFFER_BLOCK_SIZE;
        }
        else{
            num_block = 1 + remain / INIT_BUFFER_BLOCK_SIZE + 1;
        }
    }

    // pointer to the buf we are writing into
    void* buf_start = buf;
    int remain_count = bytes_count;
    int seekpos = cur_block * INIT_BUFFER_BLOCK_SIZE;

    if(block_offset != 0){
        int read_size = (bytes_count >= (INIT_BUFFER_BLOCK_SIZE - block_offset))?(INIT_BUFFER_BLOCK_SIZE - block_offset):bytes_count;
        __read_block(&fh, cur_block, block_offset, read_size, buf_start, seekpos);
        buf_start = (char*)buf_start + INIT_BUFFER_BLOCK_SIZE - block_offset;
        seekpos += INIT_BUFFER_BLOCK_SIZE;
        cur_block += 1;
        num_block -= 1;
        remain_count -= (INIT_BUFFER_BLOCK_SIZE - block_offset);
    }
    for(int i = 0; i < num_block; i++){
        size_t read_size = INIT_BUFFER_BLOCK_SIZE;
        size_t read_buffer_offset = 0;
        // need to write partial of the last block
        if(i == num_block - 1){
            read_size = remain_count - i * INIT_BUFFER_BLOCK_SIZE;
        }
        __read_block(&fh, cur_block + i, read_buffer_offset, read_size, buf_start, seekpos);
        buf_start = (char*)buf_start + INIT_BUFFER_BLOCK_SIZE;
        seekpos += INIT_BUFFER_BLOCK_SIZE;
    }
    return bytes_count / size;
}

// TODO: ask CPU to write the buffer back to disk
__device__ void __write_file(MPI_File* fh, MPI_Datatype datatype, int block_index, int seekpos, int writebytes){
    // printf("file %p, buf %p, seekpos %d\n", fh->file, fh->buffer[block_index], seekpos);
    // write specific block back to file
    int buffer_size = 128;
    char* data = (char*) allocate_host_mem(buffer_size);
    ((int*)data)[0] = I_FWRITE_BUFFER;
    ((FILE**)data)[1] = fh->file;
    ((void**)data)[2] = fh->buffer[block_index].block;
    ((int*)data)[6] = seekpos;
    ((int*)data)[7] = writebytes;
    delegate_to_host((void*)data, buffer_size);
    // wait
    while(((int*)data)[0] != I_READY){};
    free_host_mem(data);
    // int cnt = (int)(((size_t*)data)[1]);
    // return the number of how much should seek_pos change
    // return cnt;
}

__device__ void __write_block_Faster(MPI_File* fh, int block_index, int start, int count, const void* buf, int seekpos){
    // MUST WRITE THE WHOLE BUFFER
    assert(count == INIT_BUFFER_BLOCK_SIZE);
    assert(start == 0);

    void* buffer_start = allocate_host_mem(INIT_BUFFER_BLOCK_SIZE);
    memcpy(buffer_start, buf, count);
    
    mutex_lock(&fh->buffer[block_index].lock);
    if(fh->status[block_index] != BLOCK_NOT_IN){
        free_host_mem(fh->buffer[block_index].block);
    }
    fh->buffer[block_index].block = buffer_start;
    fh->status[block_index] = BLOCK_IN_DIRTY;
    mutex_unlock(&fh->buffer[block_index].lock);
}    

__device__ void __write_block(MPI_File* fh, int block_index, int start, int count, const void* buf, int seekpos){
    mutex_lock(&fh->buffer[block_index].lock);
    // TODO: check if block_index > num_blocks and whether need to allocate more buffer
    // for now only support read the whole block
    // check whether the whole block is in buffer
    if(fh->status[block_index] == BLOCK_NOT_IN){
        // data not in buffer, read to from cpu
        fh->buffer[block_index].block = allocate_host_mem(INIT_BUFFER_BLOCK_SIZE);
        // memset(fh->buffer[block_index].block, 0, INIT_BUFFER_BLOCK_SIZE);
        __read_file(fh, block_index, seekpos);
        fh->status[block_index] = BLOCK_IN_CLEAN;
    }

    // write buffer
    void* buffer_start = fh->buffer[block_index].block;
    buffer_start = (char*)buffer_start + start;
    // printf("buffer_start %p, buf %p, count %d\n", buffer_start, buf, count);
    memcpy(buffer_start, buf, count);
    fh->status[block_index] = BLOCK_IN_DIRTY;
    mutex_unlock(&fh->buffer[block_index].lock);
    // printf("%s", "__write_block succeeds\n");
}

__device__ unsigned int sizelock = 0;
__device__ int __write_buffer(MPI_File fh, const void *buf, size_t size, size_t count, int pos){
    int cur_block = pos / INIT_BUFFER_BLOCK_SIZE;
    int block_offset = pos % INIT_BUFFER_BLOCK_SIZE;
    int num_block;
    int bytes_count = count * size;
    if(block_offset == 0){
        if(bytes_count % INIT_BUFFER_BLOCK_SIZE == 0){
            num_block = bytes_count / INIT_BUFFER_BLOCK_SIZE;
        }
        else{
            num_block = bytes_count / INIT_BUFFER_BLOCK_SIZE + 1;
        }
    }
    else{
        int remain = bytes_count - (INIT_BUFFER_BLOCK_SIZE - block_offset);
        if(remain < 0){
            num_block = 1;
        }
        else if(remain % INIT_BUFFER_BLOCK_SIZE == 0){
            num_block = 1 + remain / INIT_BUFFER_BLOCK_SIZE;
        }
        else{
            num_block = 1 + remain / INIT_BUFFER_BLOCK_SIZE + 1;
        }
    }

    // assume we can always write bytes_count, update the filesize if necessary
    if( bytes_count + pos > *(fh.filesize)){
        mutex_lock(&sizelock);
        *(fh.filesize) = bytes_count + pos;
        mutex_unlock(&sizelock);
    }

    // printf("rank is %d, cur_block is %d, num_block is %d\n", rank, cur_block, num_block);
    // pointer to the buf we are reading from
    const void* buf_start = buf;
    int remain_count = bytes_count;
    int seekpos = cur_block * INIT_BUFFER_BLOCK_SIZE;

    // TODO: revice buffer structure, so that each block has a lock?
    // need to write partial of the first block
    if(block_offset != 0){
        int write_size = (bytes_count >= (INIT_BUFFER_BLOCK_SIZE - block_offset))?(INIT_BUFFER_BLOCK_SIZE - block_offset):bytes_count;
        __write_block(&fh, cur_block, block_offset, write_size, buf_start, seekpos);
        buf_start = (const char*)buf_start + INIT_BUFFER_BLOCK_SIZE - block_offset;
        seekpos += INIT_BUFFER_BLOCK_SIZE;
        cur_block += 1;
        num_block -= 1;
        remain_count -= INIT_BUFFER_BLOCK_SIZE - block_offset;
    }
    // printf("rank is %d, offset is %d, cur_block is %d, num_block is %d\n", rank, block_offset, cur_block, num_block);
    // Is it alright to mix size and count here? what if the MPI_Datatype is double, whose size is not 1 as char?
    for(int i = 0; i < num_block; i++){
        int write_size = (int)INIT_BUFFER_BLOCK_SIZE;
        int write_buffer_offset = 0;
        // need to write partial of the last block
        if(i == num_block - 1){
            write_size = remain_count - i * INIT_BUFFER_BLOCK_SIZE;
        }
        __write_block(&fh, cur_block + i, write_buffer_offset, write_size, buf_start, seekpos);
        buf_start = (const char*)buf_start + INIT_BUFFER_BLOCK_SIZE;
        seekpos += INIT_BUFFER_BLOCK_SIZE;
    }
    return bytes_count / size;
}

#endif


/* ------FILE MANIPULATION------ */

__device__ MPI_File shared_fh;
__device__ int MPI_File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh){ 
    int err_code;

    // create MPI_FILE
    int rank;
    MPI_Comm_rank(comm, &rank);
    // auto block = cooperative_groups::this_thread_block();
    // rank = block.thread_rank();
    if(rank == 0){
        err_code = MPI_SUCCESS;
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
        if(err_code == MPI_SUCCESS){
            shared_fh.amode = amode;
            shared_fh.comm = comm;
            shared_fh.filename = filename;

            // initialize fh->file
            int file_exist;
            shared_fh.file = __open_file(filename, amode, &file_exist);
            // check file existence
            if(!(amode & MPI_MODE_CREATE) && (file_exist == 0)){
                err_code = MPI_ERR_NO_SUCH_FILE;
            }

            if(err_code == MPI_SUCCESS) {
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
                shared_fh.views = (MPI_File_View*)malloc(size*sizeof(MPI_File_View)); 
                for (int i = 0; i < size; i++){
                    shared_fh.seek_pos[i] = init_pos;
                    // see documentation p492 line 17
                    shared_fh.views[i] = MPI_File_View(0, MPI_BYTE, MPI_BYTE, "native");
                }   
                // init_pos = 1;
                shared_fh.shared_seek_pos = (int*)malloc(sizeof(int));
                *(shared_fh.shared_seek_pos) = init_pos;

                #if USE_BUFFER
                    // TODO: allocate and initialize buffer array, status array, size, shared_seek_pos, filesize
                    shared_fh.num_blocks = (int*)malloc(sizeof(int));
                    *(shared_fh.num_blocks) = INIT_BUFFER_BLOCK_NUM;
                    shared_fh.shared_seek_pos = (int*)malloc(sizeof(int));
                    *(shared_fh.shared_seek_pos) = init_pos;
                    shared_fh.buffer = (BufBlock*)malloc(INIT_BUFFER_BLOCK_NUM * sizeof(BufBlock));
                    // shared_fh.buffer = (void**)malloc(INIT_BUFFER_BLOCK_NUM * sizeof(void*));
                    shared_fh.status = (int*)malloc(INIT_BUFFER_BLOCK_NUM * sizeof(int));
                    for(int i = 0; i < INIT_BUFFER_BLOCK_NUM; i++){
                        shared_fh.buffer[i].block = nullptr;
                        // shared_fh.buffer[i] = nullptr;
                        shared_fh.status[i] = BLOCK_NOT_IN;
                    }
                    shared_fh.filesize = (int*)malloc(sizeof(int));
                    *(shared_fh.filesize) = 0;
                    if((amode & MPI_MODE_RDWR) || (amode & MPI_MODE_RDONLY) || (amode & MPI_MODE_APPEND)){
                        int sz = __get_file_size(shared_fh.file);
                        *(shared_fh.filesize) = sz;
                    }
                #endif
            }   
        } 
    }
    
    MPI_Barrier(comm); 
    // __syncthreads(); 
    MPI_Bcast(&err_code, 1, MPI_INT, 0, comm);
    *fh = shared_fh;

    // printf("rank %d, amode %d, file %p, err_code %d, %p\n", rank, shared_fh.amode, shared_fh.file, err_code, &err_code);
    return err_code;
}

__device__ int MPI_File_delete(const char *filename, MPI_Info info){
    int err_code; 

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0){
        int file_exist;
        FILE* file = __open_file(filename, MPI_MODE_RDONLY, &file_exist);
        if(file_exist == 0){
            return MPI_ERR_NO_SUCH_FILE;
        }
        int res = __delete_file(filename);
        err_code = (res == 0) ? MPI_SUCCESS : MPI_ERR_IO; 
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&err_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return err_code;
}

__device__ int MPI_File_get_size(MPI_File fh, MPI_Offset *size){
    // int sz; 

    // int rank;
    // MPI_Comm_rank(fh.comm, &rank);
    // if(rank == 0){
    //     sz = __get_file_size(fh.file);
    // }

    // MPI_Barrier(fh.comm);
    // MPI_Bcast(&sz, 1, MPI_INT, 0, fh.comm);
    // *size = sz;
    *size = *(fh.filesize);

    return MPI_SUCCESS;
} 

__device__ int MPI_File_close(MPI_File *fh){
    // synchronize file state
    MPI_Barrier(fh->comm);
    // __syncthreads();

    int rank;
    MPI_Comm_rank(fh->comm, &rank);
    // only free the file handle object once
    if(rank == 0){
        // release the fh object
        free(fh->seek_pos);
        free(fh->views);
        free(fh->shared_seek_pos);

        #if USE_BUFFER
            // check status of buffer blocks, write dirty blocks back
            int num_blocks = *(fh->num_blocks);
            for(int i = 0; i < num_blocks; i++){
                if(fh->status[i] == BLOCK_IN_DIRTY){
                    // write back
                    // special check when it's the 'real' last block of the file 
                    int writebytes = INIT_BUFFER_BLOCK_SIZE;
                    if((i + 1) * INIT_BUFFER_BLOCK_SIZE - 1 > *(fh->filesize)){
                        writebytes = *(fh->filesize) - i * INIT_BUFFER_BLOCK_SIZE;
                    }
                    __write_file(fh, fh->views->etype, i, i * INIT_BUFFER_BLOCK_SIZE, writebytes);
                    fh->status[i] = BLOCK_IN_CLEAN;
                }
            }

            // release buffer array
            for(int i = 0; i < num_blocks; i++){
                if(fh->status[i] != BLOCK_NOT_IN)
                    free_host_mem(fh->buffer[i].block);
            }
            free(fh->buffer);
            free(fh->status);
            free(fh->num_blocks);
            free(fh->filesize);
        #endif
    
        // close the file associated with file handle
        __close_file(fh->file);

        // TODO: if fh->amode is MPI_MODE_DELETE_ON_CLOSE, need to delete that file
        if(fh->amode == MPI_MODE_DELETE_ON_CLOSE){
            __delete_file(fh->filename);
        }
        fh->filename = nullptr;
    }

    // __syncthreads();
    MPI_Barrier(fh->comm);
    return 0;
}

/* ------FILE VIEWS------ */

__device__ int MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, const char *datarep, MPI_Info info){
    // now only support single type repitition
    // assert(etype == filetype);

    for(int i = 0; i < filetype.typemap_len; i++){
        if(etype != filetype.typemap[i].basic_type){
            return MPI_ERR_UNSUPPORTED_DATAREP;
        }
    }

    int rank;
    MPI_Comm_rank(fh.comm, &rank);
    if((fh.amode & MPI_MODE_SEQUENTIAL) && disp == MPI_DISPLACEMENT_CURRENT){
        int position;
        MPI_File_get_position(fh, &position);
        disp = position/gpu_mpi::TypeSize(etype);
    }
    fh.views[rank] = MPI_File_View(disp, etype, filetype, datarep);
    // printf("-- SET VIEW -- disp: %d, file_disp: %d, rank, %d\n", disp, fh.views[rank].disp, rank);
    // resets the individual file pointers and the shared file pointer to zero
    MPI_File_seek(fh, 0, MPI_SEEK_SET);

    return MPI_SUCCESS;
}

__device__ int MPI_File_get_view(MPI_File fh, MPI_Offset *disp, MPI_Datatype *etype, MPI_Datatype *filetype, char *datarep){
    int rank;
    MPI_Comm_rank(fh.comm, &rank);

    MPI_File_View view = fh.views[rank];
    *disp = view.disp;
    *etype = view.etype;
    *filetype = view.filetype;
    if(view.datarep){
        int datarep_size = 0;
        while (view.datarep[datarep_size] != '\0') datarep_size++;
        memcpy(datarep , view.datarep, datarep_size+1);
    }

    return MPI_SUCCESS;
}


/* ------DATA ACCESS------ */

__device__ int MPI_File_seek(MPI_File fh, MPI_Offset offset, int whence){
    if(fh.amode & MPI_MODE_SEQUENTIAL){
        return MPI_ERR_UNSUPPORTED_OPERATION;
    }

    int rank;
    MPI_Comm_rank(fh.comm, &rank);
    int view_pos = -1;
    if(whence == MPI_SEEK_SET){
        view_pos = offset;
    }else if(whence == MPI_SEEK_CUR){
        view_pos = __file_pos_to_view_pos(fh.views[rank], fh.seek_pos[rank]) + offset;
    }else if(whence == MPI_SEEK_END){
        view_pos = __get_file_size(fh.file) + offset;
    }
    if (view_pos < 0){
        return MPI_ERR_UNSUPPORTED_OPERATION;
    }

    int new_offset = __view_pos_to_file_pos(fh.views[rank], view_pos);
    if(new_offset < 0){
        // see documentation p521 line 11
        return MPI_ERR_UNSUPPORTED_OPERATION;
    }
    fh.seek_pos[rank] = new_offset;
    
    // view layout related
    if(!(fh.views[rank].layout_len == 1 && fh.views[rank].filetype.typemap_gap == 0)){
        // not contigues
        int pos = view_pos % fh.views[rank].filetype.typemap_len;
        for(int i=0; i < fh.views[rank].layout_len; i++){
            if(pos < fh.views[rank].layout[i].count){
                fh.views[rank].layout_cur_idx = i;
                fh.views[rank].layout_cur_disp = pos;
                break;
            }
            pos -= fh.views[rank].layout[i].count;
        }
    }

    return MPI_SUCCESS;
}

__device__ int MPI_File_get_position(MPI_File fh, MPI_Offset *offset){
    int rank;
    MPI_Comm_rank(fh.comm, &rank);
    int file_offset = __file_pos_to_view_pos(fh.views[rank], fh.seek_pos[rank]);
    if(file_offset < 0){
        return MPI_ERR_UNSUPPORTED_OPERATION;
    }
    *offset = file_offset;
    return MPI_SUCCESS;
}

#if USE_BUFFER && USE_VIEW_LAYOUT

__device__ int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    if (!(fh.amode & MPI_MODE_RDONLY) && !(fh.amode & MPI_MODE_RDWR)) return MPI_ERR_AMODE;
    if (fh.amode & MPI_MODE_SEQUENTIAL) return MPI_ERR_UNSUPPORTED_OPERATION;  // p514 l43

    int rank;
    MPI_Comm_rank(fh.comm, &rank);
    if (datatype != fh.views[rank].etype) return MPI_ERR_UNSUPPORTED_DATAREP;
    // TODO: Only one thread with RDWR can gain access; unlimited threads with RDONLY can gain access (?)
    // TODO: write into MPI_Status

    // If seek_pos smaller than the start of the valid area of the thread's view, then seek to the beginning
    if(fh.seek_pos[rank] < fh.views[rank].disp) {
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
    }
    int cur_pos = fh.seek_pos[rank];

    int read_size = 0;
    MPI_File_View thread_view = fh.views[rank];
    if(thread_view.layout_len == 1 && thread_view.filetype.typemap_gap == 0){
        // contiguous, no gap
        read_size = __read_buffer(fh, buf, datatype.size(), count, cur_pos);
    }else{
        // with gap
        int idx = thread_view.layout_cur_idx;
        int read_count = 0;
        int seek_pos = cur_pos;
        size_t etype_size = datatype.size();
        if(thread_view.isBegin()){
            // if not at the file beginning, need to add gap
            seek_pos += (seek_pos == thread_view.disp ? 0 : thread_view.filetype.typemap_gap);
        } else {
            // calculate the "seek_pos" for the layout beginning
            seek_pos -= (thread_view.layout[idx].disp - thread_view.layout[0].disp + thread_view.layout_cur_disp * etype_size);
        }
        
        if(thread_view.layout_cur_disp != 0){
            int count_to_read = count > thread_view.layout[idx].count - thread_view.layout_cur_disp ? thread_view.layout[idx].count - thread_view.layout_cur_disp : count;
            read_size += __read_buffer(fh, buf, etype_size, count_to_read, seek_pos+thread_view.layout[idx].disp+thread_view.layout_cur_disp*etype_size);
            read_count += count_to_read;
            idx++;
            if(idx == thread_view.layout_len){
                seek_pos += thread_view.layout[idx-1].disp + thread_view.layout[idx-1].count * etype_size + thread_view.filetype.typemap_gap;
                idx %= thread_view.layout_len;
            }
        }

        while(read_count < count){
            int count_to_read = count - read_count < thread_view.layout[idx].count ? count - read_count : thread_view.layout[idx].count;
            read_size += __read_buffer(fh, (char*)buf + read_count * etype_size, etype_size, count_to_read, seek_pos+thread_view.layout[idx].disp);
            read_count += thread_view.layout[idx].count;

            idx++;
            if(idx == thread_view.layout_len){
                seek_pos += thread_view.layout[idx-1].disp + thread_view.layout[idx-1].count * etype_size + thread_view.filetype.typemap_gap;
                idx %= thread_view.layout_len;
            }
        }
    }
    // assume we always can read count data
    MPI_File_seek(fh, read_size*datatype.size(), MPI_SEEK_CUR);
    
    return read_size;
}

__device__ int MPI_File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    // TODO: check amode
    if (fh.amode & MPI_MODE_RDONLY) return MPI_ERR_READ_ONLY;
    if (!(fh.amode & MPI_MODE_WRONLY) && !(fh.amode & MPI_MODE_RDWR)) return MPI_ERR_AMODE;

    int rank;
    MPI_Comm_rank(fh.comm, &rank);
    if (datatype != fh.views[rank].etype) return MPI_ERR_UNSUPPORTED_DATAREP;
    // write into buffer
    // If seek_pos smaller than the start of the valid area of the thread's view, then seek to the beginning
    if(fh.seek_pos[rank] < fh.views[rank].disp) {
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
    }
    int cur_pos = fh.seek_pos[rank];

    int write_size = 0;
    MPI_File_View thread_view = fh.views[rank];
    if(thread_view.layout_len == 1 && thread_view.filetype.typemap_gap == 0){
        // contiguous, no gap
        write_size = __write_buffer(fh, buf, datatype.size(), count, cur_pos);
    }else{
        // with gap
        int idx = thread_view.layout_cur_idx;
        int write_count = 0;
        int seek_pos = cur_pos;
        size_t etype_size = datatype.size();
        if(thread_view.isBegin()){
            // if not at the file beginning, need to add gap
            seek_pos += (seek_pos == thread_view.disp ? 0 : thread_view.filetype.typemap_gap);
        } else {
            // calculate the "seek_pos" for the layout beginning
            seek_pos -= (thread_view.layout[idx].disp - thread_view.layout[0].disp + thread_view.layout_cur_disp * etype_size);
        }

        if(thread_view.layout_cur_disp != 0){
            int count_to_write = count > thread_view.layout[idx].count - thread_view.layout_cur_disp ? thread_view.layout[idx].count - thread_view.layout_cur_disp : count;
            write_size += __write_buffer(fh, buf, etype_size, count_to_write, seek_pos+thread_view.layout[idx].disp+thread_view.layout_cur_disp*etype_size);
            write_count += count_to_write;
            idx++;
            if(idx == thread_view.layout_len){
                seek_pos += thread_view.layout[idx-1].disp + thread_view.layout[idx-1].count * etype_size + thread_view.filetype.typemap_gap;
                idx %= thread_view.layout_len;
            }
        }

        while(write_count < count){
            int count_to_write = count - write_count < thread_view.layout[idx].count ? count - write_count : thread_view.layout[idx].count;                        

            write_size += __write_buffer(fh, (char*)buf + write_count * etype_size, etype_size, count_to_write, seek_pos+thread_view.layout[idx].disp);
            write_count += thread_view.layout[idx].count;

            idx++;
            if(idx == thread_view.layout_len){
                seek_pos += thread_view.layout[idx-1].disp + thread_view.layout[idx-1].count * etype_size + thread_view.filetype.typemap_gap;
                idx %= thread_view.layout_len;
            }
        }
    }
    // assume we can always write count data
    MPI_File_seek(fh, write_size*datatype.size(), MPI_SEEK_CUR);

    return write_size;
}

# elif USE_BUFFER

__device__ int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    if (!(fh.amode & MPI_MODE_RDONLY) && !(fh.amode & MPI_MODE_RDWR)) return MPI_ERR_AMODE;
    if (fh.amode & MPI_MODE_SEQUENTIAL) return MPI_ERR_UNSUPPORTED_OPERATION;  // p514 l43

    int rank;
    MPI_Comm_rank(fh.comm, &rank);
    if (datatype != fh.views[rank].etype) return MPI_ERR_UNSUPPORTED_DATAREP;

    if(fh.seek_pos[rank] < fh.views[rank].disp) {
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
    }
    int seek_pos = fh.seek_pos[rank];

    MPI_File_View thread_view = fh.views[rank];
    int read_size = 0;

    int typemap_offset_idx = seek_pos % thread_view.filetype.typemap_len;
    if(typemap_offset_idx==0){
        // if not at the file beginning, need to add gap
        seek_pos += (seek_pos == thread_view.disp ? 0 : thread_view.filetype.typemap_gap);
    } else {
        // calculate the "seek_pos" for the layout beginning
        seek_pos -= thread_view.filetype.typemap[typemap_offset_idx].disp;
    }

    int read_count = 0;
    size_t etype_size = datatype.size();
    while(read_count < count){
        read_size += __read_buffer(fh, (char*)buf + read_count * etype_size, etype_size, 1, seek_pos+thread_view.filetype.typemap[typemap_offset_idx].disp);
        read_count++;

        typemap_offset_idx++;
        if(typemap_offset_idx == thread_view.filetype.typemap_len){
            seek_pos += thread_view.filetype.size();
            typemap_offset_idx %= thread_view.filetype.typemap_len;
        }
    }
    MPI_File_seek(fh, read_size*datatype.size(), MPI_SEEK_CUR);
    return read_size;
}

__device__ int MPI_File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    if (fh.amode & MPI_MODE_RDONLY) return MPI_ERR_READ_ONLY;
    if (!(fh.amode & MPI_MODE_WRONLY) && !(fh.amode & MPI_MODE_RDWR)) return MPI_ERR_AMODE;

    int rank;
    MPI_Comm_rank(fh.comm, &rank);
    if (datatype != fh.views[rank].etype) return MPI_ERR_UNSUPPORTED_DATAREP;

    if(fh.seek_pos[rank] < fh.views[rank].disp) {
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
    }
    int seek_pos = fh.seek_pos[rank];

    int write_size = 0;
    MPI_File_View thread_view = fh.views[rank];
    
    int typemap_offset_idx = seek_pos % thread_view.filetype.typemap_len;
    if(typemap_offset_idx==0){
        // if not at the file beginning, need to add gap
        seek_pos += (seek_pos == thread_view.disp ? 0 : thread_view.filetype.typemap_gap);
    } else {
        // calculate the "seek_pos" for the layout beginning
        seek_pos -= thread_view.filetype.typemap[typemap_offset_idx].disp;
    }

    int write_count = 0;
    size_t etype_size = datatype.size();
    while(write_count < count){
        write_size += __read_buffer(fh, (char*)buf + write_count * etype_size, etype_size, 1, seek_pos+thread_view.filetype.typemap[typemap_offset_idx].disp);
        write_count++;

        typemap_offset_idx++;
        if(typemap_offset_idx == thread_view.filetype.typemap_len){
            seek_pos += thread_view.filetype.size();
            typemap_offset_idx %= thread_view.filetype.typemap_len;
        }
    }
    MPI_File_seek(fh, write_size*datatype.size(), MPI_SEEK_CUR);
    return write_size;
}

# else

__device__ int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    // nb. buf is in device's address space, cannot be accessed directly by host
    if (!(fh.amode & MPI_MODE_RDONLY) && !(fh.amode & MPI_MODE_RDWR)) return MPI_ERR_AMODE;
    if (fh.amode & MPI_MODE_SEQUENTIAL) return MPI_ERR_UNSUPPORTED_OPERATION;  // p514 l43
    int rank;
    MPI_Comm_rank(fh.comm, &rank);
    if (datatype != fh.views[rank].etype) return MPI_ERR_UNSUPPORTED_DATAREP;
    // TODO: Only one thread with RDWR can gain access; unlimited threads with RDONLY can gain access (?)
    // TODO: write into MPI_Status

    // assert(datatype == MPI_CHAR);  // TODO: adapt to different datatypes
    int buffer_size = 2048;  // sizeof(int) + sizeof(r_param) + sizeof(char) * (count + 1);: misaligned address
        // (TODO: dynamic size) sizeof(datatype) * (count + 1);
    char* data = (char*)allocate_host_mem(buffer_size);
    if (data == nullptr) return 0;

    // struct rw_params {
    //     MPI_File fh;
    //     MPI_Datatype datatype;
    //     void* buf;
    //     int count;
    // } r_param;
    // r_param.fh = fh;  r_param.datatype = datatype;  r_param.count = count;
    // r_param.buf = data + sizeof(int) + sizeof(r_param);  // CPU cannot directly write into buf, so write into mem first
    // *((int*)data) = I_FREAD;
    
    // If seek_pos smaller than the start of the valid area of the thread's view, then seek to the beginning
    if(fh.seek_pos[rank] < fh.views[rank].disp) {
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
    }
    // __rw_params r_params(I_READY,fh.file,datatype,data + sizeof(__rw_params),count,fh.seek_pos[rank]);
    // int layout_size = fh.views[rank].layout_len * sizeof(layout_segment);
    int layout_size = fh.views[rank].layout_len * sizeof(MPI_File_View::layout_segment);
    int seek_pos = fh.seek_pos[rank];
    size_t etype_size = datatype.size();
    
    MPI_File_View thread_view = fh.views[rank];
    if(!(thread_view.layout_len == 1 && thread_view.filetype.typemap_gap == 0)){
        int idx = thread_view.layout_cur_idx;
        if(thread_view.isBegin()){
            // if not at the file beginning, need to add gap
            seek_pos += (seek_pos == thread_view.disp ? 0 : thread_view.filetype.typemap_gap);
        } else {
            // calculate the "seek_pos" for the layout beginning
            seek_pos -= (thread_view.layout[idx].disp - thread_view.layout[0].disp + thread_view.layout_cur_disp * etype_size);
        }
    }

    __rw_params r_params(I_FREAD_BASIC,fh.file,
                        etype_size,
                        data + sizeof(__rw_params) + layout_size,
                        count,
                        seek_pos,
                        fh.views[rank].layout_len,
                        fh.views[rank].filetype.typemap_gap,
                        fh.views[rank].layout_cur_idx,
                        fh.views[rank].layout_cur_disp,
                        (MPI_File_View::layout_segment*)data + sizeof(__rw_params));
    *((__rw_params*)data) = r_params;
    memcpy(r_params.layout, fh.views[rank].layout, layout_size);
    // printf("READ POS: %d, rank %d\n", r_params.seek_pos, rank);
    
    delegate_to_host(data, buffer_size);
    while (*((int*)data) != I_READY)
    {
        // blocking wait (p506 l44)
    }

    memcpy(buf, r_params.buf, datatype.size() * count);
    size_t ret = ((size_t*)data)[1];
    MPI_File_seek(fh, ret*datatype.size(), MPI_SEEK_CUR);
    free_host_mem(data);
    return ret;
}

//not thread safe
__device__ int MPI_File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    if (fh.amode & MPI_MODE_RDONLY) return MPI_ERR_READ_ONLY;
    if (!(fh.amode & MPI_MODE_WRONLY) && !(fh.amode & MPI_MODE_RDWR)) return MPI_ERR_AMODE;
    int rank;
    MPI_Comm_rank(fh.comm, &rank);
    if (datatype != fh.views[rank].etype) return MPI_ERR_UNSUPPORTED_DATAREP;
    //TODO: dynamically assign buffer size
    int buffer_size = 2048;
    // int MPI_Type_size(MPI_Datatype datatype, int *size)
    //TODO: MPI_Type_size not implemented
    // assert(datatype==MPI_CHAR);
    assert(buffer_size > sizeof(int*)*2+sizeof(FILE**)+datatype.size()*count);
    //init
    char* data = (char*) allocate_host_mem(buffer_size);
    //assemble metadata TODO: why we need to re-seek every time? is there redundant seek?
    // If seek_pos smaller than the start of the valid area of the thread's view, then seek to the beginning
    if(fh.seek_pos[rank] < fh.views[rank].disp) {
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
    }
    // __rw_params w_params(I_FWRITE,fh.file,datatype,data + sizeof(__rw_params),count,fh.seek_pos[rank]);
    int layout_size = fh.views[rank].layout_len * sizeof(MPI_File_View::layout_segment);
    int seek_pos = fh.seek_pos[rank];
    size_t etype_size = datatype.size();

    MPI_File_View thread_view = fh.views[rank];
    if(!(thread_view.layout_len == 1 && thread_view.filetype.typemap_gap == 0)){
        int idx = thread_view.layout_cur_idx;
        if(thread_view.isBegin()){
            // if not at the file beginning, need to add gap
            seek_pos += (seek_pos == thread_view.disp ? 0 : thread_view.filetype.typemap_gap);
        } else {
            // calculate the "seek_pos" for the layout beginning
            seek_pos -= (thread_view.layout[idx].disp - thread_view.layout[0].disp + thread_view.layout_cur_disp * etype_size);
        }
    }

    __rw_params w_params(I_FWRITE_BASIC,fh.file,
                        etype_size,
                        data + sizeof(__rw_params) + layout_size,
                        count,
                        seek_pos,
                        fh.views[rank].layout_len,
                        fh.views[rank].filetype.typemap_gap,
                        fh.views[rank].layout_cur_idx,
                        fh.views[rank].layout_cur_disp,
                        (MPI_File_View::layout_segment*)data + sizeof(__rw_params));
    //embed metadata
    *((__rw_params*)data) = w_params;
    // memcpy(data, (void*)&w_params, sizeof(__rw_params));
    //embed data
    memcpy(w_params.layout, fh.views[rank].layout, layout_size);
    memcpy(w_params.buf, buf, datatype.size()*count);
    // printf("WRITE POS: %d, rank, %d\n", w_params.seek_pos, rank);

    //execute on CPU
    delegate_to_host((void*)data, buffer_size);
    // wait
    while(((int*)data)[0] != I_READY){};
    size_t return_value = ((size_t*)data)[1];
    //TODO: assuming individual file pointer, but how does shared pointer differ from this?
    MPI_File_seek(fh, return_value*datatype.size(), MPI_SEEK_CUR);
    free_host_mem(data);
    //TODO: step 4 error catching
    //#memory cosistency: assuming that write is not reordered with write
    return return_value;
}

#endif

__device__ int MPI_File_read_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    MPI_Offset old_offset;
    int ret;
    ret = MPI_File_get_position(fh, &old_offset);
    if(ret != 0){
        return ret;
    }
    ret = MPI_File_seek(fh, offset, MPI_SEEK_SET);
    if(ret != 0){
        return ret;
    }
    ret = MPI_File_read(fh, buf, count, datatype, status);
    if(ret != 0){
        return ret;
    }
    ret = MPI_File_seek(fh, old_offset, MPI_SEEK_SET);
    if(ret != 0){
        return ret;
    }
    return 0;
}

__device__ int MPI_File_write_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    MPI_Offset old_offset;
    int ret;
    ret = MPI_File_get_position(fh, &old_offset);
    if(ret != 0){
        return ret;
    }
    ret = MPI_File_seek(fh, offset, MPI_SEEK_SET);
    if(ret != 0){
        return ret;
    }
    ret = MPI_File_write(fh, buf, count, datatype, status);
    if(ret != 0){
        return ret;
    }
    ret = MPI_File_seek(fh, old_offset, MPI_SEEK_SET);
    if(ret != 0){
        return ret;
    }
    return 0;
}

__device__ unsigned int shared_read_lock = 0;
__device__ int MPI_File_read_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    mutex_lock(&shared_read_lock);
    MPI_Offset old_ind_offset;
    int ret;
    ret = MPI_File_get_position(fh, &old_ind_offset);
    if(ret != 0){
        mutex_unlock(&shared_read_lock);
        return ret;
    }
    ret = MPI_File_seek(fh, *fh.shared_seek_pos, MPI_SEEK_SET);
    if(ret != 0){
        mutex_unlock(&shared_read_lock);
        return ret;
    }
    // update the shared seek position
    int res = MPI_File_read(fh, buf, count, datatype, status);
    (*fh.shared_seek_pos) += res;
    // reset individual seek position
    ret = MPI_File_seek(fh, old_ind_offset, MPI_SEEK_SET);
    if(ret != 0){
        mutex_unlock(&shared_read_lock);
        return ret;
    }
    mutex_unlock(&shared_read_lock);
    return 0;
}

__device__ unsigned int shared_write_lock = 0;
__device__ int MPI_File_write_shared(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    mutex_lock(&shared_write_lock);
    MPI_Offset old_ind_offset;
    int ret;
    ret = MPI_File_get_position(fh, &old_ind_offset);
    if(ret != 0){
        mutex_unlock(&shared_write_lock);
        return ret;
    }
    ret = MPI_File_seek(fh, *fh.shared_seek_pos, MPI_SEEK_SET);
    if(ret != 0){
        mutex_unlock(&shared_read_lock);
        return ret;
    }
    // update the shared seek position
    int res = MPI_File_write(fh, buf, count, datatype, status);
    (*fh.shared_seek_pos) += res;
    // reset individual seek position
    ret = MPI_File_seek(fh, old_ind_offset, MPI_SEEK_SET);
    if(ret != 0){
        mutex_unlock(&shared_write_lock);
        return ret;
    }
    mutex_unlock(&shared_write_lock);
    return 0;
}