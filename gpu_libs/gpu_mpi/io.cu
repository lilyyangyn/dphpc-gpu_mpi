#include "io.cuh"
#include "mpi_common.cuh"
#include "datatypes.cuh"
#include "../gpu_main/device_host_comm.cuh"
#include <cassert>

// #include "mpi.cuh"
#define N 100
namespace gpu_mpi {
}

/* ------HELPER FUNCTIONS------ */

__device__ void mutex_lock(unsigned int *mutex) {
    unsigned int ns = 8;
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
    printf("C\n");
    delegate_to_host((void*)data, buffer_size);
    // wait
    while(((int*)data)[0] != I_READY){};
    //file remove done
    int res = ((int*)data)[1];
    free_host_mem(data);

    return res;
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

/* ------FILE MANIPULATION------ */

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
                shared_fh.shared_seek_pos = (int*)malloc(sizeof(int));
                *(shared_fh.shared_seek_pos) = init_pos;
                shared_fh.views = (MPI_File_View*)malloc(size*sizeof(MPI_File_View)); 
                for (int i = 0; i < size; i++){
                    shared_fh.seek_pos[i] = init_pos;
                    shared_fh.views[i] = MPI_File_View(0, MPI_CHAR, MPI_CHAR, "native");
                }               
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
        // TODO: file not exist error
        int res = __delete_file(filename);
        err_code = (res == 0) ? MPI_SUCCESS : MPI_ERR_IO; 
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&err_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return err_code;
}

__device__ int MPI_File_get_size(MPI_File fh, MPI_Offset *size){
    int sz; 

    int rank;
    MPI_Comm_rank(fh.comm, &rank);
    if(rank == 0){
        sz = __get_file_size(fh.file);
    }

    MPI_Barrier(fh.comm);
    MPI_Bcast(&sz, 1, MPI_INT, 0, fh.comm);
    *size = sz;

    return MPI_SUCCESS;
} 

__device__ int MPI_File_close(MPI_File *fh){
    // synchronize file state
    // __syncthreads();
    MPI_Barrier(fh->comm);

    int rank;
    MPI_Comm_rank(fh->comm, &rank);

    // only free the file handle object once
    if(rank == 0){
        // close the file associated with file handle
        // fclose(fh->file);
        
        // release the fh object
        free(fh->seek_pos);
        free(fh->views);

        __close_file(fh->file);
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
        assert(etype == filetype.typemap[i].basic_type);
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
    int new_offset = -1;
    if(whence == MPI_SEEK_SET){
        new_offset = __view_pos_to_file_pos(fh.views[rank], offset);       
    }else if(whence == MPI_SEEK_CUR){
        new_offset = __view_pos_to_file_pos(fh.views[rank], fh.seek_pos[rank] + offset);
    }else if(whence == MPI_SEEK_END){
        int sz = __get_file_size(fh.file);
        new_offset = __view_pos_to_file_pos(fh.views[rank], sz + offset);
    }

    if(new_offset < 0){
        // see documentation p521 line 11
        return MPI_ERR_UNSUPPORTED_OPERATION;
    }
    fh.seek_pos[rank] = new_offset;

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

__device__ int MPI_File_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    // nb. buf is in device's address space, cannot be accessed directly by host

    if (!(fh.amode & MPI_MODE_RDONLY) && !(fh.amode & MPI_MODE_RDWR)) return MPI_ERR_AMODE;
    if (fh.amode & MPI_MODE_SEQUENTIAL) return MPI_ERR_UNSUPPORTED_OPERATION;  // p514 l43
    int rank;
    MPI_Comm_rank(fh.comm, &rank);
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
    int layout_size = fh.views[rank].layout_len * sizeof(layout_segment);
    __rw_params r_params(I_FREAD,fh.file,
                        datatype.size(),
                        data + sizeof(__rw_params) + layout_size,
                        count,
                        fh.seek_pos[rank],
                        fh.views[rank].layout_len,
                        fh.views[rank].filetype.typemap_gap,
                        (layout_segment*)data + sizeof(__rw_params));
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
    // fh.seek_pos[rank]+=ret;
    MPI_File_seek(fh, ret, MPI_SEEK_CUR);
    free_host_mem(data);
    return ret;
}

//not thread safe
__device__ int MPI_File_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status){
    int rank;
    MPI_Comm_rank(fh.comm, &rank);
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
    int layout_size = fh.views[rank].layout_len * sizeof(layout_segment);
    __rw_params w_params(I_FWRITE,fh.file,
                        datatype.size(),
                        data + sizeof(__rw_params) + layout_size,
                        count,
                        fh.seek_pos[rank],
                        fh.views[rank].layout_len,
                        fh.views[rank].filetype.typemap_gap,
                        (layout_segment*)data + sizeof(__rw_params));
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
    int return_value = (int) *((size_t*)(data+8));
    //TODO: assuming individual file pointer, but how does shared pointer differ from this?
    // fh.seek_pos[rank]+=return_value;
    MPI_File_seek(fh, return_value, MPI_SEEK_CUR);
    free_host_mem(data);
    //TODO: step 4 error catching
    //#memory cosistency: assuming that write is not reordered with write
    return return_value;
}

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
