#include "libc_processor.cuh"
#include "../gpu_mpi/io.cuh"

#include <cassert>
#include <cstdio>

int get_i_flag(void* mem){
    return ((int*)mem)[0];
}

void set_i_ready_flag(void* mem){
    ((int*)mem)[0] = I_READY;
}

void process_gpu_libc(void* mem, size_t size) {
    // intend to change to switch-case statement, but need to inform everyone to remember break;
    // switch(get_i_flag(mem)){
    //     case I_FSEEK: {
    //         FILE* file = ((FILE**)mem)[1];
    //         // todo: is lock needed?
    //         fseek(file, 0L, SEEK_END);
    //         ((long int*)mem)[1] = ftell(file);
    //         set_i_ready_flag(mem);
    //         return; //intentionally use return instead of break
    //     }
    //     case I_FWRITE: {
    //         FILE* file = ((FILE**)mem)[1]; // encoded FILE* retrieve with (FILE* )*
    //         MPI_Datatype* datatype = ((MPI_Datatype*)mem)[2];
    //         //TODO: MPI_Type_size not implemented
    //         assert(datatype==MPI_CHAR);
    //         fwrite( ((char*)mem)[3], sizeof(char), size-sizeof(MPI_Datatype*)-sizeof(FILE*)-sizeof(int*), file);
    //         // printf("cjc\n");
    //         // FILE* file = fopen("cjc","w");
    //         // char *a = "hello world";
    //         // printf("cjc\n");
    //         // fwrite(a,sizeof(char),11,file);
    //         // fclose(file);
    //         break;
    //     }
    //     case I_FFLUSH: {
    //         FILE* file = ((FILE**)mem)[1];
    //         fflush(file);
    //         set_i_ready_flag(mem);
    //         break;
    //     }
    //     case I_FCLOSE:{
    //         FILE* file = ((FILE**)mem)[1];
    //         fclose(file);
    //         set_i_ready_flag(mem);
    //         break;
    //     }
    // }
    if(get_i_flag(mem) == I_FSEEK){
        FILE* file = ((FILE**)mem)[1];
        // todo: is lock needed?
        fseek(file, 0L, SEEK_END);
        ((long int*)mem)[1] = ftell(file);
        set_i_ready_flag(mem);
    }
    if(get_i_flag(mem) == I_FFLUSH){
        FILE* file = ((FILE**)mem)[1];
        fflush(file);
        set_i_ready_flag(mem);
    }
    if(get_i_flag(mem) == I_FCLOSE){
        FILE* file = ((FILE**)mem)[1];
        fclose(file);
        set_i_ready_flag(mem);
    }
    if(get_i_flag(mem) == I_FWRITE){
        FILE* file = *((FILE**)(mem+8)); // encoded FILE* retrieve with (FILE* )*
        int count = *((int*)(mem+4));
        // __show_memory((char *)mem,64);
        MPI_Datatype datatype = *((MPI_Datatype *)(mem+16));

        //TODO: MPI_Type_size not implemented
        assert(datatype==MPI_CHAR);
        ((size_t*)mem)[1] = fwrite( ((const char**)mem+24), sizeof(char), count, file);
        set_i_ready_flag(mem);
    }
    if(get_i_flag(mem) == I_FOPEN){
        int mode_flag = ((int*)mem)[1];
        const char* mode;
        if(mode_flag == I_FOPEN_MODE_RD){
            mode = "r";
        }else if (mode_flag == I_FOPEN_MODE_RW) {
            mode = "w+";
        }else if (mode_flag == I_FOPEN_MODE_WD) {
            mode = "w";
        }else if (mode_flag == I_FOPEN_MODE_RW_APPEND) {
            mode = "a+";
        }else if (mode_flag == I_FOPEN_MODE_WD_APPEND) {
            mode = "a";
        }
        const char* filename = (const char*)((const char**)mem + 2);
        FILE* file = fopen(filename, mode);
        ((FILE**)mem)[1] = file;
        set_i_ready_flag(mem);
    }
    if(get_i_flag(mem) == I_FREAD){
        FILE* file = ((FILE**)mem)[1];
        void* buf = ((void**)mem)[2];
        int seekpos = ((int*)mem)[6];
        int ret = fseek(file, seekpos, SEEK_SET);
        assert(ret == 0);
        int cnt = fread(buf, 1, INIT_BUFFER_BLOCK_SIZE, file);
        assert(cnt == INIT_BUFFER_BLOCK_SIZE);
        set_i_ready_flag(mem);
    }
    if(get_i_flag(mem) == I_FDELETE){
        const char* filename = (const char*)((const char**)mem + 1);
        int res = remove(filename);
        set_i_ready_flag(mem);
    }
}
