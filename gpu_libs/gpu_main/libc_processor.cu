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
        char * data = (char *)mem;
        __rw_params w_params = *((__rw_params*)data);


        // FILE* file = *((FILE**)(mem+8)); // encoded FILE* retrieve with (FILE* )*
        // int count = *((int*)(mem+4));
        // // __show_memory((char *)mem,64);
        // MPI_Datatype datatype = *((MPI_Datatype *)(mem+16));

        // //TODO: MPI_Type_size not implemented
        assert(w_params.datatype==MPI_CHAR);
        //seek pos
        fseek(w_params.file,w_params.seek_pos,SEEK_SET);
        ((size_t*)mem)[1] = fwrite( w_params.buf, sizeof(char), w_params.count, w_params.file);
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
        // This param order is in accordance with FWRITE
        struct rw_params {
            MPI_File fh;
            MPI_Datatype datatype;
            void* buf;
            int count;
        } r_param = *((rw_params*)((char*)mem + sizeof(int)));

        ((size_t*)mem)[1] = fread(r_param.buf, sizeof(char), r_param.count, r_param.fh.file);
        // p507 l42
        // nb. fread() forwards the file pointer, so no need to manually forward it.
        
        set_i_ready_flag(mem);
    }
}
