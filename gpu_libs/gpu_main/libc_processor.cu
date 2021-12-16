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

    else if(get_i_flag(mem) == I_FSEEK){
        FILE* file = ((FILE**)mem)[1];
        // todo: is lock needed?
        fseek(file, 0L, SEEK_END);
        ((long int*)mem)[1] = ftell(file);
        set_i_ready_flag(mem);
    }

    else if(get_i_flag(mem) == I_FREAD_BASIC){
        // This param order is in accordance with FWRITE
        // struct rw_params {
        //     MPI_File fh;
        //     MPI_Datatype datatype;
        //     void* buf;
        //     int count;
        // } r_param = *((rw_params*)((char*)mem + sizeof(int)));
        __rw_params r_params = *((__rw_params*)mem);
        if(r_params.layout_count == 1){
            // contiguous, no gap
            fseek(r_params.file,r_params.seek_pos,SEEK_SET);
            ((size_t*)mem)[1] = fread(r_params.buf, r_params.etype_size, r_params.count, r_params.file);
            // p507 l42
            // nb. fread() forwards the file pointer, so no need to manually forward it.
        } else {
            // with gap
            int read_count = 0;
            int idx = -1;
            int seek_pos = r_params.seek_pos;
            size_t read_size = 0;
            while(read_count < r_params.count){
                idx++;
                if(idx == r_params.layout_count){
                    seek_pos += r_params.layout[idx-1].disp + r_params.layout[idx-1].count * r_params.etype_size + r_params.layout_gap;
                    idx %= r_params.layout_count;

                }

                fseek(r_params.file, seek_pos+r_params.layout[idx].disp, SEEK_SET);
                read_size += fread((char*)r_params.buf + read_count * r_params.etype_size, r_params.etype_size, r_params.layout[idx].count, r_params.file);
                read_count += r_params.layout[idx].count;
            }
            ((size_t*)mem)[1] = read_size;
        }
        set_i_ready_flag(mem);
    }

    else if(get_i_flag(mem) == I_FWRITE_BASIC){
        char * data = (char *)mem;
        __rw_params w_params = *((__rw_params*)data);
        if(w_params.layout_count == 1){
            // contiguous, no gap
            fseek(w_params.file,w_params.seek_pos,SEEK_SET);
            ((size_t*)mem)[1] = fwrite( w_params.buf, w_params.etype_size, w_params.count, w_params.file);
        } else {
            // with gap
            int written_count = 0;
            int idx = -1;
            int seek_pos = w_params.seek_pos;
            // printf("Written_pos: %d\n", seek_pos);
            size_t written_size = 0;
            while(written_count < w_params.count){
                idx++;
                if(idx == w_params.layout_count){
                    seek_pos += w_params.layout[idx-1].disp + w_params.layout[idx-1].count * w_params.etype_size + w_params.layout_gap;
                    idx %= w_params.layout_count;
                }

                fseek(w_params.file, seek_pos+w_params.layout[idx].disp, SEEK_SET);
                written_size += fwrite( (char*)w_params.buf + written_count * w_params.etype_size, w_params.etype_size, w_params.layout[idx].count, w_params.file);
                written_count += w_params.layout[idx].count;
                // printf("Written_count: %d, count: %d, pos: %d, disp: %d\n", written_count, w_params.layout[idx].count, seek_pos+w_params.layout[idx].disp, w_params.layout[idx].disp);
            }
            ((size_t*)mem)[1] = written_size;
        }

        set_i_ready_flag(mem);
    }

    else if(get_i_flag(mem) == I_FREAD_BUFFER){
        FILE* file = ((FILE**)mem)[1];
        void* buf = ((void**)mem)[2];
        int seekpos = ((int*)mem)[6];
        int ret = fseek(file, seekpos, SEEK_SET);
        assert(ret == 0);
        size_t cnt = fread(buf, 1, INIT_BUFFER_BLOCK_SIZE, file);
        //assert(cnt == INIT_BUFFER_BLOCK_SIZE);
        ((size_t*)mem)[1] = cnt;
        set_i_ready_flag(mem);
    }

    else if(get_i_flag(mem) == I_FWRITE_BUFFER){
        FILE* file = ((FILE**)mem)[1];
        const void* buf = ((void**)mem)[2];
        int seekpos = ((int*)mem)[6];
        // printf("file %p, buf %p, seekpos %d\n", file, buf, seekpos);
        // fflush(stdout);
        int ret = fseek(file, seekpos, SEEK_SET);
        // perror("Error:");
        // printf("ret: %d\n", ret);
        assert(ret == 0);
        int cnt = fwrite(buf, 1, INIT_BUFFER_BLOCK_SIZE, file);
        ((size_t*)mem)[1] = cnt;
        assert(cnt == INIT_BUFFER_BLOCK_SIZE);
        // printf("we write %d bytes: \n", cnt);
        set_i_ready_flag(mem);
    }

    else if(get_i_flag(mem) == I_FFLUSH){
        FILE* file = ((FILE**)mem)[1];
        fflush(file);
        set_i_ready_flag(mem);
    }

    else if(get_i_flag(mem) == I_FDELETE){
        const char* filename = (const char*)((const char**)mem + 1);
        int res = remove(filename);
        ((int*)mem)[1] = res;
        set_i_ready_flag(mem);
    }

    else if(get_i_flag(mem) == I_FCLOSE){
        FILE* file = ((FILE**)mem)[1];
        fclose(file);
        set_i_ready_flag(mem);
    }
    
    
    
    
    
}
