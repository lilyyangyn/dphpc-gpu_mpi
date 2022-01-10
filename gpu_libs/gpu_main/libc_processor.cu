#include "libc_processor.cuh"
#include "../gpu_mpi/io.cuh"

#include <cassert>
#include <cstdio>
#include <errno.h>
#include <aio.h>

int get_i_flag(void* mem){
    return ((int*)mem)[0];
}

void set_i_ready_flag(void* mem){
    ((int*)mem)[0] = I_READY;
}

int libc_iread_posixaio(void* mem, size_t size){
    char * data = (char *)mem;

    aiocb cb;
    memset(&cb, 0, sizeof(aiocb));

    char * p = data;
    p += 8; cb.aio_fildes = fileno( *((FILE**)p) );
    p += 8; cb.aio_nbytes = *((size_t*)p);
    p += 8; cb.aio_offset = *((off_t*)p);
    p += 8; cb.aio_buf = *((void**)p);

    //TODO: error checking
    int ret = aio_read(&cb);
    p = (char*)mem+8;
    memcpy((aiocb*)p,&cb,sizeof(aiocb));

    // *((aiocb*)p) = cb; //send cb back.
    return ret;
}

int libc_itest_posixaio(void* mem, size_t size){
    char *p = (char*)mem;
    p+=8;
    aiocb cb = *((aiocb*)p);
    aiocb* list_p[1] = {&cb};
    int ret = aio_suspend(list_p,1,0);
    printf("%s", (char*)cb.aio_buf);
    return ret;
    
    // while(aio_error(&cb) == EINPROGRESS){printf("working\n");}
    // return 0;
}

void libc_iread_io(void* mem, size_t size){
    
}

void process_gpu_libc(void* mem, size_t size) {

    if(get_i_flag(mem) == I_FOPEN){
        int amode = ((int*)mem)[1];
        const char* filename = (const char*)((const char**)mem + 2);
        FILE* file = fopen(filename, "rb");
        int file_exist = 1;
        if(file == NULL){
            file_exist = 0;
        }
        if(!(amode & MPI_MODE_RDONLY)){
            if(file_exist == 1) fclose(file);
            if(amode & MPI_MODE_RDWR){
                if(amode & MPI_MODE_APPEND) {
                    file = fopen(filename, "ab+");
                }else{
                    if(file_exist == 0){
                        file = fopen(filename, "wb");
                        fclose(file);
                    }
                    file = fopen(filename, "rb+");
                }
            }else if(amode & MPI_MODE_WRONLY){
                if(amode & MPI_MODE_APPEND) {
                    file = fopen(filename, "ab");
                }else{
                    file = fopen(filename, "wb");
                }
            }
        }
        ((int*)mem)[1] = file_exist;
        ((FILE**)mem)[2] = file;
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
        if(r_params.layout_count == 1 && r_params.layout_gap == 0){
            // contiguous, no gap
            fseek(r_params.file,r_params.seek_pos,SEEK_SET);
            ((size_t*)mem)[1] = fread(r_params.buf, r_params.etype_size, r_params.count, r_params.file);
            // p507 l42
            // nb. fread() forwards the file pointer, so no need to manually forward it.
        } else {
            // with gap
            int read_count = 0;
            int idx = r_params.layout_cur_idx;
            int seek_pos = r_params.seek_pos;
            size_t read_size = 0;

            if(r_params.layout_cur_disp != 0){
                int count_to_read = r_params.count > r_params.layout[idx].count - r_params.layout_cur_disp ? r_params.layout[idx].count - r_params.layout_cur_disp : r_params.count;
                fseek(r_params.file, seek_pos+r_params.layout[idx].disp+r_params.layout_cur_disp*r_params.etype_size, SEEK_SET);
                read_size += fread(r_params.buf, r_params.etype_size, count_to_read, r_params.file);
                read_count += count_to_read;
                idx++;
                if(idx == r_params.layout_count){
                    seek_pos += r_params.layout[idx-1].disp + r_params.layout[idx-1].count * r_params.etype_size + r_params.layout_gap;
                    idx %= r_params.layout_count;
                }
            }

            while(read_count < r_params.count){
                int count_to_read = r_params.count - read_count < r_params.layout[idx].count ? r_params.count - read_count : r_params.layout[idx].count;

                fseek(r_params.file, seek_pos+r_params.layout[idx].disp, SEEK_SET);
                read_size += fread((char*)r_params.buf + read_count * r_params.etype_size, r_params.etype_size, count_to_read, r_params.file);
                read_count += r_params.layout[idx].count;

                idx++;
                if(idx == r_params.layout_count){
                    seek_pos += r_params.layout[idx-1].disp + r_params.layout[idx-1].count * r_params.etype_size + r_params.layout_gap;
                    idx %= r_params.layout_count;
                }
            }
            ((size_t*)mem)[1] = read_size;
        }
        set_i_ready_flag(mem);
    }

    else if(get_i_flag(mem) == I_FWRITE_BASIC){
        char * data = (char *)mem;
        __rw_params w_params = *((__rw_params*)data);
        if(w_params.layout_count == 1 && w_params.layout_gap == 0){
            // contiguous, no gap
            fseek(w_params.file,w_params.seek_pos,SEEK_SET);
            ((size_t*)mem)[1] = fwrite( w_params.buf, w_params.etype_size, w_params.count, w_params.file);
        } else {
            // with gap
            int written_count = 0;
            int idx = w_params.layout_cur_idx;
            int seek_pos = w_params.seek_pos;
            size_t written_size = 0;

            if(w_params.layout_cur_disp != 0){
                int count_to_write = w_params.count > w_params.layout[idx].count - w_params.layout_cur_disp ? w_params.layout[idx].count - w_params.layout_cur_disp : w_params.count;
                fseek(w_params.file, seek_pos+w_params.layout[idx].disp+w_params.layout_cur_disp*w_params.etype_size, SEEK_SET);
                written_size += fwrite( w_params.buf, w_params.etype_size, count_to_write, w_params.file);
                written_count += count_to_write;
                idx++;
                if(idx == w_params.layout_count){
                    seek_pos += w_params.layout[idx-1].disp + w_params.layout[idx-1].count * w_params.etype_size + w_params.layout_gap;
                    idx %= w_params.layout_count;
                }
            }

            while(written_count < w_params.count){
                int count_to_write = w_params.count - written_count < w_params.layout[idx].count ? w_params.count - written_count : w_params.layout[idx].count;

                fseek(w_params.file, seek_pos+w_params.layout[idx].disp, SEEK_SET);
                written_size += fwrite( (char*)w_params.buf + written_count * w_params.etype_size, w_params.etype_size, count_to_write, w_params.file);
                written_count += w_params.layout[idx].count;
                // printf("Written_count: %d, count: %d, pos: %d, disp: %d\n", written_count, w_params.layout[idx].count, seek_pos+w_params.layout[idx].disp, w_params.layout[idx].disp);
                
                idx++;
                if(idx == w_params.layout_count){
                    seek_pos += w_params.layout[idx-1].disp + w_params.layout[idx-1].count * w_params.etype_size + w_params.layout_gap;
                    idx %= w_params.layout_count;
                }
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
        int writebytes = ((int*)mem)[7];
        // printf("file %p, buf %p, seekpos %d\n", file, buf, seekpos);
        // fflush(stdout);
        int ret = fseek(file, seekpos, SEEK_SET);
        // perror("Error:");
        // printf("ret: %d\n", ret);
        assert(ret == 0);
        int cnt = fwrite(buf, 1, writebytes, file);
        ((size_t*)mem)[1] = cnt;
        // assert(cnt == INIT_BUFFER_BLOCK_SIZE);
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

//  design a prefix or some binary encoding.
    else if(get_i_flag(mem) == I_FILE_IREAD){
        ((int*)mem)[1] = libc_iread_posixaio(mem,size);
        set_i_ready_flag(mem);
    }

    else if(get_i_flag(mem) == I_FILE_ITEST){
        ((int*)mem)[1] = libc_itest_posixaio(mem,size); 
        set_i_ready_flag(mem);
    }
}
