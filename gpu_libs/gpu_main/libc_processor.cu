#include "libc_processor.cuh"

#include <stdio.h>

void process_gpu_libc(void* mem, size_t size) {

    // step 1, copy mem(int devide) to a host addr
    void* buf = malloc(size);
    cudaMemcpy(buf, mem, size, cudaMemcpyDeviceToHost); // bug

    // step 2, execute the instruction





    // FILE* file = fopen("/home/raliao/gpu_mpi/testfile", "w");
    // fwrite("data\n", 1, 5, file);



}
