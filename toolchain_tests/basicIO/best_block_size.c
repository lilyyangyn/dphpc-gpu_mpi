#include <stdio.h>
#include <string.h>
#include "mpi.h"

#define SIZE 32768

int main(int argc, char *argv[])
{
    MPI_File fh;
    MPI_Info info;
    int rank;
    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_File_open(MPI_COMM_WORLD, "/home/xindzuo/dphpc-gpu_mpi/toolchain_tests/basicIO/testfile.txt", MPI_MODE_RDWR, info, &fh);
    double start, end;
    double tot_time = 0;
    double time;
    double total_time;

    char write_buf[SIZE];
    for(int i = 0; i < SIZE; i++){
        write_buf[i] = '0' + i % 8;
    }
    // write 10 times
    for(int i = 0; i < 100; i++){
        // each thread read a fixed region [rank*SIZE, rank*SIZE + SIZE - 1]
        MPI_File_seek(fh, rank * SIZE, MPI_SEEK_SET);
        start = MPI_Wtime();
        MPI_File_write(fh, write_buf, SIZE, MPI_CHAR, NULL);
        end = MPI_Wtime();
        time = end - start;
        tot_time += time;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Reduce(&tot_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        printf("the average time of write is: %f\n", total_time/4);
    }


    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}