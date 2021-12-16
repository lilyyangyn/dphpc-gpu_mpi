#include <stdio.h>
#include <string.h>
#include "mpi.h"

#define SIZE 4096

int main(int argc, char *argv[])
{
    MPI_File fh;
    MPI_Info info;
    int rank;
    int size;
    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_File_open(MPI_COMM_WORLD, "./testfile.txt", MPI_MODE_RDWR, info, &fh);
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
        printf("num of proc %d, the average time of write is: %f\n", size, total_time/size);
    }

    // MPI_Barrier(MPI_COMM_WORLD);

    // // read 10 times
    // char read_buf[64];
    // total_time = 0;
    // tot_time = 0;
    // for(int i = 0; i < 100; i++){        
        // MPI_File_seek(fh, rank * 64, MPI_SEEK_SET);
        // start = MPI_Wtime();
        // MPI_File_read(fh, read_buf, 64, MPI_CHAR, NULL);
        // end = MPI_Wtime();
        // time = end - start;
        // tot_time += time;
    // }
    // MPI_Reduce(&tot_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // if(rank == 0){
    //     printf("the average time of read is: %f\n", total_time/4);
    // }
    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}