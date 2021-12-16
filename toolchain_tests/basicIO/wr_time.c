#include <stdio.h>
#include <string.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    MPI_File fh;
    MPI_Info info;
    int rank;
    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_File_open(MPI_COMM_WORLD, "./testfile.txt", MPI_MODE_WRONLY, info, &fh);
    double start, end, time, tot_time = 0;

    char write_buf[128]="0123456701234567012345670123456701234567012345670123456701234567";
    // write 10 times
    for(int i = 0; i < 100; i++){
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        start = MPI_Wtime();
        MPI_File_write(fh, write_buf, 64, MPI_CHAR, NULL);
        end = MPI_Wtime();
        time = end - start;
        tot_time += time;
    }
    printf("the average time of write is: %f\n", tot_time/10);

    // read 10 times

    // char read_buf[64];
    // tot_time = 0;
    // for(int i = 0; i < 100; i++){
    //     MPI_File_seek(fh, 0, MPI_SEEK_SET);
    //     start = MPI_Wtime();
    //     MPI_File_read(fh, read_buf, 64, MPI_CHAR, NULL);
    //     end = MPI_Wtime();
    //     time = end - start;
    //     tot_time += time;
    // }
    // printf("the average time of read is: %f\n", tot_time/10);
    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}