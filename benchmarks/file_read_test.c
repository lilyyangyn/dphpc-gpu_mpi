#include "../opt/openmpi/include/mpi.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char* argv[]){
    // create a test file and write something into it
    FILE* file = fopen("./input.txt","w");
    const int MAX_SIZE = 512 * 1024 * 1024 ;
    if(file == NULL){
        printf("test file creation error");
        return -1;
    }
    char* buffer = (char*)malloc(MAX_SIZE*sizeof(char));
    memset(buffer, 0, MAX_SIZE*sizeof(char));
    for(int i = 0; i < MAX_SIZE; i++){
        buffer[i] = 'a' + i%8;
    }
    //int res = fputs(buffer, file);
    int res = fwrite(buffer, sizeof(buffer[0]), MAX_SIZE, file);
    if(res == EOF){
        printf("%s", "fail to write");
    }
    fclose(file);
    free(buffer);
    buffer = NULL;


    // mpi test part
    int numprocs, myrank;
    MPI_Offset fsize, offset;
    int nbytes, nchars;
    MPI_File fh;
    double start, end, time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    // TODO: design a pattern that processes read the
    // different places of the same file for many times
    MPI_File_open(MPI_COMM_WORLD, "input.txt", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_get_size(fh, &fsize);

    nbytes = fsize / numprocs;
    nchars = nbytes / sizeof(char);
    offset = myrank * nbytes;

    char* mpi_buf = (char*)malloc(nbytes + 1);
    memset(mpi_buf, 0, nbytes);
    
    // MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    // each process read the part it is responsible for
    MPI_File_read_at(fh, offset, mpi_buf, nbytes, MPI_CHAR, MPI_STATUS_IGNORE);
    // Using MPI_Barrier means not calculating the end time until the last process finishes reading
    // MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    MPI_File_close(&fh);

    // TODO: check the correctness of MPI_FREAD
    FILE* f = fopen("./input.txt","r");
    char* corr_buf = (char*)malloc(nbytes + 1);
    memset(corr_buf, 0, nbytes);
    fseek(f, offset, SEEK_SET);
    fread(corr_buf, sizeof(char), nchars, f);
    mpi_buf[nbytes] = '\0';
    corr_buf[nbytes] = '\0';

    // printf("the content read by MPI process is: %s\n", mpi_buf);
    // printf("the length of content by MPI: %ld\n", strlen(mpi_buf));
    // printf("the content read by c fgets is: %s\n", corr_buf);
    // printf("the length of content by fgets: %ld\n", strlen(corr_buf));
    assert(strcmp(mpi_buf, corr_buf) == 0);

    free(mpi_buf);
    free(corr_buf);

    // calculate the time
    time = end - start;

    // ask process 0 to sum all the time
    double total_time;
    MPI_Reduce(&time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(myrank == 0){
        printf("The average time is: %f%s", total_time/numprocs, "\n");
    }

    MPI_Finalize();

}