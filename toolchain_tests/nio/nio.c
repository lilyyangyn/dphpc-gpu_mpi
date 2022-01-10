#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Uses asynchronous I/O. Each process writes to separate files and
reads them back. The file name is taken as a command-line argument,
and the process rank is appended to it.*/ 

int main()
{
    MPI_File fh;
    MPI_Info info;
    MPI_Request request;
    MPI_Init(0, 0); 
    char buf[100]="hello world from GPUMPI IO";
    char rdbuf[100]={0};
    // MPI_File_open(MPI_COMM_WORLD, "aiotest.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &fh);
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if(rank==0)
    //     MPI_File_write(fh, buf, strlen(buf), MPI_CHAR, MPI_STATUS_IGNORE);
    // MPI_File_close(&fh);

    printf("start again--------\n");
    MPI_File_open(MPI_COMM_WORLD, "aiotest.txt", MPI_MODE_RDONLY, info, &fh);
    MPI_File_iread(fh, rdbuf, 16, MPI_CHAR, &request);

    MPI_Wait(&request, MPI_STATUS_IGNORE);
    
    MPI_File_close(&fh);
    printf("heres the result %s", rdbuf);

    fflush(stdout);
    return 0;
}