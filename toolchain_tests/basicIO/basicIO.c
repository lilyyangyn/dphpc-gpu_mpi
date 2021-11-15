#include <stdio.h>
#include <string.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    MPI_File fh;
    MPI_Info info;
    char buf[100]="hello world from GPUMPI IO";
    int rank;
    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_File_open(MPI_COMM_WORLD, "test.out",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &fh);
    if (rank == 0){
        MPI_File_write(fh, buf, strlen(buf), MPI_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}