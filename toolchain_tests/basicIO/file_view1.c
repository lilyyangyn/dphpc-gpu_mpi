// source: https://cvw.cac.cornell.edu/parallelio/fileviewex 
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#define N 100
int main(int argc, char *argv[])
{
    MPI_Datatype arraytype;
    MPI_Datatype etype;
    MPI_Offset disp;
    MPI_File fh;
    int rank;
    int buf[N];
    for (int i=0;i<N;i++){
        buf[i]=i;
    }

    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    disp = rank*sizeof(int)*N; 
    etype = MPI_INT;
    MPI_Type_contiguous(N, MPI_INT, &arraytype);
    MPI_Type_commit(&arraytype);

    MPI_File_open(MPI_COMM_WORLD, "./test_file_view1.out", 
                    MPI_MODE_CREATE | MPI_MODE_RDWR, 
                    MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, disp, etype, arraytype, 
                    "native", MPI_INFO_NULL);
    MPI_File_write(fh, buf, N, etype, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}