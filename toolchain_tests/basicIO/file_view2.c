// source: https://cvw.cac.cornell.edu/parallelio/fileviewex 
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#define NW 50
int main(int argc, char *argv[])
{
    MPI_Datatype fileblk;// file block, not determined so not runnable yet
    MPI_Datatype ablk;//a block
    MPI_Offset disp;
    MPI_File fh;
    int buf[NW*2];// may need initialize?
    int rank;
    int npes; // size
    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);


    MPI_File_open(MPI_COMM_WORLD, "./test_file_view2.out", 
                    MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    /* want to see 2 blocks of NW ints, NW*npes apart */
    MPI_Type_vector(2, NW, NW*npes, MPI_INT, &fileblk);
    MPI_Type_commit(                         &fileblk);
    disp = (MPI_Offset)rank*NW*sizeof(int);
    MPI_File_set_view(fh, disp, MPI_INT, fileblk, 
                        "native", MPI_INFO_NULL);

    /* processor writes 2 'ablk', each with NW ints */
    MPI_Type_contiguous(NW,   MPI_INT, &ablk);
    MPI_Type_commit(&ablk);
    MPI_File_write(fh, (void *)buf, 2, ablk, MPI_STATUS_IGNORE);//&status); <--originally 
    MPI_File_close(&fh);
    MPI_Finalize();
    return 0;
}

