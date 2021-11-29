#include "test_runner.cuh"

#include "mpi.h.cuh"
#include <string.h.cuh>
#include <stdio.h>

using namespace gpu_mpi;


struct FileWriteTest {
    static __device__ void run(bool& ok) {
        MPI_File fh;
        MPI_Info info;
        int rank;
        MPI_Init(0, 0);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int ret_val=0;
        char *a="ccchello world";

        MPI_File_open(MPI_COMM_WORLD, "test_open.txt",
                  MPI_MODE_CREATE | MPI_MODE_RDWR, info, &fh);
        if(rank==0)
            ret_val = MPI_File_write(fh,a,__gpu_strlen(a),MPI_CHAR,nullptr);
        MPI_File_close(&fh);
        
        ok = __gpu_strlen(a) == ret_val ;
        // TODO: auto clean
        // if( remove( "test.txt" ) != 0 )
        //     printf( "Error deleting file" );
        // else
        //     printf( "File successfully deleted" );
        MPI_Finalize();
    }
};

TEST_CASE("FileWriteTest", "[FileWriteTest]") {
    TestRunner testRunner(1);
    testRunner.run<FileWriteTest>();
}


struct FileWriteAnyPlaceAnyLen {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, "testfile.txt", MPI_MODE_RDWR, info, &fh);

        // char write_buf[128]="01234567012345670123456701234567012345670123456701234567012345670123456701234567012345670123456701234567";
        char write_buf[128] = "00000000111111112222222233333333444444445555555566666666777777788888888999999999";
        MPI_File_seek(fh, 30, MPI_SEEK_SET);
        int count = MPI_File_write(fh, write_buf, 76, MPI_CHAR, NULL);
        ok = count == 76;

        MPI_File_close(&fh);
        MPI_Finalize();
    }
};

TEST_CASE("FileWriteAnyPlaceAnyLen", "[FileWriteAnyPlaceAnyLen]") {
    TestRunner testRunner(1);
    testRunner.run<FileWriteAnyPlaceAnyLen>();
}