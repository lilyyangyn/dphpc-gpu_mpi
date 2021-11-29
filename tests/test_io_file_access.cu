#include "test_runner.cuh"

#include "mpi.h.cuh"

using namespace gpu_mpi;

struct FileRead {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, "1.txt", MPI_MODE_RDONLY, info, &fh);

        char buf;
        MPI_Status status;
        int err = MPI_File_read(fh, &buf, 1, MPI_CHAR, &status);
        ok = err == 0;

        MPI_Finalize();
    }
};

TEST_CASE("FileRead", "[FileRead]") {
    TestRunner testRunner(1);
    testRunner.run<FileRead>();
}

struct FileReadAmodeIncorrect {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        char buf;
        int err;

        MPI_File_open(MPI_COMM_WORLD, "./1.txt", MPI_MODE_WRONLY | MPI_MODE_CREATE, info, &fh);
        err = MPI_File_read(fh, &buf, 1, MPI_CHAR, nullptr);
        ok = err == MPI_ERR_AMODE;
        if(!ok) { MPI_Finalize(); return; }
        MPI_File_close(&fh);

        MPI_File_open(MPI_COMM_WORLD, "./1.txt", MPI_MODE_RDONLY | MPI_MODE_SEQUENTIAL, info, &fh);
        err = MPI_File_read(fh, &buf, 1, MPI_CHAR, nullptr);
        ok = err == MPI_ERR_UNSUPPORTED_OPERATION;
        if(!ok) { MPI_Finalize(); return; }
        MPI_File_close(&fh);

        MPI_Finalize();
    }
};

TEST_CASE("FileReadAmodeIncorrect", "[FileReadAmodeIncorrect]") {
    TestRunner testRunner(1);
    testRunner.run<FileReadAmodeIncorrect>();
}


struct FileReadAnyPlaceAnyLen {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, "testfile.txt", MPI_MODE_RDONLY, info, &fh);

        char read_buf[81];
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        int count = MPI_File_read(fh, read_buf, 64, MPI_CHAR, NULL);
        read_buf[80] = '\0';
        printf("we read %d, content is %s\n", count, read_buf);
        ok = count == 64;

        MPI_Finalize();
    }
};

TEST_CASE("FileReadAnyPlaceAnyLen", "[FileReadAnyPlaceAnyLen]") {
    TestRunner testRunner(1);
    testRunner.run<FileReadAnyPlaceAnyLen>();
}
