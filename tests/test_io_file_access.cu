#include "test_runner.cuh"

#include "mpi.h.cuh"

using namespace gpu_mpi;

struct FileRead {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        MPI_Info info;
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, "1.txt", MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);
        if (rank == 0) MPI_File_write(fh, "a", 1, MPI_CHAR, nullptr);
        MPI_File_close(&fh);

        MPI_File_open(MPI_COMM_WORLD, "1.txt", MPI_MODE_RDONLY, info, &fh);
        char buf;
        MPI_Status status;
        int read_size = MPI_File_read(fh, &buf, 1, MPI_CHAR, &status);
        MPI_File_close(&fh);
        
        ok = read_size != 0;

        MPI_Finalize();
    }
};

// TEST_CASE("FileRead", "[FileRead]") {
//     TestRunner testRunner(1);
//     testRunner.run<FileRead>();
// }

struct FileReadAmodeIncorrect {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        char buf;
        int err;

        MPI_File_open(MPI_COMM_WORLD, "2.txt", MPI_MODE_WRONLY | MPI_MODE_CREATE, info, &fh);
        err = MPI_File_read(fh, &buf, 1, MPI_CHAR, nullptr);
        ok = err == MPI_ERR_AMODE;
        if(!ok) { MPI_Finalize(); return; }
        MPI_File_close(&fh);

        MPI_File_open(MPI_COMM_WORLD, "2.txt", MPI_MODE_RDONLY | MPI_MODE_SEQUENTIAL, info, &fh);
        err = MPI_File_read(fh, &buf, 1, MPI_CHAR, nullptr);
        ok = err == MPI_ERR_UNSUPPORTED_OPERATION;
        if(!ok) { MPI_Finalize(); return; }
        MPI_File_close(&fh);

        MPI_Finalize();
    }
};

// TEST_CASE("FileReadAmodeIncorrect", "[FileReadAmodeIncorrect]") {
//     TestRunner testRunner(1);
//     testRunner.run<FileReadAmodeIncorrect>();
// }

struct FileSize {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        MPI_Info info;
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, "1.txt", MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);
        if (rank == 0) MPI_File_write(fh, "a", 1, MPI_CHAR, nullptr);

        MPI_Offset size;
        MPI_File_get_size(fh, &size);

        MPI_File_close(&fh);

        ok = size == 1;

        MPI_Finalize();
    }
};

TEST_CASE("FileSize", "[FileSize]") {
    TestRunner testRunner(2);
    testRunner.run<FileSize>();
}
