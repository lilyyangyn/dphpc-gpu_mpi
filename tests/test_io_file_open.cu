#include "test_runner.cuh"

#include "mpi.h.cuh"

using namespace gpu_mpi;

struct FileOpenAmodeCorrect {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        int err = MPI_File_open(MPI_COMM_WORLD, nullptr, MPI_MODE_RDONLY, info, &fh);
        // int err = MPI_File_open(MPI_COMM_WORLD, "test.txt", MPI_MODE_RDWR || MPI_MODE_CREATE, info, &fh);
        ok = err == 0;
        // ok = fh.file != 0;

        MPI_Finalize();
    }
};

TEST_CASE("FileOpenAmodeCorrect", "[FileOpenAmodeCorrect]") {
    TestRunner testRunner(1);
    testRunner.run<FileOpenAmodeCorrect>();
}

struct FileOpenAmodeIncorrect {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        int err;
        err = MPI_File_open(MPI_COMM_WORLD, nullptr, MPI_MODE_RDONLY | MPI_MODE_RDWR, info, &fh);
        ok = err == MPI_ERR_AMODE;
        if(!ok){
            MPI_Finalize();
            return;
        }

        err = MPI_File_open(MPI_COMM_WORLD, nullptr, MPI_MODE_RDONLY | MPI_MODE_WRONLY, info, &fh);
        ok = err == MPI_ERR_AMODE;
        if(!ok){
            MPI_Finalize();
            return;
        }

        err = MPI_File_open(MPI_COMM_WORLD, nullptr, MPI_MODE_RDWR | MPI_MODE_WRONLY, info, &fh);
        ok = err == MPI_ERR_AMODE;
        if(!ok){
            MPI_Finalize();
            return;
        }

        err = MPI_File_open(MPI_COMM_WORLD, nullptr, MPI_MODE_RDONLY | MPI_MODE_RDWR | MPI_MODE_WRONLY, info, &fh);
        ok = err == MPI_ERR_AMODE;
        if(!ok){
            MPI_Finalize();
            return;
        }

        err = MPI_File_open(MPI_COMM_WORLD, nullptr, MPI_MODE_CREATE | MPI_MODE_RDONLY, info, &fh);
        ok = err == MPI_ERR_AMODE;
        if(!ok){
            MPI_Finalize();
            return;
        }

        err = MPI_File_open(MPI_COMM_WORLD, nullptr, MPI_MODE_EXCL | MPI_MODE_RDONLY, info, &fh);
        ok = err == MPI_ERR_AMODE;
        if(!ok){
            MPI_Finalize();
            return;
        }

        err = MPI_File_open(MPI_COMM_WORLD, nullptr, MPI_MODE_SEQUENTIAL | MPI_MODE_RDWR, info, &fh);
        ok = err == MPI_ERR_AMODE;
        if(!ok){
            MPI_Finalize();
            return;
        }

        MPI_Finalize();
    }
};

TEST_CASE("FileOpenAmodeIncorrect", "[FileOpenAmodeIncorrect]") {
    TestRunner testRunner(1);
    testRunner.run<FileOpenAmodeIncorrect>();
}
