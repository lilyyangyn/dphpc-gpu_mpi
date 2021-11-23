#include "test_runner.cuh"

#include "mpi.h.cuh"

using namespace gpu_mpi;

struct FileOpenAmodeCorrect {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        // int err MPI_File_open(MPI_COMM_WORLD, nullptr, MPI_MODE_RDONLY, info, &fh);
        int err = MPI_File_open(MPI_COMM_WORLD, "test.txt", MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);
        // ok = err == MPI_ERR_NO_SUCH_FILE;
        // ok = err == 0;
        ok = fh.file != NULL;     
        MPI_File_close(&fh);   

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

struct FileSharedRW {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        int err = MPI_File_open(MPI_COMM_WORLD, "test.txt", MPI_MODE_WRONLY, info, &fh);
        
        char write_buf[128]="0123456701234567012345670123456701234567012345670123456701234567";
        ok = MPI_File_write_shared(fh, write_buf, 64, MPI_CHAR, NULL) == 0;
        // char read_buf[65];
        // ok = MPI_File_read_shared(fh, read_buf, 64, MPI_CHAR, NULL) == 0;
        MPI_File_close(&fh);
        MPI_Finalize();
    }
};

TEST_CASE("FileSharedRW", "[FileSharedRW]") {
    TestRunner testRunner(4);
    testRunner.run<FileSharedRW>();
}
