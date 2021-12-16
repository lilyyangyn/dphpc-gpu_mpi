#include "test_runner.cuh"

#include "mpi.h.cuh"

using namespace gpu_mpi;

#define USE_BUFFER true

struct FileRead {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, "1.txt", MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);
        
        int w_buf = 6;
        #if USE_BUFFER
            // MPI_File_write(fh, "a", 1, MPI_CHAR, nullptr);
            MPI_File_write(fh, &w_buf, 1, MPI_INT, nullptr);
        #else
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            // if (rank == 0) MPI_File_write(fh, "a", 1, MPI_CHAR, nullptr);
            if (rank == 0) MPI_File_write(fh, &w_buf, 1, MPI_INT, nullptr);
        #endif
        MPI_File_close(&fh);

        MPI_File_open(MPI_COMM_WORLD, "1.txt", MPI_MODE_RDONLY, info, &fh);
        // char buf;
        int buf;
        MPI_Status status;
        // int read_size = MPI_File_read(fh, &buf, 1, MPI_CHAR, &status);
        int read_size = MPI_File_read(fh, &buf, 1, MPI_INT, &status);
        MPI_File_close(&fh);
        
        // ok = read_size != 0;
        ok = buf == 6;

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

        MPI_Info info;
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, "1.txt", MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);
        #if USE_BUFFER
            MPI_File_write(fh, "aaa", 3, MPI_CHAR, nullptr);
        #else
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) MPI_File_write(fh, "aaa", 3, MPI_CHAR, nullptr);
        #endif

        MPI_Offset size;
        MPI_File_get_size(fh, &size);

        MPI_File_close(&fh);

        ok = size == 3;

        MPI_Finalize();
    }
};

// TEST_CASE("FileSize", "[FileSize]") {
//     TestRunner testRunner(2);
//     testRunner.run<FileSize>();
// }

struct FileReadAnyPlaceAnyLen {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, "testfile.txt", MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);
        char write_buf[64]; 
        for(int i = 0; i < 63; i++){
            write_buf[i] = 'a';
        }
        write_buf[63] = '\0';
        #if USE_BUFFER
            MPI_File_write(fh, write_buf, 64, MPI_CHAR, nullptr);
        #else
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) MPI_File_write(fh, write_buf, 64, MPI_CHAR, nullptr);
        #endif

        char read_buf[81];
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        int count = MPI_File_read(fh, read_buf, 64, MPI_CHAR, NULL);
        read_buf[80] = '\0';
        printf("we read %d, content is %s\n", count, read_buf);
        ok = count == 64;

        MPI_Finalize();
    }
};

// TEST_CASE("FileReadAnyPlaceAnyLen", "[FileReadAnyPlaceAnyLen]") {
//     TestRunner testRunner(1);
//     testRunner.run<FileReadAnyPlaceAnyLen>();
// }


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

// TEST_CASE("FileSharedRW", "[FileSharedRW]") {
//     TestRunner testRunner(4);
//     testRunner.run<FileSharedRW>();
// }
