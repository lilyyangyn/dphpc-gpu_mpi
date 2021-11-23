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

TEST_CASE("FileReadAmodeIncorrect", "[FileReadAmodeIncorrect]") {
    TestRunner testRunner(1);
    testRunner.run<FileReadAmodeIncorrect>();
}

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

struct FileView {
    static __device__ void run(bool& ok) {
        MPI_Datatype arraytype;
        MPI_Datatype etype;
        MPI_Offset disp;
        MPI_File fh;
        MPI_Info info;
        int rank;
        int N = 10;
        int* buf = (int *)malloc(N*sizeof(int));;
        for (int i=0;i<N;i++){
            buf[i]=i;
        }

        MPI_Init(0, 0);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        disp = rank*sizeof(int)*N; 
        etype = MPI_INT;
        arraytype = MPI_INT;
        const char* datarep = "native";
        // MPI_Type_contiguous(N, MPI_INT, &arraytype);
        // MPI_Type_commit(&arraytype);        

        MPI_File_open(MPI_COMM_WORLD, "test_file_view.out", 
                        MPI_MODE_CREATE | MPI_MODE_RDWR, 
                        info, &fh);

        MPI_Datatype arraytype_before;
        MPI_Datatype etype_before;
        MPI_Offset disp_before;
        char* datarep_before = (char *)malloc(N*sizeof(char));
        MPI_File_get_view(fh, &disp_before, &etype_before, &arraytype_before, datarep_before);
        assert(disp_before == 0);
        assert(etype_before == MPI_CHAR);
        assert(arraytype_before == MPI_CHAR);         
        for(int i = 0; i < sizeof(datarep_before); i++){
            assert(datarep_before[i]==datarep[i]);
        }

        MPI_File_set_view(fh, disp, etype, arraytype, datarep, info);

        MPI_Datatype arraytype_get;
        MPI_Datatype etype_get;
        MPI_Offset disp_get;
        char* datarep_get = (char *)malloc(N*sizeof(char));
        MPI_File_get_view(fh, &disp_get, &etype_get, &arraytype_get, datarep_get);

        MPI_File_close(&fh);

        assert(disp_get == disp);
        assert(etype_get == etype);
        assert(arraytype_get == arraytype);         
        for(int i = 0; i < sizeof(datarep_get); i++){
            assert(datarep_get[i]==datarep[i]);
        }        

        ok = true;

        free(datarep_before);
        free(datarep_get);
        free(buf);

        MPI_Finalize();
    }
};

TEST_CASE("FileView", "[FileView]") {
    TestRunner testRunner(1);
    testRunner.run<FileView>();
}
