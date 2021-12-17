#include "test_runner.cuh"

#include "mpi.h.cuh"
#include <string.h.cuh>
#include <stdio.h>

using namespace gpu_mpi;

struct FileViewSetter {
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

TEST_CASE("FileViewSetter", "[FileViewSetter]") {
    TestRunner testRunner(1);
    testRunner.run<FileViewSetter>();
}

struct FileViewRWSimple {
    static __device__ void run(bool& ok) {
        MPI_Datatype arraytype;
        MPI_Datatype etype;
        MPI_Offset disp;
        MPI_File fh;
        MPI_Info info;
        int rank;
        int N = 10;
        MPI_Init(0, 0);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        etype = MPI_INT;
        arraytype = etype;
        disp = rank*etype.size()*N; 
        const char* datarep = "native";

        int* wbuf = (int *)malloc(N*etype.size());
        int* rbuf = (int *)malloc(N*etype.size());
        for (int i=0;i<N;i++){
            wbuf[i]=rank;
        }
        
        MPI_File_open(MPI_COMM_WORLD, "viewwrite.txt", MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);
        MPI_File_set_view(fh, disp, etype, arraytype, datarep, info);

        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        MPI_File_write(fh, wbuf, N, etype, nullptr);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        MPI_File_read(fh, rbuf, N, etype, nullptr);

        MPI_File_close(&fh);

        for (int i=0;i<N;i++){
            // printf("rank: %d, w: %d, r: %d\n", rank, wbuf[i], rbuf[i]);
            ok = wbuf[i]==rbuf[i];
            if (ok == false){
                break;
            }
        }

        free(wbuf);
        free(rbuf);

        MPI_Finalize();
    }
};

TEST_CASE("FileViewRWSimple", "[FileViewRWSimple]") {
    TestRunner testRunner(2);
    testRunner.run<FileViewRWSimple>();
}

struct FileViewRWContiguous {
    static __device__ void run(bool& ok) {
        MPI_Datatype arraytype;
        MPI_Datatype etype;
        MPI_Datatype my_etype;
        MPI_Offset disp;
        MPI_File fh;
        MPI_Info info;
        int rank;
        int N = 10;
        MPI_Init(0, 0);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        etype = MPI_CHAR;
        my_etype = MPI_CHAR;
        // etype = MPI_INT;
        MPI_Type_contiguous(N, etype, &arraytype);
        MPI_Type_commit(&arraytype);
        disp = rank*etype.size()*N; 
        const char* datarep = "native";

        char* wbuf = (char *)malloc(N*etype.size());
        char* rbuf = (char *)malloc(N*etype.size());
        for (int i=0;i<N;i++){
            wbuf[i]=('0' + rank);
            rbuf[i]='?';
        }
        
        MPI_File_open(MPI_COMM_WORLD, "viewcontigious.txt", MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);
        MPI_File_set_view(fh, disp, etype, arraytype, datarep, info);

        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        MPI_File_write(fh, wbuf, N, my_etype, nullptr);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        MPI_File_read(fh, rbuf, N, my_etype, nullptr);

        MPI_File_close(&fh);

        // ok = *wbuf==*rbuf;
        for (int i=0;i<N;i++){
            // printf("rank: %d, w: %c, r: %c\n", rank, wbuf[i], rbuf[i]);
            ok = wbuf[i]==rbuf[i];
            if (ok == false){
                break;
            }
        }

        free(wbuf);
        free(rbuf);

        MPI_Finalize();
    }
};

TEST_CASE("FileViewRWContiguous", "[FileViewRWContiguous]") {
    TestRunner testRunner(2);
    testRunner.run<FileViewRWContiguous>();
}

struct FileViewRWVector {
    static __device__ void run(bool& ok) {
        MPI_Datatype arraytype;
        MPI_Datatype etype;
        MPI_Datatype my_etype;
        MPI_Offset disp;
        MPI_File fh;
        MPI_Info info;
        int rank;
        int NW = 5;
        MPI_Init(0, 0);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        etype = MPI_CHAR;
        my_etype = MPI_CHAR;
        // etype = MPI_INT;

        int npes; 
        MPI_Comm_size(MPI_COMM_WORLD, &npes);
        MPI_Type_vector(2, NW, NW*npes, etype, &arraytype);
        MPI_Type_commit(&arraytype);
        // for (int i = 0; i < arraytype.typemap_len; i++){
        //     printf("-- TYPEMAP -- idx: %d, disp: %d\n", i, arraytype.typemap[i].disp);
        // }

        disp = rank*etype.size()*NW; 
        const char* datarep = "native";

        char* wbuf = (char *)malloc(3*NW*etype.size());
        char* rbuf = (char *)malloc(3*NW*etype.size());
        for (int i=0;i<3*NW;i++){
            wbuf[i]=('0' + rank);
            rbuf[i]='?';
        }
        // printf("filetype size: %d\n", arraytype.size());
        
        MPI_File_open(MPI_COMM_WORLD, "viewvector.txt", MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);
        MPI_File_set_view(fh, disp, etype, arraytype, datarep, info);
        // printf("-- SET VIEW -- disp: %d, file_disp: %d, rank, %d\n", disp, fh.views[rank].disp, rank);

        // printf("SEEK POS BEFORE: %d, rank, %d\n", fh.seek_pos[rank], rank);
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        // printf("SEEK POS AFTER: %d, rank, %d\n", fh.seek_pos[rank], rank);
        MPI_File_write(fh, wbuf, NW*3, my_etype, nullptr);

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        MPI_File_read(fh, rbuf, NW*3, my_etype, nullptr);

        MPI_File_close(&fh);

        // ok = *wbuf==*rbuf;
        for (int i=0;i<3*NW;i++){
            // printf("rank: %d, i: %d, w: %c, r: %c\n", rank, i, wbuf[i], rbuf[i]);
            ok = wbuf[i]==rbuf[i];
            if (ok == false){
                break;
            }
        }

        free(wbuf);
        free(rbuf);

        MPI_Finalize();
    }
};

TEST_CASE("FileViewRWVector", "[FileViewRWVector]") {
    TestRunner testRunner(4);
    testRunner.run<FileViewRWVector>();
}

struct FileViewRWAnyPlace {
    static __device__ void run(bool& ok) {
        MPI_Datatype arraytype;
        MPI_Datatype etype;
        MPI_Datatype my_etype;
        MPI_Offset disp;
        MPI_File fh;
        MPI_Info info;
        int rank;
        int NW = 5;
        MPI_Init(0, 0);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        etype = MPI_CHAR;
        my_etype = MPI_CHAR;

        int npes; 
        MPI_Comm_size(MPI_COMM_WORLD, &npes);
        MPI_Type_vector(2, NW, NW*npes, etype, &arraytype);
        MPI_Type_commit(&arraytype);

        disp = rank*etype.size()*NW; 
        const char* datarep = "native";

        char* wbuf = (char *)malloc(3*NW*etype.size());
        char* rbuf = (char *)malloc(3*NW*etype.size());
        for (int i=0;i<3*NW;i++){
            wbuf[i]=('0' + rank);
            rbuf[i]='?';
        }
        // printf("filetype size: %d\n", arraytype.size());

        int first_round = NW*2 - 2;
        int second_round = NW + 2;
        
        MPI_File_open(MPI_COMM_WORLD, "viewanyplace.txt", MPI_MODE_RDWR | MPI_MODE_CREATE, info, &fh);
        MPI_File_set_view(fh, disp, etype, arraytype, datarep, info);
        // printf("-- SET VIEW -- disp: %d, file_disp: %d, rank, %d\n", disp, fh.views[rank].disp, rank);

        // printf("SEEK POS BEFORE: %d, rank, %d\n", fh.seek_pos[rank], rank);
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        // printf("SEEK POS AFTER: %d, rank, %d\n", fh.seek_pos[rank], rank);
        MPI_File_write(fh, wbuf, first_round, my_etype, nullptr);
        MPI_File_write(fh, &wbuf[first_round], second_round, my_etype, nullptr);

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        MPI_File_read(fh, rbuf, first_round, my_etype, nullptr);
        MPI_File_read(fh, &rbuf[first_round], second_round, my_etype, nullptr);

        MPI_File_close(&fh);

        // ok = *wbuf==*rbuf;
        for (int i=0;i<3*NW;i++){
            // printf("rank: %d, i: %d, w: %c, r: %c\n", rank, i, wbuf[i], rbuf[i]);
            ok = wbuf[i]==rbuf[i];
            if (ok == false){
                break;
            }
        }

        free(wbuf);
        free(rbuf);

        MPI_Finalize();
    }
};

TEST_CASE("FileViewRWAnyPlace", "[FileViewRWAnyPlace]") {
    TestRunner testRunner(2);
    testRunner.run<FileViewRWAnyPlace>();
}