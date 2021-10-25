#include "test_runner.cuh"

#include "mpi.h.cuh"

using namespace gpu_mpi;

struct MPI_Info;
struct MPI_File;

struct FileOpenTest {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        MPI_Info info;
        MPI_File fh;
        int err = MPI_FILE_OPEN(MPI_COMM_WORLD, nullptr, 0, info, &fh);
        ok = err == 0;

        MPI_Finalize();
    }
};

TEST_CASE("empty test", "[empty_test]") {
    TestRunner testRunner(1);
    testRunner.run<FileOpenTest>();
}