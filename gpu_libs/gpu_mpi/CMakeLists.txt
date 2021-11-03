add_library(gpu_mpi
    mpi.cu
    communicator.cu
    group.cu
    datatypes.cu
    operators.cu
    io.cu
    )
target_link_libraries(gpu_mpi PUBLIC gpu_libc common)
#target_compile_options(gpu_mpi PRIVATE -g -G)
