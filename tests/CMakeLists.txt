cmake_minimum_required(VERSION 3.12)

project(gpu_mpi_tests LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # prevents mess with host/device linking of the functions with the same name

# enable system-wide atomics: atomicCAS_system (compute capability >= 6.0)
# It is important to incude multiple compute capabilities, because
# sometimes managed memory doesn't on newer devices when compiled with old
# compute capability.
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_70,code=sm_70")


add_executable(cuda_mpi_test cuda_mpi_test.cu)
target_link_libraries(cuda_mpi_test gpu_mpi gpu_libc libc_processor)
target_compile_definitions(cuda_mpi_test PRIVATE ENABLE_GPU_MPI_LOG)
#target_compile_options(cuda_mpi_test PRIVATE -g -G)
add_test(cuda_mpi_test cuda_mpi_test)

add_executable(sample_test sample_test.cu)
target_link_libraries(sample_test gpu_libc gpu_mpi libc_processor)
add_test(sample_test sample_test)

add_executable(communicator_test communicator_test.cu)
target_link_libraries(communicator_test PUBLIC gpu_libc gpu_mpi common libc_processor)
#target_compile_options(communicator_test PRIVATE -g -G)
add_test(communicator_test communicator_test)


add_executable(collectives_test collectives_test.cu)
target_link_libraries(collectives_test gpu_libc gpu_mpi libc_processor)
add_test(collectives_test collectives_test)

add_executable(test_io_file_access test_io_file_access.cu)
target_link_libraries(test_io_file_access gpu_libc gpu_mpi libc_processor)
add_test(test_io_file_access test_io_file_access)

add_executable(test_io_file_open test_io_file_open.cu)
target_link_libraries(test_io_file_open gpu_libc gpu_mpi libc_processor)
add_test(test_io_file_open test_io_file_open)

add_executable(test_io_file_view test_io_file_view.cu)
target_link_libraries(test_io_file_view gpu_libc gpu_mpi libc_processor)
add_test(test_io_file_view test_io_file_view)