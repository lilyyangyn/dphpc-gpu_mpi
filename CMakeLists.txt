cmake_minimum_required(VERSION 3.12)

project(gpumpi LANGUAGES C CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON) # all subdirs will be automatically added to include path of targets
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON) # all subdirs will be automatically added to include path of dependencies

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G") 

# Prevents mess with host/device linking of the functions with the same name.
# It seems that it also passes -rdc=true required by some features like "dynamic parallelism".
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) 

# specify cuda architectures for newer cmake
set(CMAKE_CUDA_ARCHITECTURES 60 61 70)

# specify cuda architectures for older cmake
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_70,code=sm_70")

add_subdirectory(common)
add_subdirectory(source_converter)
add_subdirectory(gpu_libs)
add_subdirectory(scripts)

include(CTest) # enables Dashboard actions support 
enable_testing()
add_subdirectory(tests)
# add_subdirectory(toolchain_tests)

export(TARGETS 
    gpu_libs
    gpu_libc
    gpu_mpi
    gpu_main
    common
    libc_processor
    APPEND FILE gpu_libs-exports.cmake)


