#!/bin/bash

scriptdir=$(readlink -f $(dirname "$0"))
projectdir="$scriptdir/../"
llvmcompiledir="$projectdir/llvm/"
llvmbuild="$llvmcompiledir/build-llvm/"
llvminstall="$llvmcompiledir/install-llvm/"
gpumpibuild="$projectdir/build/"

numthreads=$(grep -c ^processor /proc/cpuinfo)

export CMAKE_BUILD_PARALLEL_LEVEL=$numthreads



cd "$gpumpibuild"
cmake "$projectdir" -DLLVM_DIR="$llvminstall/lib/cmake/llvm" -DClang_DIR="$llvminstall/lib/cmake/clang" 


cmake --build "$gpumpibuild"
