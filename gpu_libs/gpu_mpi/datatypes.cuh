#ifndef DATATYPES_CUH
#define DATATYPES_CUH

#include <stdbool.h>
#include <stdint.h>
#include <list>

// #include <vector>
// #include <algorithm>
// struct MPI_Typemap {
//     std::vector< std::pair<MPI_Datatype, int>> Typemap;  // <datatype, displacement>
//     size_t size() const {}
// };

#include "mpi_types_list.cuh"

#define MPI_TYPES_ENUM_F(name, type) name
#define MPI_TYPES_ENUM_SEP ,

enum MPI_Datatype {
    MPI_TYPES_LIST(MPI_TYPES_ENUM_F, MPI_TYPES_ENUM_SEP)
};

#undef MPI_TYPES_ENUM_F
#undef MPI_TYPES_ENUM_SEP

struct MPI_Datatype_Ext;
// add a declaration before definition, because we have a circular reference Datatype_Ext -> Datatype -> Datatype_Ext

namespace gpu_mpi {


namespace detail {
    
template <MPI_Datatype> struct GetPlainType {};

#define MPI_TYPES_CONV_F(NAME, TYPE) template <> struct GetPlainType<NAME> { using type = TYPE; };
#define MPI_TYPES_CONV_SEP

MPI_TYPES_LIST(MPI_TYPES_CONV_F, MPI_TYPES_CONV_SEP)

#undef MPI_TYPES_CONV_F
#undef MPI_TYPES_CONV_SEP

} // namespace

template <MPI_Datatype T>
using PlainType = typename detail::GetPlainType<T>::type;

__device__ int plainTypeSize(MPI_Datatype type);
__device__ int TypeSize(MPI_Datatype_Ext type);

}// namespace

#undef MPI_TYPES_LIST

// const int TYPENAME_MAXLEN = 32;

struct MPI_Datatype_Ext {  // MPI datatype in extended format, for customized datatypes
    // For now, only two use cases are supported:
    // 1. get size of Datatype/Datatype_ext
    // 2. Compare Datatype_ext to Datatype so that we can simplify our code in special cases
    //    (comparing Datatype_ext to Datatype_ext is not needed atm)

    // int id;
    size_t _size;
    // char name[TYPENAME_MAXLEN];
    bool committed;
    int __basic_type;  // for porting to MPI_Datatype

    __device__ MPI_Datatype_Ext(): committed(false), _size(0), __basic_type(-1) {}
    __device__ MPI_Datatype_Ext(MPI_Datatype type): committed(false), _size(gpu_mpi::plainTypeSize(type)), __basic_type(type) {}
    // operator MPI_Datatype() const {return __basic_type;} - return value type does not match the function type

    __device__ size_t size() const {return _size;}
    __device__ friend bool operator == (const MPI_Datatype_Ext a, const MPI_Datatype b) {return a.__basic_type == b;}
};

#endif
