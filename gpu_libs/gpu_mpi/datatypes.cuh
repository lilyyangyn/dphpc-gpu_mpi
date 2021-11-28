#ifndef DATATYPES_CUH
#define DATATYPES_CUH

#include <stdbool.h>
#include <stdint.h>
// #include <list>

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
const int TYPEMAP_MAXLEN = 101;

struct MPI_Datatype_Ext {  // MPI datatype in extended format, for customized datatypes
    // For now, only two use cases are supported:
    // 1. get size of Datatype/Datatype_ext
    // 2. Compare Datatype_ext to Datatype so that we can simplify our code in special cases
    //    (comparing Datatype_ext to Datatype_ext is not needed atm)

    // int id;
    size_t _size;  // extent of this datatype (including padding spaces)
    // char name[TYPENAME_MAXLEN];
    bool committed;
    struct pii {int basic_type;  int disp;} typemap[TYPEMAP_MAXLEN];
    int typemap_len;
    // int mul;  MPI_Datatype_Ext* subtype;  // provisioned, maybe for tree-shape optimized storage

    __device__ MPI_Datatype_Ext(): committed(false), _size(0), typemap_len(0) {}
    __device__ MPI_Datatype_Ext(MPI_Datatype in_type): committed(true), _size(gpu_mpi::plainTypeSize(in_type)), typemap_len(1)
    {
        typemap[0].basic_type = in_type;
        typemap[0].disp = 0;
    }  // for porting to MPI_Datatype
    // operator MPI_Datatype() const {return typemap[0].basic_type;} - return value type does not match the function type

    __device__ size_t size() const {return _size;}
    __device__ friend bool operator == (const MPI_Datatype_Ext a, const MPI_Datatype b)
    {
        if (a.typemap_len != 1) return false;
        if (a.typemap[0].disp != 0) return false;
        return a.typemap[0].basic_type == b;
    }
    __device__ friend void copy_typemap_once(MPI_Datatype_Ext* newtype, const MPI_Datatype_Ext oldtype, int mul)
    {
        int o_typemap_len = oldtype.typemap_len;
        for (int i = 0; i < o_typemap_len; i++)
        {
            (newtype -> typemap)[mul * o_typemap_len + i].basic_type = (oldtype.typemap)[i].basic_type;
            (newtype -> typemap)[mul * o_typemap_len + i].disp       = (oldtype.typemap)[i].disp + mul * oldtype.size();
        }
    }
    // __device__ friend bool operator == (const MPI_Datatype_Ext a, const MPI_Datatype_Ext b)
    // {
    //     if (a.mul != b.mul) return false;
    //     if ((a.subtype == NULL) ^ (b.subtype == NULL)) return false;
    //     return a.subtype == b.subtype || *(a.subtype) == *(b.subtype);
    // }
};

#endif
