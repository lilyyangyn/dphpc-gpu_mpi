#include "datatypes.cuh"

#include "mpi_types_list.cuh"

namespace gpu_mpi {

#define MPI_TYPES_SIZE_F(NAME, TYPE) case NAME: return sizeof(TYPE);
#define MPI_TYPES_SIZE_SEP
    
__device__ int plainTypeSize(MPI_Datatype type) {
    switch (type) {
        MPI_TYPES_LIST(MPI_TYPES_SIZE_F, MPI_TYPES_SIZE_SEP)
        default: return -1;
    }
}

__device__ int TypeSize(MPI_Datatype_Ext type) { return type.size(); }
// std::list<MPI_Datatype_Ext*> Typelist;

#undef MPI_TYPES_SIZE_F
#undef MPI_TYPES_SIZE_SEP

} // namespace
