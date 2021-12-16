#include "datatypes.cuh"
#include "mpi_common.cuh"
#include "operators.cuh"

#include "cuda_mpi.cuh"
#include "mpi.h.cuh"
#include "mpi_operators_list.cuh"

#include "mpi_types_list.cuh"

#include "stdlib.cuh"
#include <stdio.h>

#include <type_traits>

#define MPI_OP_LIST_DEF_F(name, class) __device__ MPI_Op name = nullptr;
#define MPI_OP_LIST_DEF_SEP

MPI_OPERATORS_LIST(MPI_OP_LIST_DEF_F, MPI_OP_LIST_DEF_SEP)

#undef MPI_OP_LIST_DEF_F
#undef MPI_OP_LIST_DEF_SEP

namespace {
template <MPI_Datatype_Basic Type> struct IsIntegerType { enum { value = false }; };

template <> struct IsIntegerType<             MPI_SHORT> { enum { value = true }; };
template <> struct IsIntegerType<               MPI_INT> { enum { value = true }; };
template <> struct IsIntegerType<              MPI_LONG> { enum { value = true }; };
template <> struct IsIntegerType<     MPI_LONG_LONG_INT> { enum { value = true }; };
template <> struct IsIntegerType<         MPI_LONG_LONG> { enum { value = true }; };
template <> struct IsIntegerType<       MPI_SIGNED_CHAR> { enum { value = true }; };
template <> struct IsIntegerType<     MPI_UNSIGNED_CHAR> { enum { value = true }; };
template <> struct IsIntegerType<    MPI_UNSIGNED_SHORT> { enum { value = true }; };
template <> struct IsIntegerType<          MPI_UNSIGNED> { enum { value = true }; };
template <> struct IsIntegerType<     MPI_UNSIGNED_LONG> { enum { value = true }; };
template <> struct IsIntegerType<MPI_UNSIGNED_LONG_LONG> { enum { value = true }; };
template <> struct IsIntegerType<            MPI_INT8_T> { enum { value = true }; };
template <> struct IsIntegerType<           MPI_INT16_T> { enum { value = true }; };
template <> struct IsIntegerType<           MPI_INT32_T> { enum { value = true }; };
template <> struct IsIntegerType<           MPI_INT64_T> { enum { value = true }; };
template <> struct IsIntegerType<           MPI_UINT8_T> { enum { value = true }; };
template <> struct IsIntegerType<          MPI_UINT16_T> { enum { value = true }; };
template <> struct IsIntegerType<          MPI_UINT32_T> { enum { value = true }; };
template <> struct IsIntegerType<          MPI_UINT64_T> { enum { value = true }; };

template <MPI_Datatype_Basic Type> struct IsFloatingType { enum { value = false }; };

template <> struct IsFloatingType<            MPI_FLOAT> { enum { value = true }; };
template <> struct IsFloatingType<           MPI_DOUBLE> { enum { value = true }; };
template <> struct IsFloatingType<      MPI_LONG_DOUBLE> { enum { value = true }; };

template <MPI_Datatype_Basic Type> struct IsLogicalType { enum { value = false }; };

template <> struct IsLogicalType <           MPI_C_BOOL> { enum { value = true }; };

template <MPI_Datatype_Basic Type> struct IsComplexType { enum { value = false }; };

template <> struct IsComplexType<        MPI_C_COMPLEX> { enum { value = true }; };
template <> struct IsComplexType<  MPI_C_FLOAT_COMPLEX> { enum { value = true }; };
template <> struct IsComplexType< MPI_C_DOUBLE_COMPLEX> { enum { value = true }; };
template <> struct IsComplexType<MPI_C_LONG_DOUBLE_COMPLEX> { enum { value = true }; };

template <MPI_Datatype_Basic Type> struct IsByteType { enum { value = false }; };

template <> struct IsByteType   <             MPI_BYTE> { enum { value = true }; };

template <MPI_Datatype_Basic Type> struct IsTextType { enum { value = false }; };

template <> struct IsTextType   <             MPI_CHAR> { enum { value = true }; };
template <> struct IsTextType   <            MPI_WCHAR> { enum { value = true }; };

template <typename T>
struct OpMax {
    __device__ T operator()(T a, T b) { return a > b ? a : b; }
};

template <typename T>
struct OpMin {
    __device__ T operator()(T a, T b) { return a > b ? b : a; }
};

template <typename T>
struct OpSum {
    __device__ T operator()(T a, T b) { return a + b; }
};

template <typename T>
struct OpProd {
    __device__ T operator()(T a, T b) { return a * b; }
};

template <typename T>
struct OpLAnd {
    __device__ T operator()(T a, T b) { return a && b; }
};

template <typename T>
struct OpBAnd {
    __device__ T operator()(T a, T b) { return a & b; }
};

template <typename T>
struct OpLOr {
    __device__ T operator()(T a, T b) { return a || b; }
};

template <typename T>
struct OpBOr {
    __device__ T operator()(T a, T b) { return a | b; }
};

template <typename T>
struct OpLXor {
    __device__ T operator()(T a, T b) { return a != b; }
};

template <typename T>
struct OpBXor {
    __device__ T operator()(T a, T b) { return a ^ b; }
};

template <typename T>
struct OpMaxLoc {
    __device__ T operator()(T a, T b) { 
        T c;
        if (a.val > b.val) {
            c = a;
        } else if (a.val < b.val) {
            c = b;
        } else {
            c = a.idx < b.idx ? a : b;
        }
        return c; 
    }
};

template <typename T>
struct OpMinLoc {
    __device__ T operator()(T a, T b) { 
        T c;
        if (a.val > b.val) {
            c = b;
        } else if (a.val < b.val) {
            c = a;
        } else {
            c = a.idx < b.idx ? a : b;
        }
        return c; 
    }
};

template <MPI_Datatype_Basic T, template <class> class Op, class Enable = void>
struct IsTypeOpAllowed { constexpr static bool value = false; };

template <MPI_Datatype_Basic T>
struct IsTypeOpAllowed<T, OpMax, std::enable_if_t<IsIntegerType<T>::value || IsFloatingType<T>::value>> { constexpr static bool value = true; };

template <MPI_Datatype_Basic T>
struct IsTypeOpAllowed<T, OpMin, std::enable_if_t<IsIntegerType<T>::value || IsFloatingType<T>::value>> { constexpr static bool value = true; };

template <MPI_Datatype_Basic T>
struct IsTypeOpAllowed<T, OpSum, std::enable_if_t<IsIntegerType<T>::value || IsFloatingType<T>::value || IsComplexType<T>::value>> {
    constexpr static bool value = true;
};

template <MPI_Datatype_Basic T>
struct IsTypeOpAllowed<T, OpProd, std::enable_if_t<IsIntegerType<T>::value || IsFloatingType<T>::value || IsComplexType<T>::value>> {
    constexpr static bool value = true;
};

template <MPI_Datatype_Basic T>
struct IsTypeOpAllowed<T, OpLAnd, std::enable_if_t<IsIntegerType<T>::value || IsLogicalType<T>::value>> {
    constexpr static bool value = true;
};

template <MPI_Datatype_Basic T>
struct IsTypeOpAllowed<T, OpLOr, std::enable_if_t<IsIntegerType<T>::value || IsLogicalType<T>::value>> {
    constexpr static bool value = true;
};

template <MPI_Datatype_Basic T>
struct IsTypeOpAllowed<T, OpLXor, std::enable_if_t<IsIntegerType<T>::value || IsLogicalType<T>::value>> {
    constexpr static bool value = true;
};

template <MPI_Datatype_Basic T>
struct IsTypeOpAllowed<T, OpBAnd, std::enable_if_t<IsIntegerType<T>::value || IsByteType<T>::value>> {
    constexpr static bool value = true;
};

template <MPI_Datatype_Basic T>
struct IsTypeOpAllowed<T, OpBOr, std::enable_if_t<IsIntegerType<T>::value || IsByteType<T>::value>> {
    constexpr static bool value = true;
};

template <MPI_Datatype_Basic T>
struct IsTypeOpAllowed<T, OpBXor, std::enable_if_t<IsIntegerType<T>::value || IsByteType<T>::value>> {
    constexpr static bool value = true;
};

template <typename T, template <class> class Op, bool allowed>
struct TypeOpMatcher {
    static __device__ void run(void* /*invec*/, void* /*inoutvec*/, int */*len*/) {
        printf("gpu_mpi: Combination of type and operator is not allowed in reduce operation\n");
        __gpu_abort();
    }
};

template <typename T, template <class> class Op>
struct TypeOpMatcher<T, Op, true> {
    static __device__ void run(void* invec, void* inoutvec, int *len) {
        T* in = (T*) invec;
        T* inout = (T*) inoutvec;
        Op<T> op;
        for (int i = 0; i < *len; i++) {
            inout[i] = op(in[i], inout[i]);
        }
    }
};

#define MPI_TYPES_SWITCH_F(name, type) case name: TypeOpMatcher<type, Op, IsTypeOpAllowed<name, Op>::value>::run(invec, inoutvec, len); break;
#define MPI_TYPES_SWITCH_SEP

template <template <class> class Op>
__device__ void OpDispatcher(void* invec, void* inoutvec, int *len, MPI_Datatype *datatype) {
    switch (*datatype) {
        MPI_TYPES_LIST(MPI_TYPES_SWITCH_F, MPI_TYPES_SWITCH_SEP)
        default:
            assert(0);
    }
}

#undef MPI_TYPES_SWITCH_F
#undef MPI_TYPES_SWITCH_SEP



} // namespace

struct MPI_Op_impl {
    __device__ MPI_Op_impl(MPI_User_function* fn) : user_fn(fn) {}
    MPI_User_function* user_fn;
};

__device__ int MPI_Op_create(MPI_User_function* user_fn, int commute, MPI_Op* op) {
    (void)commute; // ignored
    *op = new MPI_Op_impl(user_fn);
    return MPI_SUCCESS;
}

__device__ int MPI_Op_free(MPI_Op *op) {
    delete *op;
    *op = MPI_OP_NULL;
    return MPI_SUCCESS;
}

namespace gpu_mpi {

#define MPI_OP_LIST_INIT_F(name, class) name = new MPI_Op_impl(OpDispatcher<class>);
#define MPI_OP_LIST_INIT_SEP
__device__ void initializeOps() {
    if (CudaMPI::sharedState().gridRank() == 0) {
        MPI_OPERATORS_LIST(MPI_OP_LIST_INIT_F, MPI_OP_LIST_INIT_SEP)
    }
    CudaMPI::sharedState().gridBarrier();
}
#undef MPI_OP_LIST_INIT_F
#undef MPI_OP_LIST_INIT_SEP

#define MPI_OP_LIST_DEL_F(name, class) delete name;
#define MPI_OP_LIST_DEL_SEP

__device__ void destroyOps() {
    CudaMPI::sharedState().gridBarrier();
    if (CudaMPI::sharedState().gridRank() == 0) {
        MPI_OPERATORS_LIST(MPI_OP_LIST_DEL_F, MPI_OP_LIST_DEL_SEP)
    }
}

#undef MPI_OP_LIST_DEL_F
#undef MPI_OP_LIST_DEL_SEP

__device__ void invokeOperator(MPI_Op op, const void* in, void* inout, int* len, MPI_Datatype* datatype) {
    op->user_fn(const_cast<void*>(in), inout, len, datatype);
}

} // namespace
