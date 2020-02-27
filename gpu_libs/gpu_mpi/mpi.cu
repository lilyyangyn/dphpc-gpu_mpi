#include "mpi.h.cuh"

// cuda_mpi.cuh should be included before device specific standard library functions
// because it relies on standard ones
#include "cuda_mpi.cuh"

#include "stdlib.h.cuh"
#include "string.h.cuh"

#include <cooperative_groups.h>


using namespace cooperative_groups;


__device__ MPI_Comm MPI_COMM_WORLD = 0;
__device__ MPI_Comm MPI_COMM_NULL = 1;


__device__ int MPI_Init(int *argc, char ***argv) {
    // nothing to do
    return MPI_SUCCESS;
}

__device__ int MPI_Finalize(void) {
    // TODO: due to exit() you need to perform
    // all MPI related memory deallocation here

    // notify host that there will be no messages from this thread anymore
    CudaMPI::sharedState().deviceToHostCommunicator.delegateToHost(0, 0);

    return MPI_SUCCESS;
}

__device__ int MPI_Comm_size(MPI_Comm comm, int *size) {
    assert(size);
    if (comm == MPI_COMM_WORLD) {
        auto multi_grid = this_multi_grid();
        *size = multi_grid.size();
    } else {
        NOT_IMPLEMENTED
    }
    return MPI_SUCCESS;
}

__device__ int MPI_Comm_rank(MPI_Comm comm, int *rank) {
    assert(rank);
    if (comm == MPI_COMM_WORLD) {
        auto multi_grid = this_multi_grid();
        *rank = multi_grid.thread_rank();
    } else {
        NOT_IMPLEMENTED
    }
    return MPI_SUCCESS;
}

__device__ int MPI_Get_processor_name(char *name, int *resultlen) {
    const char hardcoded_name[] = "GPU thread";
    strcpy(name, hardcoded_name);
    *resultlen = sizeof(hardcoded_name);
    return MPI_SUCCESS;
}

// #define CAT1(X,Y) X##Y
// #define CAT(X,Y) CAT1(X,Y)
// #define $ \
//     int CAT(rank, __LINE__);\
//     MPI_Comm_rank(MPI_COMM_WORLD, &CAT(rank, __LINE__));\
//     printf("ALIVE %s:%d rank %d\n", __FILE__, __LINE__, CAT(rank, __LINE__));

__device__ int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                         int root, MPI_Comm comm)
{
    int dataSize = -1;
    switch (datatype) {
        case MPI_INT:
            dataSize = count * sizeof(int);
            break;
        default:
            assert(0);
    }
    int commSize = -1;
    int commRank = -1;
    
    MPI_Comm_size(comm, &commSize);
    MPI_Comm_rank(comm, &commRank);
    
    int tag = 123; // TODO use reserved tag
    
    if (commRank == root) {
        CudaMPI::PendingOperation** ops = (CudaMPI::PendingOperation**) malloc(sizeof(CudaMPI::PendingOperation*) * commSize);
        assert(ops);
        for (int dst = 0; dst < commSize; dst++) {
            if (dst != commRank) {
                ops[dst] = CudaMPI::isend(dst, buffer, dataSize, comm, tag);
            }
        }
        for (int dst = 0; dst < commSize; dst++) {
            if (dst != commRank) {
                CudaMPI::wait(ops[dst]);
            }
        }
        free(ops);
    } else {
        CudaMPI::PendingOperation* op = CudaMPI::irecv(root, buffer, dataSize, comm, tag);
        CudaMPI::wait(op);
    }
    
    return MPI_SUCCESS;
}

__device__ double MPI_Wtime(void) {
    auto clock = clock64();
    double seconds = clock * MPI_Wtick();
    return seconds;
}

__device__ int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    int commSize = -1;
    int commRank = -1;
    MPI_Comm_size(comm, &commSize);
    MPI_Comm_rank(comm, &commRank);

    assert(datatype == MPI_DOUBLE); // TODO
    int dataSize = sizeof(double) * count;
    
    int tag = 124; // TODO use reserved tag
    
    if (commRank == root) {
        auto ops = (CudaMPI::PendingOperation**) malloc(sizeof(CudaMPI::PendingOperation*) * commSize);
        double* buffers = (double*) malloc(dataSize * commSize);
        assert(ops);
        for (int src = 0; src < commSize; src++) {
            if (src != commRank) {
                ops[src] = CudaMPI::irecv(src, buffers + src * count, dataSize, comm, tag);
            }
        }
        for (int i = 0; i < count; i++) {
            assert(op == MPI_SUM);
            double* recvBufDouble = (double*) recvbuf;
            recvBufDouble[i] = 0;
        }
        for (int src = 0; src < commSize; src++) {
            double* tempBufDouble = nullptr;
            if (src != commRank) {
                CudaMPI::wait(ops[src]);
                tempBufDouble = buffers + src * count;
            } else {
                tempBufDouble = (double*) sendbuf;
            }
            double* recvBufDouble = (double*) recvbuf;
            
            for (int i = 0; i < count; i++) {
                assert(op == MPI_SUM);
                recvBufDouble[i] += tempBufDouble[i];
            }
        }
        
        free(buffers);
        free(ops);
    } else {
        CudaMPI::PendingOperation* op = CudaMPI::isend(root, sendbuf, dataSize, comm, tag);
        CudaMPI::wait(op);
    }
    
    return MPI_SUCCESS;
}

__device__ int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype) {
    return MPI_SUCCESS;
}

__device__ int MPI_Type_commit(MPI_Datatype *datatype) {
    return MPI_SUCCESS;
}

__device__ int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
                        int source, int tag, MPI_Comm comm, MPI_Status *status) {
    return MPI_SUCCESS;
}

__device__ int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag, void *recvbuf, int recvcount,
            MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status) {
    return MPI_SUCCESS;
}

__device__ int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
            int tag, MPI_Comm comm)
{
    return MPI_SUCCESS;
}

__device__ double MPI_Wtick() {
    int peakClockKHz = CudaMPI::threadPrivateState().peakClockKHz;
    return 0.001 / peakClockKHz;
}
__device__ int MPI_Comm_group(MPI_Comm comm, MPI_Group *group) {
    return MPI_SUCCESS;
}
__device__ int MPI_Group_incl(MPI_Group group, int n, const int ranks[],
            MPI_Group *newgroup) {
    return MPI_SUCCESS;
}
__device__ int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Group_free(MPI_Group *group) {
    return MPI_SUCCESS;
}
__device__ int MPI_Group_rank(MPI_Group group, int *rank) {
    return MPI_SUCCESS;
}
__device__ int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Abort(MPI_Comm comm, int errorcode) {
    return MPI_SUCCESS;
}
__device__ int MPI_Type_size(MPI_Datatype datatype, int *size) {
    return MPI_SUCCESS;
}
__device__ int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
            MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Attr_get(MPI_Comm comm, int keyval,void *attribute_val,
            int *flag ) {
    return MPI_SUCCESS;
}
__device__ int MPI_Barrier(MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Alltoall(const void *sendbuf, int sendcount,
            MPI_Datatype sendtype, void *recvbuf, int recvcount,
            MPI_Datatype recvtype, MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Alltoallv(const void *sendbuf, const int sendcounts[],
            const int sdispls[], MPI_Datatype sendtype,
            void *recvbuf, const int recvcounts[],
            const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Allgather(const void *sendbuf, int  sendcount,
             MPI_Datatype sendtype, void *recvbuf, int recvcount,
             MPI_Datatype recvtype, MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Allgatherv(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                              const int displs[], MPI_Datatype recvtype, MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype,
                           int root, MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                           MPI_Comm comm) {
    return MPI_SUCCESS;
}

__device__ int MPI_NULL_COPY_FN(MPI_Comm oldcomm, int keyval,
                     void *extra_state, void *attribute_val_in,
                     void *attribute_val_out, int *flag) {
    return MPI_SUCCESS;
}

__device__ int MPI_NULL_DELETE_FN(MPI_Comm comm, int keyval,
                       void *attribute_val, void *extra_state) {
    return MPI_SUCCESS;
}

__device__ int MPI_Keyval_create(MPI_Copy_function *copy_fn,
                                 MPI_Delete_function *delete_fn, int *keyval, void *extra_state) {
    return MPI_SUCCESS;
}
__device__ int MPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val) {
    return MPI_SUCCESS;
}
__device__ int MPI_Dims_create(int nnodes, int ndims, int dims[]) {
    return MPI_SUCCESS;
}
__device__ int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],
                               const int periods[], int reorder, MPI_Comm *comm_cart) {
    return MPI_SUCCESS;
}
__device__ int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *comm_new) {
    return MPI_SUCCESS;
}
__device__ int MPI_Comm_split(MPI_Comm comm, int color, int key,
                              MPI_Comm *newcomm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
               int source, int tag, MPI_Comm comm, MPI_Request *request) {
    return MPI_SUCCESS;
}
__device__ int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm, MPI_Request *request) {
    return MPI_SUCCESS;
}
__device__ int MPI_Testall(int count, MPI_Request array_of_requests[],
            int *flag, MPI_Status array_of_statuses[]) {
    return MPI_SUCCESS;
}
__device__ int MPI_Waitall(int count, MPI_Request array_of_requests[],
            MPI_Status *array_of_statuses) {
    return MPI_SUCCESS;
}
__device__ int MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3],
                                    MPI_Group *newgroup) {
    return MPI_SUCCESS;
}
__device__ int MPI_Comm_free(MPI_Comm *comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Initialized(int *flag) {
    return MPI_SUCCESS;
}

__device__ int MPI_Waitsome(int incount, MPI_Request array_of_requests[],
            int *outcount, int array_of_indices[],
            MPI_Status array_of_statuses[]) {
    return MPI_SUCCESS;
}
__device__ int MPI_Wait(MPI_Request *request, MPI_Status *status) {
    return MPI_SUCCESS;
}








