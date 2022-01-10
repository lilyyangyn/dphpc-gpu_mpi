#include "datatypes.cuh"
#include "mpi.cuh"

// cuda_mpi.cuh should be included before device specific standard library functions
// because it relies on standard ones
#include "cuda_mpi.cuh"

#include "stdlib.cuh"
#include "string.cuh"

#include "mpi_common.cuh"

#include "device_vector.cuh"

#include "operators.cuh"

#define MPI_COLLECTIVE_TAG (-2)

//nio
#define USE_AIO true
#define USE_URING false

// internal opaque object
struct MPI_Request_impl {
    //TODO: need better implementation, should not direct IOOperation in the declairation of the func.

    
    enum Type { SR, IO };

    union {
        CudaMPI::PendingOperation* pendingOperation;
        CudaMPI::PendingIOOperation* pendingIOOperation;
    };

    Type type;
    int ref_count;

    __device__ MPI_Request_impl(CudaMPI::PendingOperation* pendingOperation) 
        : ref_count(1) 
        , pendingOperation(pendingOperation) 
    {this->type = SR; }
    __device__ MPI_Request_impl(CudaMPI::PendingIOOperation* ioop, int io) 
        : ref_count(1)
        , pendingIOOperation(ioop)
    {this->type = Type(io);}
};
namespace gpu_mpi {
    
__device__ void incRequestRefCount(MPI_Request request) {
    assert(request->ref_count > 0);
    request->ref_count++;
}

#undef MPI_TYPES_LIST

} // namespace

__device__ int MPI_Init(int *argc, char ***argv) {
    gpu_mpi::initializeGlobalGroups();
    gpu_mpi::initializeGlobalCommunicators();
    gpu_mpi::initializeOps();
    return MPI_SUCCESS;
}

__device__ int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    (void) required;
    *provided = MPI_THREAD_SINGLE;
    return MPI_Init(argc, argv);
}

__device__ int MPI_Finalize(void) {
    // TODO: due to exit() you need to perform
    // all MPI related memory deallocation here


    gpu_mpi::destroyGlobalGroups();
    gpu_mpi::destroyGlobalCommunicators();
    
    gpu_mpi::destroyOps();
    
    return MPI_SUCCESS;
}

__device__ int MPI_Get_processor_name(char *name, int *resultlen) {
    const char hardcoded_name[] = "GPU thread";
    __gpu_strcpy(name, hardcoded_name);
    *resultlen = sizeof(hardcoded_name);
    return MPI_SUCCESS;
}

static __device__ CudaMPI::DeviceVector<char>* native_buf;
__device__ CudaMPI::DeviceVector<char>& nativeBuf() {
    if (!native_buf) {
        native_buf = new CudaMPI::DeviceVector<char>;
    }
    __gpu_assert(native_buf);
    return *native_buf;
}

__device__ int MPI_Bcast_native(void* buffer, int size, int root) {
    int commRank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    CudaMPI::sharedState().gridBarrier();
    if (root == commRank) {
        nativeBuf().resize(size);
        __gpu_memcpy(&nativeBuf()[0], buffer, size);
    }
    CudaMPI::sharedState().gridBarrier();
    if (root != commRank) {
        __gpu_memcpy(buffer, &nativeBuf()[0], size);
    }
    return MPI_SUCCESS;
}

__device__ int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                         int root, MPI_Comm comm)
{
    int dataSize = gpu_mpi::TypeSize(datatype) * count;
    assert(dataSize > 0);

    if (comm == MPI_COMM_WORLD) {
        return MPI_Bcast_native(buffer, dataSize, root);
    }
    
    int commSize = -1;
    int commRank = -1;
    
    MPI_Comm_size(comm, &commSize);
    MPI_Comm_rank(comm, &commRank);
    
    int tag = MPI_COLLECTIVE_TAG;
    int ctx = gpu_mpi::getCommContext(comm);
    
    if (commRank == root) {
        CudaMPI::PendingOperation** ops = (CudaMPI::PendingOperation**) malloc(sizeof(CudaMPI::PendingOperation*) * commSize);
        assert(ops);
        for (int dst = 0; dst < commSize; dst++) {
            if (dst != commRank) {
                ops[dst] = CudaMPI::isend(dst, buffer, dataSize, ctx, tag);
            }
        }
        for (int dst = 0; dst < commSize; dst++) {
            if (dst != commRank) {
                CudaMPI::wait(ops[dst]);
            }
        }
        free(ops);
    } else {
        CudaMPI::PendingOperation* op = CudaMPI::irecv(root, buffer, dataSize, ctx, tag);
        CudaMPI::wait(op);
    }
    
    return MPI_SUCCESS;
}

__device__ double MPI_Wtime(void) {
    auto clock = clock64();
    double seconds = clock * MPI_Wtick();
    return seconds;
}

__device__ int MPI_Reduce_native(
    const void *sendbuf, void *recvbuf, int count, int root) 
{
    int commRank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

    int elemSize = gpu_mpi::plainTypeSize(MPI_DOUBLE);
    int dataSize = elemSize * count;
    __gpu_assert(dataSize > 0);
    CudaMPI::sharedState().gridBarrier();
    if (root == commRank) {
        nativeBuf().resize(dataSize);
        double* native_buf_start = (double*)&nativeBuf()[0];
        for (int i = 0; i < count; i++) {
            native_buf_start[i] = ((double*)sendbuf)[i];
        }
    }
    CudaMPI::sharedState().gridBarrier();
    double* native_buf_start = (double*)&nativeBuf()[0];
    if (root != commRank) {
        for (int i = 0; i < count; i++) {
            atomicAdd(&native_buf_start[i], ((double*)sendbuf)[i]);
        }
    }
    CudaMPI::sharedState().gridBarrier();
    if (root == commRank) {
        for (int i = 0; i < count; i++) {
            ((double*)recvbuf)[i] = native_buf_start[i];
        }
    }
    return MPI_SUCCESS;
}

__device__ int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    if (comm == MPI_COMM_WORLD && op == MPI_SUM && datatype == MPI_DOUBLE) {
        return MPI_Reduce_native(sendbuf, recvbuf, count, root);
    }

    int elemSize = gpu_mpi::TypeSize(datatype);
    int dataSize = elemSize * count;
    __gpu_assert(dataSize > 0);
    
    int commSize = -1;
    int commRank = -1;
    MPI_Comm_size(comm, &commSize);
    MPI_Comm_rank(comm, &commRank);
    
    int tag = MPI_COLLECTIVE_TAG;
    int ctx = gpu_mpi::getCommContext(comm);
    
    if (commRank == root) {
        auto ops = (CudaMPI::PendingOperation**) malloc(sizeof(CudaMPI::PendingOperation*) * commSize);
        void* buffers = malloc(dataSize * commSize);
        assert(ops);
        for (int src = 0; src < commSize; src++) {
            if (src != commRank) {
                ops[src] = CudaMPI::irecv(src, ((char*)buffers) + src * dataSize, dataSize, ctx, tag);
            }
        }
        for (int src = 0; src < commSize; src++) {
            const void* tempbuf = nullptr;
            if (src != commRank) {
                CudaMPI::wait(ops[src]);
                tempbuf = ((char*)buffers) + src * dataSize;
            } else {
                tempbuf = sendbuf;
            }
            
            if (src == 0) {
                for (int i = 0; i < dataSize; i++) {
                    ((char*)recvbuf)[i] = ((char*)tempbuf)[i];
                }
            } else {
                gpu_mpi::invokeOperator(op, tempbuf, recvbuf, &count, &datatype);
            }
        }
        
        free(buffers);
        free(ops);
    } else {
        CudaMPI::PendingOperation* op = CudaMPI::isend(root, sendbuf, dataSize, ctx, tag);
        CudaMPI::wait(op);
    }
    
    return MPI_SUCCESS;
}

// extern std::list<MPI_Datatype*> Typelist;

__device__ int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype) {
    // newtype -> _size = count * oldtype.size();
    // newtype -> typemap_len = count * oldtype.typemap_len;
    // for (int i = 0; i < count; i++) copy_typemap_once(newtype, oldtype, i, 0);
    // newtype -> committed = false;
    MPI_Datatype contiguousType;
    for (int i = 0; i < count; i++) contiguousType.add_typemap_at_end(oldtype, 0);
    *newtype = contiguousType;
    return MPI_SUCCESS;
}

__device__ int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype) {
    if (blocklength == 0 || stride == 0) return MPI_ERR_OTHER;
    if (stride < blocklength) return MPI_ERR_OTHER;
    // In all our use cases, we do not allow a typemap to overlap itself.

    MPI_Datatype vectorType;
    int gap = (int)(stride - blocklength) * oldtype.size();
    for(int i = 0; i < count; i++){
        vectorType.add_typemap_at_end(oldtype, gap);
        for(int j = 1; j < blocklength; j++){
            vectorType.add_typemap_at_end(oldtype, 0);
        }
    }
    vectorType.typemap_gap = gap;

    *newtype = vectorType;  
    return MPI_SUCCESS;
}

__device__ int MPI_Type_commit(MPI_Datatype *datatype) {
    if (datatype -> committed == true) return MPI_SUCCESS;
    datatype -> committed = true;  // Typelist.push_back(datatype);
    return MPI_SUCCESS;
}

__device__ int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
                        int source, int tag, MPI_Comm comm, MPI_Status *status) {
    MPI_Request request;
    MPI_Irecv(buf, count, datatype, source, tag, comm, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
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
    MPI_Request request;
    MPI_Isend(buf, count, datatype, dest, tag, comm, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    return MPI_SUCCESS;
}

__device__ double MPI_Wtick() {
    int peakClockKHz = CudaMPI::threadPrivateState().peakClockKHz;
    return 0.001 / peakClockKHz;
}

__device__ int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int err = MPI_Reduce(sendbuf, recvbuf, count, datatype, op, 0, comm);
    if (err != MPI_SUCCESS) return err;
    return MPI_Bcast(recvbuf, count, datatype, 0, comm);
}
__device__ int MPI_Abort(MPI_Comm comm, int errorcode) {
    NOT_IMPLEMENTED;
    return MPI_SUCCESS;
}
__device__ int MPI_Type_size(MPI_Datatype datatype, int *size) {
    NOT_IMPLEMENTED;
    return MPI_SUCCESS;
}
__device__ int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                          void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                          MPI_Comm comm)
{
    // TODO implement through MPI_Gatherv
    int comm_size = -1;
    int comm_rank = -1;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    int sendElemSize = gpu_mpi::TypeSize(sendtype);
    int recvElemSize = gpu_mpi::TypeSize(recvtype);
    assert(sendElemSize > 0);
    assert(recvElemSize > 0);

    assert(sendElemSize * sendcount == recvElemSize * recvcount);
    int dataSize = sendElemSize * sendcount;

    if (comm_rank != root) {
        MPI_Send(sendbuf, sendcount, sendtype, root, MPI_COLLECTIVE_TAG, comm);
    } else {
        for (int r = 0; r < comm_size; r++) {
            if (r == root) {
                memcpy(((char*)recvbuf) + r * dataSize, sendbuf, dataSize);
            } else {
                MPI_Recv(((char*)recvbuf) + r * dataSize, recvcount, recvtype, r, MPI_COLLECTIVE_TAG, comm, MPI_STATUS_IGNORE);
            }
        }
    }
    
    return MPI_SUCCESS;
}

__device__ int MPI_Barrier(MPI_Comm comm) {
    if (comm == MPI_COMM_WORLD) { 
        CudaMPI::sharedState().gridBarrier();
    } else {
        NOT_IMPLEMENTED;
    }
    return MPI_SUCCESS;
}

__device__ int MPI_Alltoall(
    const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype,
    MPI_Comm comm)
{
    int comm_size = -1;
    MPI_Comm_size(comm, &comm_size);

    int* sdispls = (int*) malloc(comm_size * sizeof(int));
    int* rdispls = (int*) malloc(comm_size * sizeof(int));
    int* sendcounts = (int*) malloc(comm_size * sizeof(int));
    int* recvcounts = (int*) malloc(comm_size * sizeof(int));

    for (int i = 0; i < comm_size; i++) {
        sdispls[i] = i * sendcount;
        rdispls[i] = i * recvcount;
        sendcounts[i] = sendcount;
        recvcounts[i] = recvcount;
    }
    int res = MPI_Alltoallv(
        sendbuf, sendcounts, sdispls, sendtype,
        recvbuf, recvcounts, rdispls, recvtype, comm);

    free(sdispls);
    free(rdispls);
    free(sendcounts);
    free(recvcounts);

    return res;
}
__device__ int MPI_Alltoallv(
    const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype,
    void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, 
    MPI_Comm comm) 
{
    int comm_size = -1;
    int comm_rank = -1;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    int sendElemSize = gpu_mpi::TypeSize(sendtype);
    int recvElemSize = gpu_mpi::TypeSize(recvtype);

    MPI_Request* send_requests = (MPI_Request*) malloc(sizeof(MPI_Request) * comm_size);
    MPI_Request* recv_requests = (MPI_Request*) malloc(sizeof(MPI_Request) * comm_size);
    assert(send_requests && "Can't allocate memory");
    assert(recv_requests && "Can't allocate memory");

    for (int i = 0; i < comm_size; i++) {
        if (i != comm_rank) {
            MPI_Isend(((char*)sendbuf) + sdispls[i] * sendElemSize, sendcounts[i], sendtype, i, MPI_COLLECTIVE_TAG, comm, &send_requests[i]);
        }
    }

    for (int i = 0; i < comm_size; i++) {
        if (i != comm_rank) {
            MPI_Irecv(((char*)recvbuf) + rdispls[i] * recvElemSize, recvcounts[i], recvtype, i, MPI_COLLECTIVE_TAG, comm, &recv_requests[i]);
        }
    }

    memcpy(((char*)recvbuf) + rdispls[comm_rank] * recvElemSize, 
           ((char*)sendbuf) + sdispls[comm_rank] * sendElemSize,
           recvcounts[comm_rank] * recvElemSize);

    for (int i = 0; i < comm_size; i++) {
        if (i != comm_rank) {
            MPI_Wait(&send_requests[i], MPI_STATUS_IGNORE);
            MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
        }
    }

    free(send_requests);
    free(recv_requests);

    return MPI_SUCCESS;
}

__device__ int MPI_Allgather(const void *sendbuf, int  sendcount,
             MPI_Datatype sendtype, void *recvbuf, int recvcount,
             MPI_Datatype recvtype, MPI_Comm comm)
{
    MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, 0, comm);
    int comm_size = -1;
    MPI_Comm_size(comm, &comm_size);
    MPI_Bcast(recvbuf, recvcount * comm_size, recvtype, 0, comm);
    return MPI_SUCCESS;
}

__device__ int MPI_Allgatherv(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                              const int displs[], MPI_Datatype recvtype, MPI_Comm comm)
{
    NOT_IMPLEMENTED;
    return MPI_SUCCESS;
}

__device__ int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype,
                           int root, MPI_Comm comm)
{
    // TODO implement through MPI_Igatherv
    int comm_size = -1;
    int comm_rank = -1;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    int sendElemSize = gpu_mpi::TypeSize(sendtype);
    assert(sendElemSize > 0);

    if (comm_rank != root) {
        MPI_Send(sendbuf, sendcount, sendtype, root, MPI_COLLECTIVE_TAG, comm);
    } else {
        int recvElemSize = gpu_mpi::TypeSize(recvtype);
        assert(recvElemSize > 0);
        for (int r = 0; r < comm_size; r++) {
            if (r == root) {
                memcpy(((char*)recvbuf) + displs[r] * recvElemSize, sendbuf, recvcounts[r] * recvElemSize);
            } else {
                MPI_Recv(((char*)recvbuf) + displs[r] * recvElemSize, recvcounts[r], recvtype, r, MPI_COLLECTIVE_TAG, comm, MPI_STATUS_IGNORE);
            }
        }
    }
    
    return MPI_SUCCESS;
}
__device__ int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                           MPI_Comm comm)
{
    // TODO implement through MPI_Scatterv
    int comm_size = -1;
    int comm_rank = -1;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    int sendElemSize = gpu_mpi::TypeSize(sendtype);
    int recvElemSize = gpu_mpi::TypeSize(recvtype);
    assert(sendElemSize > 0);
    assert(recvElemSize > 0);

    assert(sendElemSize * sendcount == recvElemSize * recvcount);
    int dataSize = sendElemSize * sendcount;

    if (comm_rank != root) {
        MPI_Recv(recvbuf, recvcount, recvtype, root, MPI_COLLECTIVE_TAG, comm, MPI_STATUS_IGNORE);
    } else {
        for (int r = 0; r < comm_size; r++) {
            if (r == root) {
                memcpy(recvbuf, ((char*)sendbuf) + r * dataSize, dataSize);
            } else {
                MPI_Send(((char*)sendbuf) + r * dataSize, sendcount, sendtype, r, MPI_COLLECTIVE_TAG, comm);
            }
        }
    }
    
    return MPI_SUCCESS;
}
__device__ int MPI_Scatterv(const void *sendbuf, const int sendcounts[], const int displs[],
                            MPI_Datatype sendtype, void *recvbuf, int recvcount,
                            MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    int comm_size = -1;
    int comm_rank = -1;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    int recvElemSize = gpu_mpi::TypeSize(recvtype);
    assert(recvElemSize > 0);

    if (comm_rank != root) {
        MPI_Recv(recvbuf, recvcount, recvtype, root, MPI_COLLECTIVE_TAG, comm, MPI_STATUS_IGNORE);
    } else {
        int sendElemSize = gpu_mpi::TypeSize(sendtype);
        assert(sendElemSize > 0);
        for (int r = 0; r < comm_size; r++) {
            if (r == root) {
                memcpy(recvbuf, ((char*)sendbuf) + displs[r] * sendElemSize, sendcounts[r] * sendElemSize);
            } else {
                MPI_Send(((char*)sendbuf) + displs[r] * sendElemSize, sendcounts[r], sendtype, r, MPI_COLLECTIVE_TAG, comm);
            }
        }
    }
    
    return MPI_SUCCESS;
}

__device__ int MPI_NULL_COPY_FN(MPI_Comm oldcomm, int keyval,
                     void *extra_state, void *attribute_val_in,
                     void *attribute_val_out, int *flag) {
    NOT_IMPLEMENTED;
    return MPI_SUCCESS;
}

__device__ int MPI_NULL_DELETE_FN(MPI_Comm comm, int keyval,
                       void *attribute_val, void *extra_state) {
    NOT_IMPLEMENTED;
    return MPI_SUCCESS;
}

__device__ int MPI_Keyval_create(MPI_Copy_function *copy_fn,
                                 MPI_Delete_function *delete_fn, int *keyval, void *extra_state) {
    NOT_IMPLEMENTED;
    return MPI_SUCCESS;
}

__device__ int MPI_Dims_create(int nnodes, int ndims, int dims[]) {
    NOT_IMPLEMENTED;
    return MPI_SUCCESS;
}

__device__ int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
               int source, int tag, MPI_Comm comm, MPI_Request *request)
{
    int ctx = gpu_mpi::getCommContext(comm);
    
    int dataSize = gpu_mpi::TypeSize(datatype) * count;
    assert(dataSize > 0);
    
    CudaMPI::PendingOperation* op = CudaMPI::irecv(source, buf, dataSize, ctx, tag);
    
    if (request) {
        *request = new MPI_Request_impl(op);
    }
    
    return MPI_SUCCESS;
}
__device__ int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm, MPI_Request *request) 
{
    int ctx = gpu_mpi::getCommContext(comm);
    
    int dataSize = gpu_mpi::TypeSize(datatype) * count;
    assert(dataSize > 0);
    
    CudaMPI::PendingOperation* op = CudaMPI::isend(dest, buf, dataSize, ctx, tag);
    
    *request = new MPI_Request_impl(op);
    return MPI_SUCCESS;
}

__device__ int MPI_Testall(int count, MPI_Request array_of_requests[],
            int *flag, MPI_Status array_of_statuses[]) {
    NOT_IMPLEMENTED;
    return MPI_SUCCESS;
}

__device__ int MPI_Waitall(int count, MPI_Request array_of_requests[],
            MPI_Status *array_of_statuses) {
    for (int i = 0; i < count; i++) {
        MPI_Wait(&array_of_requests[i], &array_of_statuses[i]);
    }
    return MPI_SUCCESS;
}

__device__ int MPI_Initialized(int *flag) {
    NOT_IMPLEMENTED;
    return MPI_SUCCESS;
}

__device__ int MPI_Waitsome(int incount, MPI_Request array_of_requests[],
            int *outcount, int array_of_indices[],
            MPI_Status array_of_statuses[]) {
    NOT_IMPLEMENTED;
    return MPI_SUCCESS;
}

__device__ int MPI_Wait(MPI_Request *request, MPI_Status *status) {
    if (request == MPI_REQUEST_NULL) {
        if (status) *status = MPI_Status();
    }
    
    switch ((*request)->type) {
        case MPI_Request_impl::Type::SR:
            CudaMPI::wait((*request)->pendingOperation);
            MPI_Request_free(request);
            if (status) *status = MPI_Status();
            break;
        case MPI_Request_impl::Type::IO:
            CudaMPI::waitIO((*request)->pendingIOOperation);
            MPI_Request_free(request);
            if (status) *status = MPI_Status();// ?
            break;
    }

    return MPI_SUCCESS;
}



__device__ int MPI_Request_free(MPI_Request *request) {
    switch ((*request)->type) {
        case MPI_Request_impl::Type::SR:
            assert((*request)->ref_count > 0);
            (*request)->ref_count--;
            if ((*request)->ref_count == 0) delete *request;
            *request = MPI_REQUEST_NULL;
            break;
        case MPI_Request_impl::Type::IO:
            assert((*request)->ref_count > 0);
            (*request)->ref_count--;
            if ((*request)->ref_count == 0) delete *request;
            *request = MPI_REQUEST_NULL;
            break;
    }
    return MPI_SUCCESS;
}

struct MPI_File;
/* ----- Non-blocking IO ------ */
#if USE_AIO && !USE_URING
__device__ int MPI_File_iread(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request){
    if (!(fh.amode & MPI_MODE_RDONLY) && !(fh.amode & MPI_MODE_RDWR)) return MPI_ERR_AMODE;
    if (fh.amode & MPI_MODE_SEQUENTIAL) return MPI_ERR_UNSUPPORTED_OPERATION;  // p514 l43
    int rank,size;
    MPI_Comm_rank(fh.comm, &rank);
    MPI_Comm_size(fh.comm, &size);
    if(rank == 0) {
        int msg_size = 48;  // sizeof(int) + sizeof(r_param) + sizeof(char) * (count + 1);: misaligned address
        int buffer_size = count*datatype.size();
        void *msg =  CudaMPI::sharedState().freeManagedMemory.allocate(msg_size);
        void *data = CudaMPI::sharedState().freeManagedMemory.allocate(buffer_size);
        char *p = (char*)msg;
        *((int*)p) = I_FILE_IREAD;          p += 8;
        *((FILE**)p) = fh.file;             p += 8;
        *((size_t*)p) = buffer_size;        p += 8;
        *((off_t*)p) = fh.seek_pos[rank];   p += 8;
        *((void**)p) = data;                p += 8; //aio temp buf, this is managed memory
        // *((void**)p) = buf;                 //dest buf, this is in process private space


        CudaMPI::sharedState().deviceToHostCommunicator.delegateToHost(msg, msg_size);  //schedule io task
        /* // debug only:
        // while (*((int*)msg) != I_READY){}
        // memcpy(buf, p, datatype.size() * count);
        */
        
        aiocb* newcb_p = new aiocb; //TODO: delete

        size_t ret = ((size_t*)msg)[1]; //ret of scheduling
        p = (char*)msg+8; 
        memcpy(newcb_p,(aiocb*)p,sizeof(aiocb));
        // *newcb_p = *((aiocb*)p); //get the cb
        //ori:: CudaMPI::PendingOperation* op = CudaMPI::iread(newcb_p);
        CudaMPI::PendingIOOperation* ioop = new CudaMPI::PendingIOOperation;
        ioop->aiocb_p = newcb_p;
        ioop->buf = buf;
        (*request) = new (MPI_Request_impl)(ioop, 1); //TODO: delete

        MPI_File_seek(fh, newcb_p->aio_nbytes, MPI_SEEK_CUR);// documentation P518 L30
        CudaMPI::sharedState().freeManagedMemory.free(msg);
        return ret;
    }
    return true;
}


#elif USE_URING
__device__ int MPI_File_iread(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request){}
#elif !USE_AIO
__device__ int MPI_File_iread(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request){}
#endif



