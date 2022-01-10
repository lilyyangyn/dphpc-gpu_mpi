#ifndef CUDA_SEND_RECV
#define CUDA_SEND_RECV

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <type_traits>
#include <aio.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "device_vector.cuh"
#include "common.h"

// this header may change or disappear in the future cuda releases
// in this case copy its contents from CUDA 11.4 
#include <cooperative_groups/details/helpers.h>

// copied from io.cuh
#define I_READY                         0x80000000
#define I_FILE_ITEST                    0x00000404

namespace CudaMPI {



namespace THREAD_LOCALITY {
    enum Type {
        WARP,
        BLOCK,
        GRID,
        MULTI_GRID
    };
}

namespace MEMORY_LOCALITY {
    enum Type {
        GLOBAL,
        LOCAL,
        SHARED,
        CONST,
        OTHER
    };
}

struct CircularBufferState {
    __host__ __device__ CircularBufferState(int size)
        : size(size)
        , used(0)
        , head(0)
        , tail(0)
    {
    }

    __host__ __device__ bool empty() const { return used == 0; }
    __host__ __device__ bool full() const { return used == size; }

    // reserve and return position for new element at the tail of queue
    __host__ __device__ int push() {
        assert(!full());
        used += 1;
        int position = tail;
        tail = (position + 1) % size;
        return position;
    }

    // release and return (released) position of element from the head of queue
    __host__ __device__ int pop() {
        assert(!empty());
        used -= 1;
        int position = head;
        head = (position + 1) % size;
        return position;
    }

    int used;
    int head; // first
    int tail; // next after last
    int size;
};

inline __host__ __device__ void cudaGlobalFence() {
    #if defined(__CUDA_ARCH__)
        __threadfence_system();
    #else
        __sync_synchronize();
    #endif
}

class ManagedMemoryLock {
public:
    ManagedMemoryLock() : locked(0) {
    }

    __host__ __device__ bool tryLock() volatile {
        // Since CAS returns old value, the operation is successful
        // if an old value (second arg of CAS) equal to the return value
        bool success = false;
        #if defined(__CUDA_ARCH__)
            success = (0 == atomicCAS_system((unsigned*)&locked, 0, 1));
        #else
            success = (0 == __sync_val_compare_and_swap((unsigned*)&locked, 0, 1));
        #endif
        return success;
    }

    __host__ __device__ void lock() volatile {
        while (!tryLock()) {}
    }

    __host__ __device__ void unlock() volatile {
        cudaGlobalFence();
        locked = 0;
    }
private:
    unsigned locked;
};

template <typename T, template <typename> typename Vector>
class CircularQueue {
public:
    
    __host__ __device__ CircularQueue(int size)
        : data(size)
        , active(size, false)
        , bufferState(size)
    {
    }
    
    __host__ __device__ int size() {
        return bufferState.size;
    }
    
    __host__ __device__ int used() {
        return bufferState.used;
    }
    
    __host__ __device__ bool empty() {
        return bufferState.empty();
    }
    
    __host__ __device__ int full() {
        return bufferState.full();
    }

    __host__ __device__ int push(const T& md) {
        int position = bufferState.push();
        data[position] = md;
        active[position] = true;
        return position;
    }
    
    __host__ __device__ T& get(int position) {
        assert(0 <= position && position < data.size());
        assert(active[position]);
        return data[position];
    }
    
    __host__ __device__ void pop(T* elem) {
        int index = elem - &data[0];
        assert(0 <= index && index < data.size());
        active[index] = false;
        if (bufferState.head == index) {
            while (!active[index] && !bufferState.empty()) {
                int removedIndex = bufferState.pop();
                assert(removedIndex == index);
                index = (index + 1) % data.size();
            }
        }
    }

    __host__ __device__ T* head() {
        if (bufferState.empty()) return nullptr;
        assert(active[bufferState.head]);
        return &data[bufferState.head];
    }

    __host__ __device__ T* next(T* elem) {
        assert(elem);
        int curIndex = elem - &data[0];
        auto next = [size=data.size()] (int idx) { return (idx + 1) % size; };
        for (int idx = next(curIndex); idx != bufferState.tail; idx = next(idx)) {
            if (active[idx]) return &data[idx];
        }
        return nullptr;
    }

private:
    Vector<T> data;
    Vector<bool> active;
    CircularBufferState bufferState;
};

enum { ANY_SRC = -1 };
enum { ANY_TAG = -1 };

struct PendingOperation {
    enum class Type { SEND, RECV };

    // one of two state transitions are possible
    // STARTED -> COMPLETED (when operation is already posted by another thread)
    // STARTED -> POSTED -> COMPLETED (when operation is not yet posted)
    enum class State {
        STARTED,
        POSTED,
        COMPLETED
    };

    Type type = Type::SEND;
    State state = State::STARTED;

    int otherThread = 0;
    int count = 0;
    int ctx = 0;
    int tag = 0;

    // if user requested "synchronous" mode (with MPI_SSEND)
    bool isSynchronous = false;
    // if user requrested "buffered" mode (with MPI_BSEND)
    bool isBuffered = false;

    // this flag is changed by MPI_WAIT or MPI_TEST
    // when it is set to true, progress engine allowed to drop pending operation
    bool canBeFreed = false;

    void* data = nullptr;

    // only for receiver to store temporary allocated buffer
    void* buffer = nullptr;

    // this variable is changed by matching thread when operation is done
    volatile bool done = false;

};

struct PendingIOOperation { //created by student, mimicing PendingOperation
    enum class State {
        STARTED,
        COMPLETED
    };
    
    State state = State::STARTED;

    // this flag is changed by MPI_WAIT or MPI_TEST
    // when it is set to true, progress engine allowed to drop pending operation
    bool canBeFreed = false;

    //for aio control block
    union{
        aiocb* aiocb_p = nullptr;
    };

    void* buf = nullptr;

    // this variable is changed by matching thread when operation is done
    volatile bool done = false;
};


// This class is used for tracking skipped operations in STARTED state.
// It helps to prevent breaking the standardized order of MPI operations.
class ProgressState {
public:
    __device__ ProgressState()
        : mStartedSendSkip(false)
        , mStartedRecvSkip(false)
    {
    }
    
    // TODO: Current implementation is too simplistic and conservative,
    // it can be improved later by considering numbers of src and dst threads.
    __device__ void markStartedSendSkip(int /*dst*/) { mStartedSendSkip = true; }
    __device__ bool isStartedSendSkip(int /*dst*/) { return mStartedSendSkip; }
    
    __device__ void markStartedRecvSkip(int /*src*/) { mStartedRecvSkip = true; }
    __device__ bool isStartedRecvSkip(int /*src*/) { return mStartedRecvSkip; }
private:
    bool mStartedSendSkip;
    bool mStartedRecvSkip;
};

__host__ __device__ void progress();

__device__ void progressSend(PendingOperation& send, ProgressState& state);
__device__ void progressRecv(PendingOperation& recv, ProgressState& state);

__device__ void progressIO(PendingIOOperation& ioop, ProgressState& state);

__device__ void progressStartedRecv(PendingOperation& recv, ProgressState& state);
__device__ void progressPostedRecv(PendingOperation& recv);
__device__ void progressCompletedRecv(PendingOperation& recv);

__device__ void progressStartedSend(PendingOperation& send, ProgressState& state);
__device__ void progressPostedSend(PendingOperation& send);
__device__ void progressCompletedSend(PendingOperation& send);

struct GlobalVarsStorage {

    struct Entry {
        const void* key = nullptr;
        void* value = nullptr;
    };

    __device__ ~GlobalVarsStorage() {
        for (int i = 0; i < simpleSet.size(); i++) {
            free(simpleSet[i].value);
        }
    }

    __device__ int find_by_key(const void* ptr) {
        for (int i = 0; i < simpleSet.size(); i++) {
            if (simpleSet[i].key == ptr) {
                return i;
            }
        }
        return -1; // not found
    }

    __device__ int find_by_value(const void* ptr) {
        for (int i = 0; i < simpleSet.size(); i++) {
            if (simpleSet[i].value == ptr) {
                return i;
            }
        }
        return -1; // not found
    }

    __device__ void* getValue(const void* ptr, size_t size) {
        // first we check is it an adress that is already converted
        int idx = find_by_value(ptr);
        if (idx >= 0) return simpleSet[idx].value;

        // it is real global variable, so we find a mapping for it
        // and create new mapping if it is not exists
        idx = find_by_key(ptr);
        if (idx < 0) {
            // not found
            simpleSet.resize(simpleSet.size() + 1);
            idx = simpleSet.size() - 1;
            simpleSet[idx].key = ptr;
            void* copyPtr = malloc(size);
            memcpy(copyPtr, ptr, size);
            simpleSet[idx].value = copyPtr;
        }
        return simpleSet[idx].value;
    }

private:
    // TODO inefficient, refactor later
    DeviceVector<Entry> simpleSet;
};

struct ThreadPrivateState {

    struct Context {
        Context() :
            pendingBufferSize(20),
            peakClockKHz(-1)
            {}

        __device__ bool valid() const {
            if (pendingBufferSize <= 0) return false;
            if (peakClockKHz <= 0) return false;
            return true;
        }

        int pendingBufferSize;
        int peakClockKHz;
    };

    __device__ PendingOperation* allocatePendingOperation();

    using PendingOperations = CircularQueue<PendingOperation, DeviceVector>;
    __device__ PendingOperations& getPendingOperations() { return pendingOperations; }

    struct Holder {
        __device__ Holder(const Context& ctx);
        __device__ ~Holder();
    };

    const int peakClockKHz;

private:

    __device__ explicit ThreadPrivateState(const Context& ctx)
        : pendingOperations(ctx.pendingBufferSize)
        , peakClockKHz(ctx.peakClockKHz)
        , unusedCommunicationContext(0)
    {
        curand_init(0, 0, 0, &rand_state);
    }

    PendingOperations pendingOperations;
    
public:
    int unusedCommunicationContext;
    curandState_t rand_state;
    
    GlobalVarsStorage globalVarsStorage;
};

struct MessageDescriptor {
    int ctx = 0;
    int src = 0;
    int tag = 0;
    bool buffered = false;
    void* data = nullptr;
    volatile bool* done = nullptr;
};

struct SharedThreadState {
    SharedThreadState(int recvListSize)
        : unexpectedRecv(recvListSize)
        , expectedRecv(recvListSize)
    {}

    ManagedMemoryLock recvLock;
    CircularQueue<MessageDescriptor, ManagedVector> unexpectedRecv;
    CircularQueue<MessageDescriptor, ManagedVector> expectedRecv;
};

struct DeviceToHostCommunicator {
    struct Message {
        __host__ __device__ Message()
            : ptr(nullptr), size(0), threadRank(-1) {}

        __host__ __device__ Message(void* ptr, size_t size, int threadRank)
            : ptr(ptr), size(size), threadRank(threadRank) {}

        __host__ __device__ volatile Message& operator=(const Message& rhs) volatile {
            ptr = rhs.ptr;
            size = rhs.size;
            threadRank = rhs.threadRank;
            return *this;
        }

        void* ptr;
        size_t size;
        int threadRank;
    };

    DeviceToHostCommunicator(size_t queueSize, size_t numThreads);

    __device__ void delegateToHost(void* ptr, size_t size);

    template <typename F>
    void processIncomingMessages(F&& callback) {
        if (!lock.tryLock()) return;
        while (!queue.empty()) {
            Message* message = queue.head();
            callback(message->ptr, message->size, message->threadRank);
            cudaGlobalFence();
            hostFinished[message->threadRank] = true;
            queue.pop(message);
        }
        lock.unlock();
    }

    ManagedMemoryLock lock;
    CircularQueue<Message, ManagedVector> queue;

    // for each device thread store variable that says
    // if the host is finished processing device request
    ManagedVector<bool> hostFinished;
};

struct FreeManagedMemory {

    FreeManagedMemory(size_t size);

    __host__ __device__ void* allocate(size_t size);
    __host__ __device__ void free(void* ptr);

private:

    enum { FREE = 0, USED = 1 };

    struct BlockDescriptor {
        char status;
        size_t end;
    };

    __host__ __device__ void compactionWithNextBlocks(size_t currentBlock);

    ManagedMemoryLock lock;
    ManagedVector<char> buffer;
};

class SharedState {
public:
    struct Context {
        int numThreads{-1};
        int recvListSize{16};
        int deviceToHostQueueSize{128};
        int freeMemorySize{(1 << 20) * 512};

        bool valid() const {
            if (numThreads <= 0) return false;
            if (recvListSize <= 0) return false;
            if (deviceToHostQueueSize <= 0) return false;
            if (freeMemorySize <= 0) return false;
            return true;
        }
    };

private:
    SharedState(const Context& ctx)
        : sharedThreadState(ctx.numThreads, ctx.recvListSize)
        , deviceToHostCommunicator(ctx.deviceToHostQueueSize, ctx.numThreads)
        , freeManagedMemory(ctx.freeMemorySize)
        , numThreads(ctx.numThreads)
        , returnValue(0)
        , bar(0)
    {
    }

public:
    struct Holder {
        Holder(const Context& ctx) {
            assert(ctx.valid());
            CUDA_CHECK(cudaMallocManaged(&sharedState, sizeof(SharedState)));
            new (sharedState) SharedState(ctx);
            assert(sharedState);
        }
        ~Holder() {
            sharedState->~SharedState();
            CUDA_CHECK(cudaFree(sharedState));
        }
        SharedState* get() const { return sharedState; }
    private:
        SharedState* sharedState;
    };

    __device__ int gridRank() {
        return cooperative_groups::details::grid::thread_rank();
    }

    // activeGridSize <= totalGridSize
    __device__ int activeGridSize() {
        return numThreads;
    }

    // totalGridSize = blockSize * gridSize  
    __device__ int totalGridSize() {
        return cooperative_groups::details::grid::size();
    }

    // behaves exactly as this_grid().sync(), but without cooperative group kernel launch
    __device__ void gridBarrier() {
        cooperative_groups::details::grid::sync(&bar);
    }

    ManagedVector<SharedThreadState> sharedThreadState;
    //SharedFragmentBuffer sharedFragmentBuffer;

    DeviceToHostCommunicator deviceToHostCommunicator;

    FreeManagedMemory freeManagedMemory;

    int numThreads;
    int returnValue;
    unsigned int bar;
};

__device__ SharedState& sharedState();

__device__ void setSharedState(SharedState* sharedState);

__device__ ThreadPrivateState& threadPrivateState();

__device__ PendingOperation* isend(int dst, const void* data, int count, int ctx, int tag, bool synchronous = false, bool buffered = false);

__device__ PendingOperation* irecv(int src, void* data, int count, int ctx, int tag);

__device__ void progress();

__device__ bool test(PendingOperation* op);

__device__ void wait(PendingOperation* op);

__device__ bool testIO(PendingIOOperation* ioop);

__device__ void waitIO(PendingIOOperation* ioop);

__device__ void initialize();

} // namespace


#endif
