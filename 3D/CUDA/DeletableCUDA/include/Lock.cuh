//
// Created by Michael Staneker on 22.02.21.
//

#ifndef CUDA_LOCK_CUH
#define CUDA_LOCK_CUH

struct Lock{

    int *mutex;

    Lock() {
        int state = 0;
        cudaMalloc((void**)&mutex, sizeof(int));
        cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
    }

    ~Lock(){
        cudaFree(mutex);
    }

    __device__ void lock() {
        while (atomicCAS(mutex, 0 ,1) != 0);
    }

    __device__ void unlock() {
        atomicExch(mutex, 0);
    }
};

#endif //CUDA_LOCK_CUH
