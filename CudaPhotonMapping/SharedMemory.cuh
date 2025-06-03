#pragma once
#include <cuda_runtime.h>
#include "Defines.cuh"

template <typename T>
struct SharedMem {
};

template <>
struct SharedMem<idxtype> {
    __device__ idxtype* getPointer() {
        extern __shared__ idxtype shared_idxtype_mem[];
        return shared_idxtype_mem;
    }
};

template <>
struct SharedMem<int> {
    __device__ int* getPointer() {
        extern __shared__ int shared_int_mem[];
        return shared_int_mem;
    }
};

// Add your own type versions if you need