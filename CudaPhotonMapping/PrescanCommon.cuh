#pragma once
#include <crt/host_defines.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "Defines.cuh"
#include "CudaGridSynchronizer.cuh";
#include "SharedMemory.cuh"
#include <cooperative_groups.h>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n)((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

template <typename T>
struct PrescanHelperStruct {
    T* out_arr = nullptr;
    T* in_arr = nullptr;
    T* separated_sums_arr = nullptr;
    T* united_sums_arr = nullptr;
    __device__ PrescanHelperStruct() :
        out_arr(nullptr), in_arr(nullptr), separated_sums_arr(nullptr), united_sums_arr(nullptr)
    {}
    __host__ __device__ PrescanHelperStruct(T* out_arr, T* in_arr,
        T* separated_sums_arr, T* united_sums_arr) :
        out_arr(out_arr), in_arr(in_arr),
        separated_sums_arr(separated_sums_arr), united_sums_arr(united_sums_arr)
    { }
    __host__ __device__ PrescanHelperStruct(const PrescanHelperStruct<T>& other) {
        out_arr = other.out_arr;
        in_arr = other.in_arr;
        separated_sums_arr = other.separated_sums_arr;
        united_sums_arr = other.united_sums_arr;
    }
    __device__ void init_particaly() {
        separated_sums_arr = new T[PRESCAN_BLOCKS];
        united_sums_arr = new T[PRESCAN_BLOCKS];
    }
    __device__ void free_pointers() volatile {
        free(out_arr);
        free(in_arr);
        /*free(separated_sums_arr);
        free(united_sums_arr);*/
    }

    __device__ volatile PrescanHelperStruct<T>& operator=(const PrescanHelperStruct<T>& other) volatile {
        out_arr = other.out_arr;
        in_arr = other.in_arr;
        separated_sums_arr = other.separated_sums_arr;
        united_sums_arr = other.united_sums_arr;

        return *this;
    }
};

template <typename T>
__device__ PrescanHelperStruct<T> get_prescan_helper_struct(T* in_arr, idxtype length, T* global_sum_offset) {
    T* out_arr = new T[length];
    T* separated_sums_arr = new T[PRESCAN_BLOCKS];
    T* united_sums_arr = new T[PRESCAN_BLOCKS];
    *global_sum_offset = T();
    return { out_arr, in_arr, separated_sums_arr, united_sums_arr };
}