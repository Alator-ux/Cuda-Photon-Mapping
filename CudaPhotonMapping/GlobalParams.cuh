#pragma once
#include <crt/host_defines.h>
#include "Defines.cuh"

namespace GlobalParams {
    /* ===== Refractive index ===== */
    extern float cpu_default_refractive_index;
    extern __constant__ float gpu_default_refractive_index;

    __host__ __device__ __forceinline__ float inline default_refractive_index() {
#ifdef __CUDA_ARCH__
        return gpu_default_refractive_index;
#else
        return cpu_default_refractive_index;
#endif  __CUDA_ARCH__
    }

    __host__ void set_default_refractive_index(float value);

    /* ===== Max depth ===== */
    extern int cpu_max_depth;
    extern __constant__ int gpu_max_depth;

    __host__ __device__ __forceinline__ int inline max_depth() {
#ifdef __CUDA_ARCH__
        return gpu_max_depth;
#else
        return cpu_max_depth;
#endif
    }

    __host__ void set_max_depth(int value);
}