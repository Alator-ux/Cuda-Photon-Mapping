#pragma once
#include <crt/host_defines.h>
#include "Defines.cuh"

/* MEDIUM MANAGER */

namespace MManager {
    struct MediumContent {
        float refractive_index;
        int   hit_id;
        //unsigned int inside_level;
    };
}


/* Inner MM Stack */
#define MMInnerData MManager::MediumContent

extern MMInnerData* cpu_mm_inner_data;
extern __constant__ MMInnerData* gpu_mm_inner_data;

__host__ __device__ __forceinline__ MMInnerData* mm_inner_data() {
#ifdef __CUDA_ARCH__
    return gpu_mm_inner_data;
#else
    return cpu_mm_inner_data;
#endif
}

extern idxtype cpu_mm_inner_capacity;
extern __constant__ idxtype gpu_mm_inner_capacity;

__host__ __device__ __forceinline__ idxtype mm_inner_capacity() {
#ifdef __CUDA_ARCH__
    return gpu_mm_inner_capacity;
#else
    return cpu_mm_inner_capacity;
#endif
}

__host__ void set_medium_content_parameters(MMInnerData* cpu_mm_inner_data_val, MMInnerData* gpu_mm_inner_data_val, idxtype mm_inner_capacity_val);
